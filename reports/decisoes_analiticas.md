# Decisões Analíticas — Registro Formal

Este documento registra, de forma centralizada, as decisões de maior impacto tomadas ao longo do
projeto e sua justificativa. As análises que fundamentam cada decisão estão detalhadas nos
notebooks `notebooks/01_analise_exploratoria.ipynb` e `notebooks/02_modelagem.ipynb`.

## 1. Reconstrução local da camada Gold (sem acesso à AWS da Fase 2)

**Decisão:** reimplementar em pandas (`src/etl/bronze.py`, `silver.py`, `gold.py`) a mesma lógica
dos jobs PySpark/Glue da Fase 2, a partir dos 5 CSVs brutos do INEP hospedados em
`Luodev/1IAST-Fase2/dados/`.

**Motivo:** a conta AWS Learner Lab usada na Fase 2 é efêmera (ambiente de laboratório acadêmico)
e não está mais acessível. Reconstruir localmente, replicando fielmente as regras de negócio
originais (mesmos filtros de rede, mesmas fórmulas de gap/status, mesmo mapeamento IBGE→UF),
garante consistência analítica entre as duas fases sem exigir infraestrutura AWS.

**Verificação:** os totais de registros processados (23.995 município, 145 UF, 10.704 metas
municipais) batem exatamente com os volumes documentados no README da Fase 2.

## 2. Definição da variável-alvo em nível municipal (proxy de "aluno alfabetizado")

**Decisão:** `alvo_alfabetizado = 1` se `taxa_alfabetizacao >= meta_alfabetizacao_2025` do
próprio município (reaproveita a coluna `status_meta_2025` já calculada na visão Gold
`alfabetizacao_por_municipio`); `0` caso contrário.

**Motivo:** o INEP não publica microdados por aluno (sigilo estatístico + LGPD). A menor
granularidade pública disponível é município-ano-rede. Como a visão Gold já operacionaliza a
comparação com a meta do PNA — o mecanismo real usado pelo Ministério da Educação para
monitorar progresso — este projeto adota a mesma definição, mantendo rastreabilidade com a Fase 2
e alinhamento direto com a pergunta de negócio "quais municípios podem não atingir metas futuras?".

**Desvio deliberado em relação ao job original da Fase 2:** no job PySpark original, quando
`meta_alfabetizacao_2025` é nula, a expressão `F.when(...).otherwise("NAO_ATINGIU")` do Spark
classifica o registro como `NAO_ATINGIU` mesmo sem meta conhecida (comportamento padrão do
`.otherwise()` do Spark para condições nulas). Nesta Fase 3, tratamos explicitamente esse caso
como alvo **indefinido** (`NaN`) e descartamos os 242 registros afetados (2,2% da base) — rotular
como `NAO_ATINGIU` sem meta conhecida introduziria ruído artificial no alvo. Essa correção está
implementada em `src/etl/gold.py::gold_alfabetizacao_municipio`.

## 3. Exclusão de features por data leakage

Ver tabela completa no README (seção "Etapas de modelagem"). Resumo das evidências quantitativas
que fundamentaram cada exclusão (notebook 01, seção 4):

- `nivel_alfabetizacao`: nível 5 corresponde a 100% dos casos com `status_meta_2025 = ATINGIU`
  na amostra — é um bucket discreto de `taxa_alfabetizacao`, não uma medida independente.
- `proporcao_aluno_nivel_5` a `_8`: a soma dessas proporções tem correlação de **0,9855** com
  `taxa_alfabetizacao` — estatisticamente indistinguível da própria métrica-alvo.
- `serie` e `meta_alfabetizacao_2030`: variância zero em toda a base — removidas por não
  agregarem informação, não por risco de vazamento.

## 4. Engenharia de atributos: média estadual *leave-one-out*

**Decisão:** a feature `taxa_media_uf_loo` (contexto regional) é calculada excluindo o próprio
município do cálculo da média estadual: `(soma_uf - taxa_propria) / (contagem_uf - 1)`.

**Motivo:** usar a média estadual "ingênua" (incluindo o próprio município) introduziria uma
forma sutil de vazamento — o valor da própria observação contaminando sua feature de contexto,
mais perceptível em estados com poucos municípios. A versão *leave-one-out* é o padrão correto
para features agregadas de grupo em modelagem preditiva.

## 5. Split temporal como estratégia principal de validação

**Decisão:** treino = ano 2023, holdout = ano 2024 (nunca visto durante tuning), com
`GridSearchCV` + `StratifiedKFold` (5 folds) aplicado apenas dentro do treino de 2023.

**Motivo:** o objetivo de negócio é prever o **futuro** ("municípios que podem não atingir metas
futuras"), não apenas interpolar dentro de um período já observado. A validação cruzada dentro de
um único ano superestimou fortemente o desempenho de modelos de árvore (F1 ≈ 0,98), que
depois generalizaram mal para 2024 (F1 ≈ 0,33–0,39) — validando empiricamente a necessidade do
split temporal como critério de seleção final do modelo, e não apenas o score de CV.

## 6. Seleção do modelo final por métrica de holdout, não de validação cruzada

**Decisão:** o modelo final (Regressão Logística) foi selecionado pelo ROC-AUC no holdout real de
2024 (0,819), apesar de Random Forest e Gradient Boosting apresentarem F1 de CV muito superior
durante o treino (0,98 vs. 0,80).

**Motivo:** métrica de CV dentro do período de treino mede apenas a capacidade de interpolação
dentro da mesma distribuição temporal; métrica de holdout fora do tempo mede a capacidade real de
generalização, que é o que importa para uso em produção/política pública.
