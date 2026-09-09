# Roteiro — Vídeo Executivo (até 5 minutos)

Formato sugerido pelo enunciado: simular uma reunião executiva com gestores públicos, lideranças
ou stakeholders do PNA (Compromisso Nacional Criança Alfabetizada). Este roteiro é um guia de
apoio para a gravação — **a gravação em si deve ser feita pelo grupo**.

## 0. Preparação (antes de gravar)

- Ter abertos: `README.md`, `images/eda_taxa_por_regiao.png`, `images/overfitting_temporal.png`,
  `images/feature_importance.png`, `images/shap_summary.png`, e a lista de municípios de maior
  risco (saída da seção 6 de `notebooks/02_modelagem.ipynb`).
- Tom: reunião executiva — direto ao ponto, foco em decisão, não em detalhe técnico de código.

## 1. Abertura e problema (≈45s)

> "A alfabetização na idade certa é um dos indicadores mais importantes do desenvolvimento
> educacional do Brasil. O Compromisso Nacional Criança Alfabetizada estabelece metas por
> município até 2030 — mas hoje só sabemos se um município atingiu a meta *depois* do resultado
> sair. Nosso objetivo foi construir um modelo que **antecipe esse risco**, usando dados que já
> temos disponíveis hoje."

Mostrar: slide/tela com o mapa de risco por região (`eda_taxa_por_regiao.png` ou
`risco_por_regiao.png`).

## 2. O que os dados mostram (≈60s)

> "Analisamos os indicadores de alfabetização de quase 11 mil municípios em 2023 e 2024. O
> primeiro achado é claro: existe uma desigualdade regional muito forte. A probabilidade média de
> um município atingir sua meta é de apenas 6% no Norte e 21% no Nordeste — contra 36% no
> Centro-Oeste e 45% no Sul. Isso não é surpresa para quem acompanha educação no Brasil, mas agora
> temos isso quantificado e podemos usar para priorizar."

Mostrar: gráfico de risco por região.

## 3. O modelo e por que ele é confiável (≈90s)

> "Construímos um modelo de machine learning treinado com dados de 2023 e testado — sem nenhum
> ajuste — em dados de 2024, simulando exatamente o cenário real: prever o futuro com base no
> passado. Testamos três abordagens diferentes. As mais sofisticadas, baseadas em árvores de
> decisão, pareciam ótimas nos testes internos, mas na prática, ao serem levadas para o ano
> seguinte, erravam muito mais. O modelo mais simples — uma regressão logística — generalizou
> melhor: 82% de área sob a curva ROC, uma métrica que resume a capacidade do modelo de separar
> corretamente municípios em risco dos que não estão.
>
> Esse é um ponto importante para vocês, como gestores: escolhemos o modelo que funciona melhor
> no mundo real, não o que tem o número mais bonito no laboratório."

Mostrar: gráfico `overfitting_temporal.png`.

## 4. O que mais pesa na previsão (≈60s)

> "O fator que mais influencia a previsão é a proficiência em Português medida em avaliações
> intermediárias — o que já é coletado hoje. Em seguida, a localização geográfica do município
> importa bastante, confirmando o padrão regional que vimos antes. Isso significa que já temos, em
> mãos, os dados necessários para gerar esse alerta — não é preciso nenhuma nova coleta."

Mostrar: `feature_importance.png` e/ou `shap_summary.png`.

## 5. Valor estratégico e aplicação (≈60s)

> "Na prática, isso permite três coisas: primeiro, priorizar a alocação de recursos — professores,
> materiais, apoio técnico — para os municípios com maior probabilidade prevista de não atingir a
> meta, antes do resultado final sair. Segundo, adaptar a estratégia por região, já que o padrão de
> risco não é uniforme no país. Terceiro, usar quedas na proficiência em Português como um alerta
> precoce, sem esperar o ciclo de avaliação fechar."

Mostrar: lista dos municípios de maior risco (tabela do notebook 02).

## 6. Limites e próximos passos (≈45s)

> "É importante ser transparente: este modelo trabalha no nível do município, não do aluno
> individual — o Brasil não publica esse dado por questões de privacidade. E ele identifica risco,
> não substitui a decisão do gestor. Os próximos passos incluem incorporar dados do IBGE e do
> Censo Escolar, e ampliar a janela histórica conforme mais anos de dados forem publicados."

## 7. Encerramento (≈15s)

> "Mais do que um modelo, entregamos um processo replicável de inteligência aplicada à
> alfabetização — pronto para apoiar decisões reais de política pública."

---

**Duração total estimada:** ~4min45s (dentro do limite de 5 minutos).
