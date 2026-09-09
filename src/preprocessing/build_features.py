"""
Monta o dataset de modelagem (grão: município x ano, rede municipal) a
partir da camada Gold/Silver.

Variável-alvo
-------------
`alvo_alfabetizado` (binário): reaproveita a visão Gold
`alfabetizacao_por_municipio` (status_meta_2025), que compara a taxa de
alfabetização observada no município com a meta municipal de 2025 do
Compromisso Nacional Criança Alfabetizada (PNA). 1 = ATINGIU a meta
(proxy de "alfabetização adequada"), 0 = NAO_ATINGIU.

Como a base pública (INEP/ANA) não disponibiliza microdados por aluno
(dado protegido por sigilo estatístico/LGPD), o grão adotado é
município-ano-rede municipal — a menor granularidade disponível na
camada Gold da Fase 2 — e serve como proxy operacional para a pergunta
"o aluno será considerado alfabetizado?". Essa decisão está documentada
no README (seção "Objetivo analítico" e "Limitações").

Tratamento de data leakage
---------------------------
As colunas abaixo são EXCLUÍDAS do conjunto de features por serem
transformações quase-determinísticas da própria métrica usada para
construir o alvo (taxa_alfabetizacao):
  - taxa_alfabetizacao / gap_meta_2025   -> definem o alvo diretamente
  - nivel_alfabetizacao                  -> bucket discreto de taxa_alfabetizacao
                                             (verificado: nivel 5 == ATINGIU em 100% dos casos)
  - proporcao_aluno_nivel_5..8           -> soma tem corr. de 0.98 com taxa_alfabetizacao
                                             (é a mesma medida decomposta por nível SAEB)
  - serie                                -> constante (=2) em toda a base, sem variância
  - meta_alfabetizacao_2030              -> constante (=80.0) em toda a base, sem variância
"""

import logging

import pandas as pd

from src.etl.paths import GOLD_DIR, SILVER_DIR
from src.etl.uf_lookup import IBGE_UF, REGIAO_UF

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
log = logging.getLogger(__name__)


def _loo_media_uf(df: pd.DataFrame, valor_col: str, grupo_cols: list[str]) -> pd.Series:
    """Média leave-one-out por grupo: exclui a própria linha do cálculo da média
    dos pares (evita vazar a métrica do próprio registro na feature de contexto)."""
    soma = df.groupby(grupo_cols)[valor_col].transform("sum")
    cnt = df.groupby(grupo_cols)[valor_col].transform("count")
    return ((soma - df[valor_col]) / (cnt - 1)).round(2)


def build() -> pd.DataFrame:
    df_gold = pd.read_parquet(GOLD_DIR / "alfabetizacao_por_municipio.parquet")
    df_ind = pd.read_parquet(SILVER_DIR / "pass" / "indicador_municipio.parquet")
    df_ind = df_ind[df_ind["rede"] == "3"][["id_municipio", "ano", "media_portugues", "taxa_alfabetizacao"]]
    df_meta_uf = pd.read_parquet(SILVER_DIR / "pass" / "meta_uf.parquet")

    # 1) Base + alvo (descarta registros sem meta municipal conhecida -> alvo indefinido)
    df = df_gold.dropna(subset=["status_meta_2025"]).copy()
    df["alvo_alfabetizado"] = (df["status_meta_2025"] == "ATINGIU").astype(int)

    # 2) Território
    df["sigla_uf"] = df["id_municipio"].str[:2].map(IBGE_UF)
    df["regiao"] = df["sigla_uf"].map(REGIAO_UF)

    # 3) Meta estadual (feature publicada, conhecida a priori — sem leakage)
    df_meta_uf_2025 = df_meta_uf[["ano", "sigla_uf", "meta_alfabetizacao_2025"]].rename(
        columns={"meta_alfabetizacao_2025": "meta_uf_2025"}
    )
    df = df.merge(df_meta_uf_2025, on=["ano", "sigla_uf"], how="left")

    # 4) Contexto regional: média estadual leave-one-out da taxa dos municípios
    #    pares (mesma UF + ano), excluindo o próprio município do cálculo.
    df_taxa = df_ind[["id_municipio", "ano", "taxa_alfabetizacao"]].copy()
    df_taxa["sigla_uf"] = df_taxa["id_municipio"].str[:2].map(IBGE_UF)
    df_taxa["taxa_media_uf_loo"] = _loo_media_uf(df_taxa, "taxa_alfabetizacao", ["sigla_uf", "ano"])
    df = df.merge(
        df_taxa[["id_municipio", "ano", "taxa_media_uf_loo"]], on=["id_municipio", "ano"], how="left"
    )

    # 5) Features acadêmicas distintas do alvo (média Português — medida separada,
    #    correlacionada mas não uma transformação direta de taxa_alfabetizacao)
    df = df.merge(
        df_ind[["id_municipio", "ano", "media_portugues"]].drop_duplicates(),
        on=["id_municipio", "ano"], how="left", suffixes=("", "_dup"),
    )

    colunas_finais = [
        "id_municipio", "ano", "sigla_uf", "regiao",
        "media_portugues", "meta_2025", "meta_uf_2025", "taxa_media_uf_loo",
        "alvo_alfabetizado",
    ]
    df_final = df[colunas_finais].drop_duplicates(subset=["id_municipio", "ano"]).reset_index(drop=True)

    log.info(f"[FEATURES] dataset final: {df_final.shape[0]} linhas x {df_final.shape[1]} colunas")
    log.info(f"[FEATURES] distribuição do alvo:\n{df_final['alvo_alfabetizado'].value_counts(normalize=True)}")
    return df_final


if __name__ == "__main__":
    out = build()
    dest = GOLD_DIR.parent / "features" / "dataset_modelagem.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dest, index=False)
    log.info(f"[FEATURES] salvo em {dest}")
