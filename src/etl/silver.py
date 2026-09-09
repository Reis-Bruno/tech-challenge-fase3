"""
Camada Silver (SOT) — réplica em pandas do job `glue_jobs/etl_silver.py`
da Fase 2: decodifica códigos de rede, arredonda taxas/metas, aplica
regras de qualidade (DQ) por linha e separa PASS/QUARENTENA.
"""

import logging
from datetime import datetime, timezone

import pandas as pd

from .paths import BRONZE_DIR, SILVER_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
log = logging.getLogger(__name__)

# Códigos de rede INEP (dados reais do INEP/Base dos Dados)
REDE_MAP = {"0": "total", "2": "estadual", "3": "municipal", "5": "privada"}


def _transformar_indicador(df: pd.DataFrame, ingestion_ts: str) -> pd.DataFrame:
    df = df.copy()
    df["rede_desc"] = df["rede"].map(REDE_MAP).fillna(df["rede"])
    df["taxa_alfabetizacao"] = df["taxa_alfabetizacao"].round(2)
    df["media_portugues"] = df["media_portugues"].round(2)
    df["_silver_processed_at"] = ingestion_ts
    return df


def _transformar_meta(df: pd.DataFrame, ingestion_ts: str) -> pd.DataFrame:
    df = df.copy()
    meta_cols = [c for c in df.columns if c.startswith("meta_alfabetizacao_")]
    for c in meta_cols:
        df[c] = df[c].round(2)
    df["_silver_processed_at"] = ingestion_ts
    return df


TRANSFORMACOES = {
    "indicador_municipio": _transformar_indicador,
    "indicador_uf": _transformar_indicador,
    "meta_brasil": _transformar_meta,
    "meta_uf": _transformar_meta,
    "meta_municipio": _transformar_meta,
}


def _dq_indicador_municipio(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_dq_id_municipio_valido"] = df["id_municipio"].notna() & (df["id_municipio"].str.len() == 7)
    df["_dq_taxa_valida"] = df["taxa_alfabetizacao"].notna() & df["taxa_alfabetizacao"].between(0, 100)
    df["_dq_ano_valido"] = df["ano"].notna() & df["ano"].between(2020, 2030)
    df["_dq_passou"] = df["_dq_id_municipio_valido"] & df["_dq_taxa_valida"] & df["_dq_ano_valido"]
    return df


def _dq_indicador_uf(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_dq_uf_valida"] = df["sigla_uf"].notna() & (df["sigla_uf"].str.len() == 2)
    df["_dq_taxa_valida"] = df["taxa_alfabetizacao"].notna() & df["taxa_alfabetizacao"].between(0, 100)
    df["_dq_ano_valido"] = df["ano"].notna() & df["ano"].between(2020, 2030)
    df["_dq_passou"] = df["_dq_uf_valida"] & df["_dq_ano_valido"]
    return df


def _dq_meta(df: pd.DataFrame, chave_col: str) -> pd.DataFrame:
    df = df.copy()
    df["_dq_chave_valida"] = df[chave_col].notna()
    df["_dq_ano_valido"] = df["ano"].notna() & df["ano"].between(2020, 2030)
    df["_dq_passou"] = df["_dq_chave_valida"] & df["_dq_ano_valido"]
    return df


DQ_FUNCTIONS = {
    "indicador_municipio": _dq_indicador_municipio,
    "indicador_uf": _dq_indicador_uf,
    "meta_brasil": lambda df: _dq_meta(df, "ano"),
    "meta_uf": lambda df: _dq_meta(df, "sigla_uf"),
    "meta_municipio": lambda df: _dq_meta(df, "id_municipio"),
}

MOTIVOS_DQ = {
    "indicador_municipio": [
        ("_dq_id_municipio_valido", "id_municipio ausente ou inválido (esperado 7 dígitos)"),
        ("_dq_taxa_valida", "taxa_alfabetizacao fora de [0,100] ou nula"),
        ("_dq_ano_valido", "ano fora de [2020,2030] ou nulo"),
    ],
    "indicador_uf": [
        ("_dq_uf_valida", "sigla_uf ausente ou inválida (esperado 2 chars)"),
        ("_dq_ano_valido", "ano fora de [2020,2030] ou nulo"),
    ],
    "meta_brasil": [("_dq_ano_valido", "ano fora de [2020,2030] ou nulo")],
    "meta_uf": [("_dq_chave_valida", "sigla_uf nula"), ("_dq_ano_valido", "ano inválido")],
    "meta_municipio": [("_dq_chave_valida", "id_municipio nulo"), ("_dq_ano_valido", "ano inválido")],
}


def _separar_e_salvar(df: pd.DataFrame, entidade: str, ingestion_ts: str) -> tuple[int, int]:
    motivos = MOTIVOS_DQ[entidade]

    df_pass = df[df["_dq_passou"]].copy()
    df_quar = df[~df["_dq_passou"]].copy()

    def _motivo(row):
        return ", ".join(desc for col_dq, desc in motivos if not row[col_dq])

    if len(df_quar):
        df_quar["_quarentena_motivo"] = df_quar.apply(_motivo, axis=1)
        df_quar["_quarentena_ts"] = ingestion_ts

    SILVER_DIR.joinpath("pass").mkdir(parents=True, exist_ok=True)
    SILVER_DIR.joinpath("quarentena").mkdir(parents=True, exist_ok=True)
    df_pass.to_parquet(SILVER_DIR / "pass" / f"{entidade}.parquet", index=False)
    df_quar.to_parquet(SILVER_DIR / "quarentena" / f"{entidade}.parquet", index=False)

    n_pass, n_quar = len(df_pass), len(df_quar)
    total = n_pass + n_quar
    pct = round(n_pass / total * 100, 1) if total else 0
    log.info(f"[SILVER] {entidade}: pass={n_pass} | quarentena={n_quar} | score={pct}%")
    return n_pass, n_quar


def run() -> dict[str, tuple[int, int]]:
    SILVER_DIR.mkdir(parents=True, exist_ok=True)
    ingestion_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    resultados = {}
    for entidade, transformar in TRANSFORMACOES.items():
        log.info(f"[SILVER] Iniciando: {entidade}")
        df = pd.read_parquet(BRONZE_DIR / f"{entidade}.parquet")
        df_transf = transformar(df, ingestion_ts)
        df_dq = DQ_FUNCTIONS[entidade](df_transf)
        resultados[entidade] = _separar_e_salvar(df_dq, entidade, ingestion_ts)

    log.info("=" * 65)
    log.info("SUMÁRIO SILVER")
    log.info(f"  {'Entidade':<30} {'PASS':>8} {'QUARENTENA':>12}")
    for ent, (n_pass, n_quar) in resultados.items():
        log.info(f"  {ent:<30} {n_pass:>8} {n_quar:>12}")
    log.info("=" * 65)
    return resultados


if __name__ == "__main__":
    run()
