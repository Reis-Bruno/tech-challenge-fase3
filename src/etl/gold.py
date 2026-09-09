"""
Camada Gold (SPEC) — réplica em pandas do job `glue_jobs/etl_gold.py`
da Fase 2: gera as 4 visões analíticas a partir da Silver (pass).
"""

import logging
from datetime import datetime, timezone

import pandas as pd

from .paths import GOLD_DIR, SILVER_DIR
from .uf_lookup import IBGE_UF

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
log = logging.getLogger(__name__)


def _pass_(entidade: str) -> pd.DataFrame:
    return pd.read_parquet(SILVER_DIR / "pass" / f"{entidade}.parquet")


def _salvar(df: pd.DataFrame, visao: str, ingestion_ts: str, ingestion_date: str) -> int:
    df = df.copy()
    df["_gold_processed_at"] = ingestion_ts
    df["_ingestion_date"] = ingestion_date
    destino = GOLD_DIR / f"{visao}.parquet"
    df.to_parquet(destino, index=False)
    log.info(f"[GOLD] {visao}: {len(df)} registros -> {destino}")
    return len(df)


def gold_alfabetizacao_municipio(df_mun: pd.DataFrame, df_meta_mun: pd.DataFrame,
                                  ts: str, dt: str) -> int:
    log.info("[GOLD] Gerando: alfabetizacao_por_municipio")

    df_meta = df_meta_mun[
        ["id_municipio", "ano", "meta_alfabetizacao_2025", "meta_alfabetizacao_2030", "nivel_alfabetizacao"]
    ].rename(columns={"meta_alfabetizacao_2025": "meta_2025", "meta_alfabetizacao_2030": "meta_2030"})

    df = df_mun[df_mun["rede"] == "3"].merge(df_meta, on=["id_municipio", "ano"], how="left")
    df = df[
        ["id_municipio", "ano", "serie", "rede_desc", "taxa_alfabetizacao", "media_portugues",
         "meta_2025", "meta_2030", "nivel_alfabetizacao"]
    ].rename(columns={"rede_desc": "rede"})

    df["gap_meta_2025"] = (df["taxa_alfabetizacao"] - df["meta_2025"]).round(2)
    df["status_meta_2025"] = df.apply(
        lambda r: "ATINGIU" if pd.notna(r["taxa_alfabetizacao"]) and pd.notna(r["meta_2025"])
        and r["taxa_alfabetizacao"] >= r["meta_2025"] else "NAO_ATINGIU", axis=1,
    )
    # Onde a meta é desconhecida, o status também é desconhecido (evita rótulo artificial)
    df.loc[df["meta_2025"].isna(), "status_meta_2025"] = pd.NA

    return _salvar(df, "alfabetizacao_por_municipio", ts, dt)


def gold_evolucao_temporal(df_uf: pd.DataFrame, ts: str, dt: str) -> int:
    log.info("[GOLD] Gerando: evolucao_temporal")

    grp = df_uf[df_uf["rede"] == "3"].groupby(["sigla_uf", "ano", "serie"], as_index=False)
    df = grp.agg(
        taxa_media=("taxa_alfabetizacao", "mean"),
        taxa_min=("taxa_alfabetizacao", "min"),
        taxa_max=("taxa_alfabetizacao", "max"),
        taxa_desvio=("taxa_alfabetizacao", "std"),
        media_portugues_media=("media_portugues", "mean"),
    )
    for c in ["taxa_media", "taxa_min", "taxa_max", "taxa_desvio", "media_portugues_media"]:
        df[c] = df[c].round(2)
    df = df.sort_values(["sigla_uf", "ano"]).reset_index(drop=True)

    return _salvar(df, "evolucao_temporal", ts, dt)


def gold_ranking_municipios(df_mun: pd.DataFrame, ts: str, dt: str) -> int:
    log.info("[GOLD] Gerando: ranking_municipios")

    df = df_mun[df_mun["rede"] == "3"].copy()
    df["sigla_uf"] = df["id_municipio"].str[:2].map(IBGE_UF)
    df["ranking_uf"] = (
        df.groupby(["sigla_uf", "ano"])["taxa_alfabetizacao"]
        .rank(method="min", ascending=False)
        .astype("Int64")
    )
    df = df[
        ["id_municipio", "sigla_uf", "ano", "serie", "rede_desc", "taxa_alfabetizacao",
         "media_portugues", "ranking_uf"]
    ].rename(columns={"rede_desc": "rede"})

    return _salvar(df, "ranking_municipios", ts, dt)


def gold_comparacao_metas(df_uf: pd.DataFrame, df_meta_br: pd.DataFrame,
                           df_meta_uf: pd.DataFrame, ts: str, dt: str) -> int:
    log.info("[GOLD] Gerando: comparacao_metas_nacionais")

    df_media_nacional = (
        df_uf[df_uf["rede"] == "0"]
        .groupby(["sigla_uf", "ano"], as_index=False)
        .agg(taxa_uf=("taxa_alfabetizacao", "mean"))
    )
    df_media_nacional["taxa_uf"] = df_media_nacional["taxa_uf"].round(2)

    df_meta_ano = df_meta_br[["ano", "meta_alfabetizacao_2025", "meta_alfabetizacao_2030"]].rename(
        columns={"meta_alfabetizacao_2025": "meta_nacional_2025", "meta_alfabetizacao_2030": "meta_nacional_2030"}
    )
    df_muf = df_meta_uf[["ano", "sigla_uf", "meta_alfabetizacao_2025"]].rename(
        columns={"meta_alfabetizacao_2025": "meta_uf_2025"}
    )

    df = df_media_nacional.merge(df_meta_ano, on="ano", how="left").merge(
        df_muf, on=["ano", "sigla_uf"], how="left"
    )
    df["gap_meta_nacional"] = (df["taxa_uf"] - df["meta_nacional_2025"]).round(2)
    df["gap_meta_uf"] = (df["taxa_uf"] - df["meta_uf_2025"]).round(2)
    df["status_meta"] = df.apply(
        lambda r: "ATINGIU" if pd.notna(r["taxa_uf"]) and pd.notna(r["meta_nacional_2025"])
        and r["taxa_uf"] >= r["meta_nacional_2025"] else "NAO_ATINGIU", axis=1,
    )
    df.loc[df["meta_nacional_2025"].isna(), "status_meta"] = pd.NA
    df = df.sort_values(["ano", "taxa_uf"]).reset_index(drop=True)

    return _salvar(df, "comparacao_metas_nacionais", ts, dt)


def run() -> dict[str, int]:
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dt = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    df_mun = _pass_("indicador_municipio")
    df_uf = _pass_("indicador_uf")
    df_meta_br = _pass_("meta_brasil")
    df_meta_uf = _pass_("meta_uf")
    df_meta_mun = _pass_("meta_municipio")

    resultados = {
        "alfabetizacao_por_municipio": gold_alfabetizacao_municipio(df_mun, df_meta_mun, ts, dt),
        "evolucao_temporal": gold_evolucao_temporal(df_uf, ts, dt),
        "ranking_municipios": gold_ranking_municipios(df_mun, ts, dt),
        "comparacao_metas_nacionais": gold_comparacao_metas(df_uf, df_meta_br, df_meta_uf, ts, dt),
    }

    log.info("=" * 65)
    log.info("SUMÁRIO GOLD")
    for visao, total in resultados.items():
        log.info(f"  {visao:<40}: {total} registros")
    log.info("=" * 65)
    return resultados


if __name__ == "__main__":
    run()
