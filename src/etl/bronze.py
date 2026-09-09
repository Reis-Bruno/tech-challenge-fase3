"""
Camada Bronze (SOR) — réplica em pandas do job `glue_jobs/etl_bronze.py`
do Tech Challenge Fase 2 (Luodev/1IAST-Fase2).

Lê os CSVs brutos do INEP, aplica schema explícito, calcula hash de
deduplicação (_record_hash), enriquece com metadados de ingestão e roda
checks de qualidade (DQ) equivalentes aos do job original em Glue/Spark.

Como este projeto não tem acesso à conta AWS Learner Lab da Fase 2,
a camada Gold é reconstruída localmente a partir dos mesmos CSVs fonte,
seguindo a mesma lógica Bronze -> Silver -> Gold documentada no
repositório da Fase 2, para garantir consistência analítica entre fases.
"""

import hashlib
import logging
from datetime import datetime, timezone

import pandas as pd

from .paths import ARQUIVOS_RAW, BRONZE_DIR, RAW_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
log = logging.getLogger(__name__)

DTYPES = {
    "indicador_municipio": {
        "ano": "Int64", "id_municipio": "string", "serie": "Int64", "rede": "string",
        "taxa_alfabetizacao": "float64", "media_portugues": "float64",
        **{f"proporcao_aluno_nivel_{i}": "float64" for i in range(9)},
    },
    "indicador_uf": {
        "ano": "Int64", "sigla_uf": "string", "serie": "Int64", "rede": "string",
        "taxa_alfabetizacao": "float64", "media_portugues": "float64",
        **{f"proporcao_aluno_nivel_{i}": "float64" for i in range(9)},
    },
    "meta_brasil": {
        "ano": "Int64", "rede": "string", "taxa_alfabetizacao": "float64",
        **{f"meta_alfabetizacao_{y}": "float64" for y in range(2024, 2031)},
        "percentual_participacao": "float64",
    },
    "meta_uf": {
        "ano": "Int64", "sigla_uf": "string", "rede": "string", "taxa_alfabetizacao": "float64",
        **{f"meta_alfabetizacao_{y}": "float64" for y in range(2024, 2031)},
        "percentual_participacao": "float64",
    },
    "meta_municipio": {
        "ano": "Int64", "id_municipio": "string", "rede": "string", "taxa_alfabetizacao": "float64",
        **{f"meta_alfabetizacao_{y}": "float64" for y in range(2024, 2031)},
        "nivel_alfabetizacao": "string", "percentual_participacao": "float64",
    },
}

HASH_KEYS = {
    "indicador_municipio": ["ano", "id_municipio", "serie", "rede"],
    "indicador_uf": ["ano", "sigla_uf", "serie", "rede"],
    "meta_brasil": ["ano", "rede"],
    "meta_uf": ["ano", "sigla_uf", "rede"],
    "meta_municipio": ["ano", "id_municipio", "rede"],
}

CHECKS = {
    "indicador_municipio": [
        {"col": "id_municipio", "tipo": "not_null", "nivel": "FAIL"},
        {"col": "ano", "tipo": "not_null", "nivel": "FAIL"},
        {"col": "taxa_alfabetizacao", "tipo": "not_null", "nivel": "FAIL"},
        {"col": "taxa_alfabetizacao", "tipo": "range", "min": 0, "max": 100, "nivel": "FAIL"},
        {"col": "rede", "tipo": "in_set", "valores": {"0", "2", "3", "5"}, "nivel": "WARN"},
    ],
    "indicador_uf": [
        {"col": "sigla_uf", "tipo": "not_null", "nivel": "FAIL"},
        {"col": "ano", "tipo": "not_null", "nivel": "FAIL"},
        {"col": "taxa_alfabetizacao", "tipo": "not_null", "nivel": "WARN"},
        {"col": "taxa_alfabetizacao", "tipo": "range", "min": 0, "max": 100, "nivel": "WARN"},
    ],
    "meta_brasil": [
        {"col": "ano", "tipo": "not_null", "nivel": "FAIL"},
        {"col": "meta_alfabetizacao_2030", "tipo": "not_null", "nivel": "WARN"},
    ],
    "meta_uf": [
        {"col": "sigla_uf", "tipo": "not_null", "nivel": "FAIL"},
        {"col": "ano", "tipo": "not_null", "nivel": "FAIL"},
    ],
    "meta_municipio": [
        {"col": "id_municipio", "tipo": "not_null", "nivel": "FAIL"},
        {"col": "ano", "tipo": "not_null", "nivel": "FAIL"},
    ],
}


def _record_hash(df: pd.DataFrame, chaves: list[str]) -> pd.Series:
    concatenado = df[chaves].astype(str).agg("|".join, axis=1)
    return concatenado.apply(lambda s: hashlib.md5(s.encode("utf-8")).hexdigest())


def _checar_qualidade(df: pd.DataFrame, entidade: str) -> None:
    checks = CHECKS.get(entidade, [])
    criticos = 0
    for ck in checks:
        col, tipo, nivel = ck["col"], ck["tipo"], ck.get("nivel", "WARN")
        if tipo == "not_null":
            n = int(df[col].isna().sum())
            ok, detalhe = n == 0, f"{n} nulos"
        elif tipo == "range":
            mn, mx = ck["min"], ck["max"]
            fora = int((df[col].notna() & ((df[col] < mn) | (df[col] > mx))).sum())
            ok, detalhe = fora == 0, f"{fora} fora de [{mn},{mx}]"
        elif tipo == "in_set":
            inv = int((~df[col].isin(ck["valores"])).sum())
            ok, detalhe = inv == 0, f"{inv} fora do conjunto {ck['valores']}"
        else:
            ok, detalhe = False, "tipo desconhecido"

        status = "PASS" if ok else nivel
        log.info(f"[DQ:BRONZE] {entidade} | {status} | {tipo} | col={col} | {detalhe}")
        if not ok and nivel == "FAIL":
            criticos += 1

    if criticos > 0:
        raise ValueError(f"[DQ:BRONZE] {criticos} check(s) crítico(s) em '{entidade}'")


def run() -> dict[str, int]:
    BRONZE_DIR.mkdir(parents=True, exist_ok=True)
    ingestion_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    ingestion_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    resultados = {}
    for entidade, arquivo in ARQUIVOS_RAW.items():
        log.info(f"[BRONZE] Iniciando: {entidade}")
        caminho = RAW_DIR / arquivo
        df = pd.read_csv(caminho, dtype=DTYPES[entidade])

        df["_record_hash"] = _record_hash(df, HASH_KEYS[entidade])
        df["_ingestion_timestamp"] = ingestion_ts
        df["_ingestion_date"] = ingestion_date
        df["_source_entity"] = entidade
        df["_source_file"] = arquivo

        # dedup por _record_hash, mantendo o registro mais recente (equivalente
        # ao overwrite dinâmico por partição do job Glue original)
        antes = len(df)
        df = df.drop_duplicates(subset="_record_hash", keep="last")
        depois = len(df)
        if antes != depois:
            log.info(f"[BRONZE] {entidade}: removidos {antes - depois} duplicados por _record_hash")

        _checar_qualidade(df, entidade)

        destino = BRONZE_DIR / f"{entidade}.parquet"
        df.to_parquet(destino, index=False)
        log.info(f"[BRONZE] {entidade}: {len(df)} registros -> {destino}")
        resultados[entidade] = len(df)

    log.info("=" * 65)
    log.info("SUMÁRIO BRONZE")
    for entidade, total in resultados.items():
        log.info(f"  {entidade:<30}: {total:>6} registros")
    log.info("=" * 65)
    return resultados


if __name__ == "__main__":
    run()
