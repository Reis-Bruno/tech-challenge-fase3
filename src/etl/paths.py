"""Caminhos padronizados do data lake local (réplica do S3 da Fase 2)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"

ARQUIVOS_RAW = {
    "indicador_municipio": "br_inep_avaliacao_alfabetizacao_municipio.csv",
    "indicador_uf": "br_inep_avaliacao_alfabetizacao_uf.csv",
    "meta_brasil": "br_inep_avaliacao_alfabetizacao_meta_alfabetizacao_brasil.csv",
    "meta_uf": "br_inep_avaliacao_alfabetizacao_meta_alfabetizacao_uf.csv",
    "meta_municipio": "br_inep_avaliacao_alfabetizacao_meta_alfabetizacao_municipio.csv",
}
