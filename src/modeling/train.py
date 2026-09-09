"""
Treina e valida os modelos candidatos, seleciona o melhor e gera todos os
artefatos de avaliação/interpretabilidade (métricas, matriz de confusão,
curva ROC, feature importance, SHAP).

Estratégia de validação (anti data-leakage)
--------------------------------------------
- Split temporal: treino = ano 2023, holdout = ano 2024. O modelo nunca
  vê dados de 2024 durante o treino/tuning — isso testa a capacidade real
  de generalizar para o futuro (pergunta de negócio do desafio: "prever
  municípios que podem não atingir metas futuras").
- Dentro do treino (2023), usa-se StratifiedKFold (5 folds) para tuning
  de hiperparâmetros via GridSearchCV. O pré-processamento (imputação,
  encoding, escala) está dentro do Pipeline do sklearn, portanto é
  reajustado a cada fold — nenhuma estatística do fold de validação
  (nem do holdout 2024) vaza para o treino.
"""

import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.etl.paths import DATA_DIR, ROOT
from src.evaluation.interpretability import native_feature_importance, shap_summary
from src.evaluation.metrics import compute_metrics, confusion
from src.modeling.pipeline import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    PARAM_GRIDS,
    TARGET,
    build_model_pipelines,
)
from src.visualization.plots import plot_confusion_matrix, plot_feature_importance, plot_roc_curve

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
log = logging.getLogger(__name__)

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
REPORTS_DIR = ROOT / "reports"
IMAGES_DIR = ROOT / "images"


def load_dataset() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "features" / "dataset_modelagem.parquet")


def split_temporal(df: pd.DataFrame):
    treino = df[df["ano"] == 2023]
    teste = df[df["ano"] == 2024]
    log.info(f"[SPLIT] treino (2023): {len(treino)} | holdout (2024): {len(teste)}")
    return (
        treino[FEATURES], treino[TARGET],
        teste[FEATURES], teste[TARGET],
    )


def run() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    IMAGES_DIR.mkdir(exist_ok=True)

    df = load_dataset()
    X_train, y_train, X_test, y_test = split_temporal(df)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipelines = build_model_pipelines()

    resultados = []
    modelos_ajustados = {}

    for nome, pipe in pipelines.items():
        log.info(f"[TRAIN] Tuning: {nome}")
        grid = GridSearchCV(
            pipe, PARAM_GRIDS[nome], cv=cv, scoring="f1", n_jobs=-1, refit=True
        )
        grid.fit(X_train, y_train)
        best = grid.best_estimator_
        modelos_ajustados[nome] = best

        y_pred = best.predict(X_test)
        y_proba = best.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_proba)
        metrics.update({
            "modelo": nome,
            "melhores_parametros": json.dumps(grid.best_params_, ensure_ascii=False),
            "f1_cv_treino": round(grid.best_score_, 4),
        })
        resultados.append(metrics)
        log.info(f"[TRAIN] {nome} -> holdout: {metrics}")

    df_resultados = pd.DataFrame(resultados).sort_values("roc_auc", ascending=False)
    df_resultados.to_csv(REPORTS_DIR / "comparacao_modelos.csv", index=False)
    log.info(f"[TRAIN] comparação salva em {REPORTS_DIR / 'comparacao_modelos.csv'}")

    melhor_nome = df_resultados.iloc[0]["modelo"]
    melhor_pipeline = modelos_ajustados[melhor_nome]
    log.info(f"[TRAIN] Melhor modelo (por ROC-AUC no holdout 2024): {melhor_nome}")

    joblib.dump(melhor_pipeline, REPORTS_DIR / "modelo_final.joblib")

    y_pred_final = melhor_pipeline.predict(X_test)
    y_proba_final = melhor_pipeline.predict_proba(X_test)[:, 1]

    plot_confusion_matrix(y_test, y_pred_final, IMAGES_DIR / "matriz_confusao.png")
    plot_roc_curve(y_test, y_proba_final, IMAGES_DIR / "curva_roc.png", melhor_nome)

    fi = native_feature_importance(melhor_pipeline)
    if fi is not None:
        nomes, importancias = fi
        plot_feature_importance(
            nomes, importancias, IMAGES_DIR / "feature_importance.png",
            titulo=f"Feature Importance — {melhor_nome}",
        )
        shap_summary(melhor_pipeline, X_test, IMAGES_DIR / "shap_summary.png")

    cm = confusion(y_test, y_pred_final)
    resumo = {
        "melhor_modelo": melhor_nome,
        "metricas_holdout_2024": df_resultados.iloc[0].to_dict(),
        "matriz_confusao": cm.tolist(),
        "n_treino_2023": int(len(X_train)),
        "n_holdout_2024": int(len(X_test)),
    }
    with open(REPORTS_DIR / "resumo_treino.json", "w", encoding="utf-8") as f:
        json.dump(resumo, f, ensure_ascii=False, indent=2)

    log.info("[TRAIN] Artefatos gerados em reports/ e images/.")


if __name__ == "__main__":
    run()
