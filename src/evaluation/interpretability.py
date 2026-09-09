"""Interpretabilidade do modelo: Feature Importance nativa e SHAP Values."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.pipeline import Pipeline


def get_feature_names(pipeline: Pipeline) -> list[str]:
    return list(pipeline.named_steps["preprocessor"].get_feature_names_out())


def native_feature_importance(pipeline: Pipeline) -> tuple[list[str], np.ndarray] | None:
    classifier = pipeline.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        return get_feature_names(pipeline), classifier.feature_importances_
    if hasattr(classifier, "coef_"):
        # Para modelos lineares, usa o valor absoluto do coeficiente como
        # proxy de importância (features numéricas já estão padronizadas).
        return get_feature_names(pipeline), np.abs(classifier.coef_[0])
    return None


def shap_summary(pipeline: Pipeline, X_sample, out_path: Path, max_display: int = 15) -> None:
    """Gera o gráfico de resumo do SHAP (impacto médio absoluto por feature)."""
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    X_transformed = preprocessor.transform(X_sample)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
    nomes = get_feature_names(pipeline)

    if hasattr(classifier, "feature_importances_"):
        explainer = shap.TreeExplainer(classifier)
    else:
        explainer = shap.LinearExplainer(classifier, X_transformed)
    shap_values = explainer.shap_values(X_transformed)

    # Para classificadores binários, shap pode retornar lista [classe0, classe1]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    fig = plt.figure(figsize=(8, 6))
    shap.summary_plot(
        shap_values, X_transformed, feature_names=nomes, max_display=max_display, show=False
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
