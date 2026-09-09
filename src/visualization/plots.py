"""Funções de plotagem reutilizadas em notebooks e no script de treino."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

sns.set_theme(style="whitegrid")


def plot_confusion_matrix(y_true, y_pred, out_path: Path, labels=("NAO_ATINGIU", "ATINGIU")):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=labels, cmap="Blues", ax=ax, colorbar=False
    )
    ax.set_title("Matriz de confusão — holdout 2024")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc_curve(y_true, y_proba, out_path: Path, model_name: str):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    RocCurveDisplay.from_predictions(y_true, y_proba, name=model_name, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_title("Curva ROC — holdout 2024")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_feature_importance(nomes, importancias, out_path: Path, top_n: int = 20, titulo: str = "Feature Importance"):
    ordem = np.argsort(importancias)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(7, max(4, 0.35 * len(ordem))))
    ax.barh([nomes[i] for i in ordem][::-1], [importancias[i] for i in ordem][::-1], color="#2b6cb0")
    ax.set_title(titulo)
    ax.set_xlabel("Importância")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
