"""
Pipelines de pré-processamento + modelo (Scikit-learn), com o
pré-processamento integrado diretamente ao estimador via `Pipeline`
(evita vazamento entre folds de validação cruzada: o imputer/encoder é
ajustado apenas nos dados de treino de cada fold).
"""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = ["media_portugues", "meta_2025", "meta_uf_2025", "taxa_media_uf_loo"]
CATEGORICAL_FEATURES = ["sigla_uf", "regiao"]
TARGET = "alvo_alfabetizado"
ID_COLS = ["id_municipio", "ano"]


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(transformers=[
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


def build_model_pipelines() -> dict[str, Pipeline]:
    """Um Pipeline completo (pré-processamento + estimador) por modelo candidato.

    Cada Pipeline recebe sua própria instância de ColumnTransformer (não
    compartilhada) para que o GridSearchCV possa clonar e ajustar cada
    combinação de forma independente por fold.
    """
    modelos = {
        "logistic_regression": LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=42
        ),
        "random_forest": RandomForestClassifier(
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
    }

    return {
        nome: Pipeline(steps=[("preprocessor", build_preprocessor()), ("classifier", modelo)])
        for nome, modelo in modelos.items()
    }


PARAM_GRIDS = {
    "logistic_regression": {
        "classifier__C": [0.01, 0.1, 1.0, 10.0],
    },
    "random_forest": {
        "classifier__n_estimators": [200, 400],
        "classifier__max_depth": [4, 8, None],
        "classifier__min_samples_leaf": [1, 5, 10],
    },
    "gradient_boosting": {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [2, 3, 4],
        "classifier__learning_rate": [0.05, 0.1],
    },
}
