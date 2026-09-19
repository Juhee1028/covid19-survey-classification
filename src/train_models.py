from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


TARGET = "covid19_positive"
EXCLUDED_COLUMNS = [
    "Participant_ID",
    "survey_date",
    "ip_latitude",
    "ip_longitude",
    "risk_infection",
    "risk_infection_level",
    "risk_mortality",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and compare classification models on the IFN509 survey dataset."
    )
    parser.add_argument("--data", required=True, type=Path, help="Path to Dataset.csv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project directory for figures and result files",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(path)
    if TARGET not in data.columns:
        raise ValueError(f"Required target column is missing: {TARGET}")

    data = data.replace({"?": np.nan, "blank": np.nan})
    y = data[TARGET].astype(int)
    X = data.drop(columns=[TARGET, *EXCLUDED_COLUMNS], errors="ignore")
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = X.select_dtypes(include="number").columns.tolist()
    categorical_columns = X.select_dtypes(exclude="number").columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )


def evaluate_model(
    name: str,
    model: GridSearchCV,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict[str, float | str], object]:
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics: dict[str, float | str] = {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }
    return metrics, predictions


def save_target_plot(y: pd.Series, path: Path) -> None:
    counts = y.value_counts().sort_index()
    labels = ["Negative", "Positive"]

    plt.figure(figsize=(6.5, 4.5))
    ax = sns.barplot(x=labels, y=counts.values, hue=labels, palette="Set2", legend=False)
    ax.set(title="Target Class Distribution", xlabel="COVID-19 survey outcome", ylabel="Records")
    for index, value in enumerate(counts.values):
        ax.text(index, value + 35, f"{value:,}", ha="center")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_missing_values_plot(X: pd.DataFrame, path: Path) -> None:
    missing = X.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(
        x=missing.values,
        y=missing.index,
        hue=missing.index,
        palette="Set2",
        legend=False,
    )
    ax.set(title="Missing Values Before Preprocessing", xlabel="Missing records", ylabel="Feature")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_confusion_plot(y_test: pd.Series, predictions: object, path: Path) -> None:
    matrix = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
    )
    plt.title("Decision Tree Confusion Matrix")
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_model_comparison(metrics: pd.DataFrame, path: Path) -> None:
    chart_data = metrics.melt(
        id_vars="model",
        value_vars=["accuracy", "balanced_accuracy", "f1", "roc_auc"],
        var_name="metric",
        value_name="score",
    )
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(data=chart_data, x="metric", y="score", hue="model", palette="Set2")
    ax.set(title="Model Performance on the Held-out Test Set", xlabel="Metric", ylabel="Score")
    ax.set_ylim(0, 1)
    ax.legend(title="Model", loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    figures_dir = args.output_dir / "figures"
    results_dir = args.output_dir / "results"
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_dataset(args.data)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=10,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)
    model_specs = {
        "Decision Tree": (
            DecisionTreeClassifier(random_state=10),
            {
                "classifier__criterion": ["gini", "entropy"],
                "classifier__max_depth": [3, 5, 7, 10, None],
                "classifier__min_samples_leaf": [1, 5, 10, 20],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
        "KNN": (
            KNeighborsClassifier(),
            {
                "classifier__n_neighbors": [5, 9, 15, 25],
                "classifier__weights": ["uniform", "distance"],
                "classifier__p": [1, 2],
            },
        ),
    }

    rows: list[dict[str, float | str]] = []
    best_parameters: dict[str, dict[str, object]] = {}
    predictions_by_model: dict[str, object] = {}

    for model_name, (classifier, parameter_grid) in model_specs.items():
        pipeline = Pipeline(
            steps=[
                ("preprocess", build_preprocessor(X_train)),
                ("classifier", classifier),
            ]
        )
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=parameter_grid,
            scoring="f1",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)
        metrics, predictions = evaluate_model(model_name, search, X_test, y_test)
        metrics["cv_f1"] = search.best_score_
        rows.append(metrics)
        predictions_by_model[model_name] = predictions
        best_parameters[model_name] = search.best_params_

    metrics_df = pd.DataFrame(rows).sort_values("f1", ascending=False)
    metrics_df.to_csv(results_dir / "model_metrics.csv", index=False)
    (results_dir / "best_parameters.json").write_text(
        json.dumps(best_parameters, indent=2), encoding="utf-8"
    )
    (results_dir / "data_summary.json").write_text(
        json.dumps(
            {
                "records": len(X),
                "input_features_before_encoding": X.shape[1],
                "train_records": len(X_train),
                "test_records": len(X_test),
                "positive_rate": float(y.mean()),
                "excluded_columns": EXCLUDED_COLUMNS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    save_target_plot(y, figures_dir / "target_distribution.png")
    save_missing_values_plot(X, figures_dir / "missing_values.png")
    save_confusion_plot(
        y_test,
        predictions_by_model["Decision Tree"],
        figures_dir / "decision_tree_confusion_matrix.png",
    )
    save_model_comparison(metrics_df, figures_dir / "model_comparison.png")
    print(metrics_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nBest parameters:")
    print(json.dumps(best_parameters, indent=2))


if __name__ == "__main__":
    main()
