from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .labels import Task, task_labels
from .metrics import dump_json, evaluate_predictions


def _split_xy(frame: pd.DataFrame, label_column: str) -> tuple[pd.DataFrame, np.ndarray]:
    y = frame[label_column].to_numpy(dtype=int)
    X = frame.drop(columns=[label_column], errors="ignore")
    X = X.drop(columns=["path", "slide_id", "tile_idx"], errors="ignore")
    X = X.select_dtypes(include=[np.number])
    return X, y


def _balance_binary(frame: pd.DataFrame, label_column: str, seed: int) -> pd.DataFrame:
    groups = {label: subset for label, subset in frame.groupby(label_column)}
    if set(groups) != {0, 1}:
        return frame
    n = min(len(groups[0]), len(groups[1]))
    return (
        pd.concat(
            [
                groups[0].sample(n=n, random_state=seed),
                groups[1].sample(n=n, random_state=seed),
            ]
        )
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )


def _cap_per_class(frame: pd.DataFrame, label_column: str, max_per_class: int, seed: int) -> pd.DataFrame:
    capped = []
    for _, subset in frame.groupby(label_column):
        if len(subset) > max_per_class:
            subset = subset.sample(n=max_per_class, random_state=seed)
        capped.append(subset)
    return pd.concat(capped).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def build_estimator(task: Task | str, *, estimator: str = "svm", seed: int = 42, balance_train: bool = False):
    task = Task(task)
    estimator = estimator.lower()
    if estimator == "svm":
        classifier = SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            class_weight="balanced" if balance_train else None,
            random_state=seed,
            decision_function_shape="ovr",
        )
    elif estimator == "xgb":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("xgboost is not installed; use the xgb extra or choose svm") from exc
        classifier = XGBClassifier(
            random_state=seed,
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic" if task is Task.BINARY else "multi:softprob",
            eval_metric="logloss",
            num_class=len(task_labels(task)) if task is Task.MULTICLASS else None,
        )
    else:
        raise ValueError(f"unsupported estimator: {estimator}")
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )


def train_feature_classifier(
    train_csv: str | Path,
    val_csv: str | Path,
    test_csv: str | Path,
    *,
    output_dir: str | Path,
    task: Task | str,
    label_column: str = "y_label",
    estimator: str = "svm",
    balance_train: bool = False,
    max_train_per_class: int | None = None,
    experiment_name: str = "feature_classifier",
    seed: int = 42,
) -> dict:
    task = Task(task)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)
    if balance_train and task is Task.BINARY:
        train_df = _balance_binary(train_df, label_column, seed)
    if max_train_per_class is not None:
        train_df = _cap_per_class(train_df, label_column, max_train_per_class, seed)
    X_train, y_train = _split_xy(train_df, label_column)
    X_val, y_val = _split_xy(val_df, label_column)
    X_test, y_test = _split_xy(test_df, label_column)
    feature_columns = list(X_train.columns)
    model = build_estimator(task, estimator=estimator, seed=seed, balance_train=balance_train)
    model.fit(X_train, y_train)
    probabilities = {
        "train": model.predict_proba(X_train),
        "val": model.predict_proba(X_val),
        "test": model.predict_proba(X_test),
    }
    if task is Task.BINARY:
        probabilities = {name: probs[:, 1] for name, probs in probabilities.items()}
    metrics = {
        split_name: evaluate_predictions(task, y_true, probabilities[split_name])
        for split_name, y_true in (("train", y_train), ("val", y_val), ("test", y_test))
    }
    metrics_payload = {
        "experiment_name": experiment_name,
        "task": task.value,
        "estimator": estimator,
        "feature_columns": feature_columns,
        "metrics": metrics,
    }
    dump_json(output_dir / f"{experiment_name}_metrics.json", metrics_payload)
    joblib.dump(model, output_dir / f"{experiment_name}_model.joblib")
    joblib.dump(feature_columns, output_dir / f"{experiment_name}_feature_columns.joblib")
    if task is Task.BINARY:
        pd.DataFrame({"y_true": y_test, "probability": probabilities["test"]}).to_csv(
            output_dir / f"{experiment_name}_test_predictions.csv",
            index=False,
        )
    else:
        columns = {f"prob_{name}": probabilities["test"][:, idx] for idx, name in task_labels(task).items()}
        pd.DataFrame({"y_true": y_test, "prediction": probabilities["test"].argmax(axis=1), **columns}).to_csv(
            output_dir / f"{experiment_name}_test_predictions.csv",
            index=False,
        )
    return {
        "model_path": output_dir / f"{experiment_name}_model.joblib",
        "metrics_path": output_dir / f"{experiment_name}_metrics.json",
    }
