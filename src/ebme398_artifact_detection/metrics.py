from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from .labels import Task, task_labels


def safe_binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_binary_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def binary_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "base_rate": float(np.mean(y_true)),
        "auc": safe_binary_auc(y_true, probs),
        "ap": safe_binary_ap(y_true, probs),
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "cm": cm.tolist(),
        "classification_report": classification_report(
            y_true,
            preds,
            output_dict=True,
            zero_division=0,
        ),
    }


def multiclass_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict:
    preds = probs.argmax(axis=1)
    labels = sorted(task_labels(Task.MULTICLASS))
    cm = confusion_matrix(y_true, preds, labels=labels)
    y_bin = label_binarize(y_true, classes=labels)
    auc = float(
        roc_auc_score(y_bin, probs, multi_class="ovr", average="macro")
    ) if len(np.unique(y_true)) > 1 else float("nan")
    ap = float(
        average_precision_score(y_bin, probs, average="macro")
    ) if len(np.unique(y_true)) > 1 else float("nan")
    return {
        "n": int(len(y_true)),
        "auc_ovr_macro": auc,
        "ap_macro": ap,
        "accuracy": float(accuracy_score(y_true, preds)),
        "macro_f1": float(f1_score(y_true, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, preds, average="weighted", zero_division=0)),
        "cm": cm.tolist(),
        "classification_report": classification_report(
            y_true,
            preds,
            output_dict=True,
            zero_division=0,
        ),
    }


def evaluate_predictions(task: Task | str, y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    task = Task(task)
    if task is Task.BINARY:
        return binary_metrics(y_true, probs.reshape(-1), threshold=threshold)
    return multiclass_metrics(y_true, probs)


def dump_json(path: str | Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path
