from __future__ import annotations

import json
from pathlib import Path

import h5py
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torchvision import models

from .labels import Task, task_labels, to_task_label
from .metrics import dump_json, evaluate_predictions
from .tiles import CachedTileDataset


def _make_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    counts = np.bincount(labels.astype(int))
    weights = 1.0 / np.maximum(counts, 1)
    sample_weights = weights[labels.astype(int)]
    return WeightedRandomSampler(torch.as_tensor(sample_weights, dtype=torch.double), len(sample_weights), replacement=True)


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 512, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class KANClassifier(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        super().__init__()
        try:
            from efficient_kan import KAN
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("efficient-kan is not installed; use the kan extra to enable this model") from exc
        self.backbone = KAN([input_dim, hidden_dim, output_dim])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def _build_resnet(output_dim: int, arch: str = "resnet50", pretrained: bool = True) -> nn.Module:
    if arch not in {"resnet18", "resnet34", "resnet50"}:
        raise ValueError(f"unsupported resnet arch: {arch}")
    builder = getattr(models, arch)
    weight_map = {
        "resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
        "resnet34": models.ResNet34_Weights.IMAGENET1K_V1,
        "resnet50": models.ResNet50_Weights.IMAGENET1K_V2,
    }
    model = builder(weights=weight_map[arch] if pretrained else None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, output_dim)
    return model


def _criterion(task: Task) -> nn.Module:
    return nn.BCEWithLogitsLoss() if task is Task.BINARY else nn.CrossEntropyLoss()


def _tensor_targets(task: Task, y: torch.Tensor) -> torch.Tensor:
    return y.float() if task is Task.BINARY else y.long()


def _logits_to_probabilities(task: Task, logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits.view(-1)) if task is Task.BINARY else torch.softmax(logits, dim=1)


def _run_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer | None, task: Task, device: str) -> float:
    training = optimizer is not None
    model.train(training)
    criterion = _criterion(task)
    losses = []
    for X, y in loader:
        X = X.to(device)
        y = y.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        logits = model(X)
        loss = criterion(logits.view(-1) if task is Task.BINARY else logits, _tensor_targets(task, y))
        if training:
            loss.backward()
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def _predict(task: Task, model: nn.Module, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs = []
    all_targets = []
    for X, y in loader:
        logits = model(X.to(device))
        probs = _logits_to_probabilities(task, logits).detach().cpu().numpy()
        all_probs.append(probs)
        all_targets.append(y.numpy())
    probs = np.concatenate(all_probs, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    return probs, targets


def _fit_torch_model(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    task: Task,
    output_dir: Path,
    experiment_name: str,
    lr: float = 1e-3,
    epochs: int = 20,
    device: str | None = None,
) -> tuple[nn.Module, dict]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    best_val = float("-inf")
    best_path = output_dir / f"{experiment_name}_best.pt"
    history = []
    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(model, train_loader, optimizer, task, device)
        val_loss = _run_epoch(model, val_loader, None, task, device)
        val_probs, val_targets = _predict(task, model, val_loader, device)
        val_metrics = evaluate_predictions(task, val_targets, val_probs)
        score = val_metrics["auc"] if task is Task.BINARY else val_metrics["auc_ovr_macro"]
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_score": score})
        if score > best_val:
            best_val = score
            torch.save(model.state_dict(), best_path)
    model.load_state_dict(torch.load(best_path, map_location=device))
    return model, {"history": history, "best_model_path": str(best_path), "device": device}


def _load_h5_split_arrays(feature_dir: str | Path, label_dir: str | Path, splits_json: str | Path, task: Task) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    feature_dir = Path(feature_dir)
    label_dir = Path(label_dir)
    feature_map = {path.stem: path for path in feature_dir.glob("*.h5")}
    label_map = {path.stem: path for path in label_dir.glob("*.csv")}
    common = sorted(set(feature_map) & set(label_map))
    X_all, y_all, slide_ids = [], [], []
    for stem in common:
        with h5py.File(feature_map[stem], "r") as handle:
            features = handle["features"][:].astype(np.float32)
        labels = pd.read_csv(label_map[stem])
        label_col = "label" if "label" in labels.columns else "label_collapsed" if "label_collapsed" in labels.columns else None
        if label_col is None or "idx" not in labels.columns:
            raise KeyError(f"expected idx and label columns in {label_map[stem]}")
        labels = labels[(labels["idx"] >= 0) & (labels["idx"] < len(features))].copy()
        labels["y_label"] = [to_task_label(value, task) for value in labels[label_col]]
        idx = labels["idx"].to_numpy(dtype=int)
        X_all.append(features[idx])
        y_all.append(labels["y_label"].to_numpy(dtype=int))
        slide_ids.extend([stem] * len(idx))
    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    slide_ids = np.asarray(slide_ids)
    splits = json.loads(Path(splits_json).read_text())
    def _split(name: str) -> tuple[np.ndarray, np.ndarray]:
        mask = np.isin(slide_ids, splits[name])
        return X_all[mask], y_all[mask]
    return _split("train"), _split("val"), _split("test")


def _load_npz_split_arrays(train_dir: str | Path, val_dir: str | Path, test_dir: str | Path) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    def _read_dir(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
        arrays = []
        labels = []
        for npz_path in sorted(Path(path).glob("*.npz")):
            payload = np.load(npz_path, allow_pickle=True)
            arrays.append(payload["X_fused"])
            labels.append(payload["y"])
        return np.concatenate(arrays, axis=0), np.concatenate(labels, axis=0)
    return _read_dir(train_dir), _read_dir(val_dir), _read_dir(test_dir)


def train_embedding_classifier(
    *,
    output_dir: str | Path,
    task: Task | str,
    source_kind: str,
    hidden_dim: int = 512,
    model_kind: str = "mlp",
    batch_size: int = 256,
    epochs: int = 20,
    lr: float = 1e-3,
    balance_train: bool = True,
    experiment_name: str = "embedding_classifier",
    feature_dir: str | Path | None = None,
    label_dir: str | Path | None = None,
    splits_json: str | Path | None = None,
    train_dir: str | Path | None = None,
    val_dir: str | Path | None = None,
    test_dir: str | Path | None = None,
) -> dict:
    task = Task(task)
    output_dir = Path(output_dir)
    if source_kind == "h5":
        if feature_dir is None or label_dir is None or splits_json is None:
            raise ValueError("h5 training requires feature_dir, label_dir, and splits_json")
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = _load_h5_split_arrays(feature_dir, label_dir, splits_json, task)
    elif source_kind == "npz":
        if train_dir is None or val_dir is None or test_dir is None:
            raise ValueError("npz training requires train_dir, val_dir, and test_dir")
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = _load_npz_split_arrays(train_dir, val_dir, test_dir)
    else:
        raise ValueError(f"unsupported source_kind: {source_kind}")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    train_dataset = TensorDataset(torch.as_tensor(X_train, dtype=torch.float32), torch.as_tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.as_tensor(X_val, dtype=torch.float32), torch.as_tensor(y_val, dtype=torch.long))
    test_dataset = TensorDataset(torch.as_tensor(X_test, dtype=torch.float32), torch.as_tensor(y_test, dtype=torch.long))
    sampler = _make_weighted_sampler(y_train) if balance_train else None
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, shuffle=sampler is None)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    output_dim = 1 if task is Task.BINARY else len(task_labels(task))
    if model_kind == "mlp":
        model = MLPClassifier(X_train.shape[1], output_dim, hidden_dim=hidden_dim)
    elif model_kind == "kan":
        model = KANClassifier(X_train.shape[1], output_dim, hidden_dim=hidden_dim)
    else:
        raise ValueError(f"unsupported model_kind: {model_kind}")
    model, fit_info = _fit_torch_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        task=task,
        output_dir=output_dir,
        experiment_name=experiment_name,
        lr=lr,
        epochs=epochs,
    )
    device = fit_info["device"]
    probs_train, y_train_true = _predict(task, model, train_loader, device)
    probs_val, y_val_true = _predict(task, model, val_loader, device)
    probs_test, y_test_true = _predict(task, model, test_loader, device)
    metrics = {
        "train": evaluate_predictions(task, y_train_true, probs_train),
        "val": evaluate_predictions(task, y_val_true, probs_val),
        "test": evaluate_predictions(task, y_test_true, probs_test),
        "fit": fit_info,
    }
    dump_json(output_dir / f"{experiment_name}_metrics.json", metrics)
    joblib.dump(scaler, output_dir / f"{experiment_name}_scaler.joblib")
    return {
        "best_model_path": fit_info["best_model_path"],
        "metrics_path": output_dir / f"{experiment_name}_metrics.json",
        "scaler_path": output_dir / f"{experiment_name}_scaler.joblib",
    }


def train_resnet_classifier(
    *,
    train_meta_csv: str | Path,
    val_meta_csv: str | Path,
    test_meta_csv: str | Path,
    output_dir: str | Path,
    task: Task | str,
    arch: str = "resnet50",
    batch_size: int = 32,
    epochs: int = 10,
    lr: float = 1e-4,
    balance_train: bool = True,
    pretrained: bool = True,
    experiment_name: str = "resnet_classifier",
) -> dict:
    task = Task(task)
    output_dir = Path(output_dir)
    train_dataset = CachedTileDataset(train_meta_csv)
    val_dataset = CachedTileDataset(val_meta_csv)
    test_dataset = CachedTileDataset(test_meta_csv)
    train_labels = pd.read_csv(train_meta_csv)["y_label"].to_numpy(dtype=int)
    sampler = _make_weighted_sampler(train_labels) if balance_train else None
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, shuffle=sampler is None)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    output_dim = 1 if task is Task.BINARY else len(task_labels(task))
    model = _build_resnet(output_dim=output_dim, arch=arch, pretrained=pretrained)
    model, fit_info = _fit_torch_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        task=task,
        output_dir=output_dir,
        experiment_name=experiment_name,
        lr=lr,
        epochs=epochs,
    )
    device = fit_info["device"]
    probs_train, y_train = _predict(task, model, train_loader, device)
    probs_val, y_val = _predict(task, model, val_loader, device)
    probs_test, y_test = _predict(task, model, test_loader, device)
    metrics = {
        "train": evaluate_predictions(task, y_train, probs_train),
        "val": evaluate_predictions(task, y_val, probs_val),
        "test": evaluate_predictions(task, y_test, probs_test),
        "fit": fit_info,
    }
    dump_json(output_dir / f"{experiment_name}_metrics.json", metrics)
    return {
        "best_model_path": fit_info["best_model_path"],
        "metrics_path": output_dir / f"{experiment_name}_metrics.json",
    }
