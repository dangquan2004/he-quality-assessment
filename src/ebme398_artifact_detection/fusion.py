from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .labels import Task
from .paths import parse_patch_id, parse_wsi_stem_from_patch_path


def load_h5_features(h5_path: str | Path, feature_key: str = "features") -> tuple[np.ndarray, np.ndarray | None]:
    with h5py.File(h5_path, "r") as handle:
        features = handle[feature_key][:]
        coords = handle["coords"][:] if "coords" in handle else None
    return features, coords


def spearman_rho_per_feature(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    ranks_x = pd.DataFrame(X).rank(method="average").to_numpy(dtype=np.float64)
    ranks_y = pd.Series(y).rank(method="average").to_numpy(dtype=np.float64)
    ranks_x -= ranks_x.mean(axis=0, keepdims=True)
    ranks_y -= ranks_y.mean()
    numerator = (ranks_x * ranks_y[:, None]).sum(axis=0)
    denominator = np.sqrt((ranks_x**2).sum(axis=0)) * np.sqrt((ranks_y**2).sum())
    rho = np.zeros(X.shape[1], dtype=np.float64)
    valid = denominator > 0
    rho[valid] = numerator[valid] / denominator[valid]
    return rho


def _multiclass_ovr_spearman(X: np.ndarray, y: np.ndarray, classes: np.ndarray) -> np.ndarray:
    per_class = [np.abs(spearman_rho_per_feature(X, (y == class_id).astype(np.int64))) for class_id in classes]
    return np.max(np.stack(per_class, axis=0), axis=0)


def fit_spearman_selection(
    hc_csv_path: str | Path,
    h5_dir: str | Path,
    selection_json: str | Path,
    *,
    threshold: float = 0.08,
    task: Task | str = Task.BINARY,
    label_column: str = "y_label",
    feature_key: str = "features",
) -> Path:
    task = Task(task)
    df = pd.read_csv(hc_csv_path)
    if "path" not in df.columns or label_column not in df.columns:
        raise KeyError(f"expected columns 'path' and {label_column!r}")
    df["patch_id"] = df["path"].map(parse_patch_id)
    df["wsi_stem"] = df["path"].map(parse_wsi_stem_from_patch_path)
    hc_cols = [col for col in df.columns if col not in {"path", label_column, "patch_id", "wsi_stem"}]
    X_hc_all, X_emb_all, y_all = [], [], []
    h5_dir = Path(h5_dir)
    for wsi_stem, group in df.groupby("wsi_stem"):
        h5_path = h5_dir / f"{wsi_stem}.h5"
        if not h5_path.exists():
            continue
        group = group.sort_values("patch_id").reset_index(drop=True)
        X_emb, _coords = load_h5_features(h5_path, feature_key=feature_key)
        patch_ids = group["patch_id"].to_numpy(dtype=int)
        if patch_ids.max() >= X_emb.shape[0]:
            continue
        X_hc_all.append(group[hc_cols].to_numpy(dtype=np.float32))
        X_emb_all.append(X_emb[patch_ids].astype(np.float32))
        y_all.append(group[label_column].to_numpy(dtype=np.int64))
    if not X_hc_all:
        raise RuntimeError(
            f"no overlapping handcrafted rows and H5 embeddings were found for {hc_csv_path} and {h5_dir}"
        )
    X_hc = np.concatenate(X_hc_all, axis=0)
    X_emb = np.concatenate(X_emb_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    if task is Task.BINARY:
        rho_hc = np.abs(spearman_rho_per_feature(X_hc, y))
        rho_emb = np.abs(spearman_rho_per_feature(X_emb, y))
    else:
        classes = np.unique(y)
        rho_hc = _multiclass_ovr_spearman(X_hc, y, classes)
        rho_emb = _multiclass_ovr_spearman(X_emb, y, classes)
    payload = {
        "threshold": float(threshold),
        "task": task.value,
        "hc_cols_all": hc_cols,
        "hc_keep_idx": np.where(rho_hc >= threshold)[0].tolist(),
        "hc_cols_keep": [hc_cols[idx] for idx in np.where(rho_hc >= threshold)[0]],
        "embedding_keep_idx": np.where(rho_emb >= threshold)[0].tolist(),
        "feature_key": feature_key,
    }
    selection_json = Path(selection_json)
    selection_json.parent.mkdir(parents=True, exist_ok=True)
    selection_json.write_text(json.dumps(payload, indent=2))
    return selection_json


def apply_selection_and_write_npz(
    hc_csv_path: str | Path,
    h5_dir: str | Path,
    output_dir: str | Path,
    selection_json: str | Path,
    *,
    label_column: str = "y_label",
) -> list[Path]:
    selection = json.loads(Path(selection_json).read_text())
    hc_cols = selection["hc_cols_all"]
    hc_keep = np.asarray(selection["hc_keep_idx"], dtype=int)
    emb_keep = np.asarray(selection["embedding_keep_idx"], dtype=int)
    feature_key = selection.get("feature_key", "features")
    df = pd.read_csv(hc_csv_path)
    df["patch_id"] = df["path"].map(parse_patch_id)
    df["wsi_stem"] = df["path"].map(parse_wsi_stem_from_patch_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for wsi_stem, group in df.groupby("wsi_stem"):
        h5_path = Path(h5_dir) / f"{wsi_stem}.h5"
        if not h5_path.exists():
            continue
        group = group.sort_values("patch_id").reset_index(drop=True)
        embeddings, coords = load_h5_features(h5_path, feature_key=feature_key)
        patch_ids = group["patch_id"].to_numpy(dtype=int)
        if patch_ids.max() >= embeddings.shape[0]:
            continue
        X_hc = group[hc_cols].to_numpy(dtype=np.float32)
        X_hc = X_hc[:, hc_keep] if hc_keep.size else np.zeros((len(group), 0), dtype=np.float32)
        X_emb = embeddings[patch_ids].astype(np.float32)
        X_emb = X_emb[:, emb_keep] if emb_keep.size else np.zeros((len(group), 0), dtype=np.float32)
        X_fused = np.concatenate([X_hc, X_emb], axis=1)
        out_path = output_dir / f"{wsi_stem}_fused.npz"
        np.savez_compressed(
            out_path,
            X_fused=X_fused.astype(np.float32),
            y=group[label_column].to_numpy(dtype=np.int64),
            paths=group["path"].to_numpy(dtype=object),
            coords=coords[patch_ids] if coords is not None else np.empty((len(group), 0), dtype=np.int64),
        )
        written.append(out_path)
    return written


def load_npz_directory(npz_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    arrays = []
    labels = []
    for path in sorted(Path(npz_dir).glob("*.npz")):
        payload = np.load(path, allow_pickle=True)
        arrays.append(payload["X_fused"])
        labels.append(payload["y"])
    if not arrays:
        raise RuntimeError(f"no npz files found in {npz_dir}")
    return np.concatenate(arrays, axis=0), np.concatenate(labels, axis=0)
