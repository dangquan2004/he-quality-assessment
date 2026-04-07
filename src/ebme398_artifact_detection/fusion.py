from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .alignment import align_handcrafted_rows_to_feature_rows, ensure_slide_id_column
from .labels import Task
from .selection import load_selection_payload, selection_embedding_keep, selection_feature_key, selection_hc_keep


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
    df = ensure_slide_id_column(pd.read_csv(hc_csv_path))
    if "path" not in df.columns or label_column not in df.columns:
        raise KeyError(f"expected columns 'path' and {label_column!r}")
    hc_cols = [
        col
        for col in df.columns
        if col not in {"path", label_column, "slide_id", "patch_id", "tile_idx", "feature_row_idx", "x", "y", "y0"}
    ]
    X_hc_all, X_emb_all, y_all = [], [], []
    alignment_modes: list[str] = []
    h5_dir = Path(h5_dir)
    for slide_id, group in df.groupby("slide_id"):
        h5_path = h5_dir / f"{slide_id}.h5"
        if not h5_path.exists():
            continue
        X_emb, coords = load_h5_features(h5_path, feature_key=feature_key)
        group, feature_row_idx, alignment_mode = align_handcrafted_rows_to_feature_rows(
            group,
            coords=coords,
            n_features=X_emb.shape[0],
            context=f"{hc_csv_path} :: {slide_id}",
        )
        alignment_modes.append(alignment_mode)
        X_hc_all.append(group[hc_cols].to_numpy(dtype=np.float32))
        X_emb_all.append(X_emb[feature_row_idx].astype(np.float32))
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
        "alignment_modes": sorted(set(alignment_modes)),
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
    selection = load_selection_payload(selection_json)
    hc_cols = selection["hc_cols_all"]
    hc_keep = selection_hc_keep(selection)
    emb_keep = selection_embedding_keep(selection)
    feature_key = selection_feature_key(selection)
    df = ensure_slide_id_column(pd.read_csv(hc_csv_path))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for slide_id, group in df.groupby("slide_id"):
        h5_path = Path(h5_dir) / f"{slide_id}.h5"
        if not h5_path.exists():
            continue
        embeddings, coords = load_h5_features(h5_path, feature_key=feature_key)
        group, feature_row_idx, _alignment_mode = align_handcrafted_rows_to_feature_rows(
            group,
            coords=coords,
            n_features=embeddings.shape[0],
            context=f"{hc_csv_path} :: {slide_id}",
        )
        X_hc = group[hc_cols].to_numpy(dtype=np.float32)
        X_hc = X_hc[:, hc_keep] if hc_keep.size else np.zeros((len(group), 0), dtype=np.float32)
        X_emb = embeddings[feature_row_idx].astype(np.float32)
        X_emb = X_emb[:, emb_keep] if emb_keep.size else np.zeros((len(group), 0), dtype=np.float32)
        X_fused = np.concatenate([X_hc, X_emb], axis=1)
        out_path = output_dir / f"{slide_id}_fused.npz"
        np.savez_compressed(
            out_path,
            X_fused=X_fused.astype(np.float32),
            y=group[label_column].to_numpy(dtype=np.int64),
            paths=group["path"].to_numpy(dtype=object),
            coords=coords[feature_row_idx] if coords is not None else np.empty((len(group), 0), dtype=np.int64),
            feature_row_idx=feature_row_idx.astype(np.int64),
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
