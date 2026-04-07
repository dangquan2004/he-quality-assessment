from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import openslide
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from .alignment import align_handcrafted_rows_to_feature_rows, ensure_slide_id_column
from .handcrafted import extract_kba_features
from .fusion import load_h5_features
from .labels import Task, task_labels
from .metrics import dump_json, evaluate_predictions
from .selection import load_selection_payload, selection_embedding_keep, selection_feature_key, selection_hc_keep
from .tiles import TileCachingConfig, pick_level_for_patch
from .train_torch import KANClassifier, MLPClassifier, _logits_to_probabilities, resolve_torch_device
from .trident import run_trident_batch, write_custom_wsi_manifest


def _prediction_frame(task: Task, probs: np.ndarray) -> pd.DataFrame:
    if task is Task.BINARY:
        return pd.DataFrame(
            {
                "prob_unclean": probs.reshape(-1),
                "pred_idx": (probs.reshape(-1) >= 0.5).astype(int),
                "pred_label": np.where(probs.reshape(-1) >= 0.5, "unclean", "clean"),
            }
        )
    labels = task_labels(task)
    frame = pd.DataFrame({f"prob_{labels[idx]}": probs[:, idx] for idx in sorted(labels)})
    pred_idx = probs.argmax(axis=1)
    frame["pred_idx"] = pred_idx
    frame["pred_label"] = [labels[int(idx)] for idx in pred_idx]
    return frame


def _append_ground_truth(frame: pd.DataFrame, y_true: np.ndarray | None, task: Task) -> pd.DataFrame:
    if y_true is None:
        return frame
    labels = task_labels(task)
    frame = frame.copy()
    frame["y_true"] = y_true.astype(int)
    frame["y_true_label"] = [labels[int(idx)] for idx in y_true]
    return frame


def _write_prediction_outputs(
    *,
    output_csv: str | Path,
    task: Task,
    base_frame: pd.DataFrame,
    probs: np.ndarray,
    y_true: np.ndarray | None = None,
) -> dict:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pred_frame = pd.concat(
        [base_frame.reset_index(drop=True), _append_ground_truth(_prediction_frame(task, probs), y_true, task)],
        axis=1,
    )
    pred_frame.to_csv(output_csv, index=False)
    payload = {"predictions_csv": str(output_csv)}
    if y_true is not None:
        metrics = evaluate_predictions(task, y_true, probs)
        metrics_path = output_csv.with_suffix(".metrics.json")
        dump_json(metrics_path, metrics)
        payload["metrics_json"] = str(metrics_path)
    return payload


def _load_hybrid_model(
    *,
    checkpoint_path: str | Path,
    input_dim: int,
    task: Task,
    model_kind: str,
    hidden_dim: int,
    device: str,
):
    output_dim = 1 if task is Task.BINARY else len(task_labels(task))
    if model_kind == "mlp":
        model = MLPClassifier(input_dim, output_dim, hidden_dim=hidden_dim)
    elif model_kind == "kan":
        model = KANClassifier(input_dim, output_dim, hidden_dim=hidden_dim)
    else:
        raise ValueError(f"unsupported model_kind: {model_kind}")
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def _predict_arrays(task: Task, model: torch.nn.Module, X: np.ndarray, *, batch_size: int, device: str) -> np.ndarray:
    dataset = TensorDataset(torch.as_tensor(X, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_probs = []
    for (batch_X,) in loader:
        logits = model(batch_X.to(device))
        probs = _logits_to_probabilities(task, logits).detach().cpu().numpy()
        all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


def _load_hybrid_rows_from_h5(
    *,
    hc_csv: str | Path,
    h5_dir: str | Path,
    selection_json: str | Path,
    label_column: str = "y_label",
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray | None]:
    selection = load_selection_payload(selection_json)
    hc_cols_all = selection["hc_cols_all"]
    hc_keep = selection_hc_keep(selection)
    emb_keep = selection_embedding_keep(selection)
    feature_key = selection_feature_key(selection)

    df = ensure_slide_id_column(pd.read_csv(hc_csv))
    if "path" not in df.columns:
        raise KeyError(f"expected 'path' column in {hc_csv}")

    arrays = []
    y_true_batches = []
    rows = []
    h5_dir = Path(h5_dir)
    for slide_id, group in df.groupby("slide_id"):
        h5_path = h5_dir / f"{slide_id}.h5"
        if not h5_path.exists():
            continue
        embeddings, coords = load_h5_features(h5_path, feature_key=feature_key)
        group, feature_row_idx, _alignment_mode = align_handcrafted_rows_to_feature_rows(
            group,
            coords=coords,
            n_features=embeddings.shape[0],
            context=f"{hc_csv} :: {slide_id}",
        )
        X_hc = group[hc_cols_all].to_numpy(dtype=np.float32)
        X_hc = X_hc[:, hc_keep] if hc_keep.size else np.zeros((len(group), 0), dtype=np.float32)
        X_emb = embeddings[feature_row_idx].astype(np.float32)
        X_emb = X_emb[:, emb_keep] if emb_keep.size else np.zeros((len(group), 0), dtype=np.float32)
        arrays.append(np.concatenate([X_hc, X_emb], axis=1))
        if label_column in group.columns:
            y_true_batches.append(group[label_column].to_numpy(dtype=int))
        coord_subset = coords[feature_row_idx] if coords is not None and len(coords) else np.empty((len(group), 0), dtype=int)
        for idx, row in enumerate(group.itertuples(index=False)):
            record = {
                "slide_id": slide_id,
                "patch_id": int(feature_row_idx[idx]),
                "feature_row_idx": int(feature_row_idx[idx]),
                "path": row.path,
            }
            if coord_subset.shape[1] >= 2:
                record["x"] = int(coord_subset[idx, 0])
                record["y"] = int(coord_subset[idx, 1])
            rows.append(record)
    if not arrays:
        raise RuntimeError(f"no overlapping handcrafted rows and H5 embeddings were found for {hc_csv} and {h5_dir}")
    X_fused = np.concatenate(arrays, axis=0)
    y_true = np.concatenate(y_true_batches, axis=0) if y_true_batches else None
    return pd.DataFrame(rows), X_fused, y_true


def _load_hybrid_rows_from_npz(npz_dir: str | Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray | None]:
    rows = []
    arrays = []
    y_true_batches = []
    for npz_path in sorted(Path(npz_dir).glob("*.npz")):
        payload = np.load(npz_path, allow_pickle=True)
        X_fused = payload["X_fused"].astype(np.float32)
        arrays.append(X_fused)
        if "y" in payload.files:
            y_true_batches.append(payload["y"].astype(int))
        coords = payload["coords"] if "coords" in payload.files else np.empty((len(X_fused), 0), dtype=int)
        paths = payload["paths"] if "paths" in payload.files else np.asarray([""] * len(X_fused), dtype=object)
        feature_row_idx = payload["feature_row_idx"] if "feature_row_idx" in payload.files else np.arange(len(X_fused), dtype=int)
        slide_id = npz_path.stem.replace("_fused", "")
        for idx in range(len(X_fused)):
            record = {"slide_id": slide_id, "path": str(paths[idx]), "feature_row_idx": int(feature_row_idx[idx])}
            if coords.shape[1] >= 2:
                record["x"] = int(coords[idx, 0])
                record["y"] = int(coords[idx, 1])
            rows.append(record)
    if not arrays:
        raise RuntimeError(f"no npz files found in {npz_dir}")
    X = np.concatenate(arrays, axis=0)
    y_true = np.concatenate(y_true_batches, axis=0) if y_true_batches else None
    return pd.DataFrame(rows), X, y_true


def predict_hybrid_classifier(
    *,
    output_csv: str | Path,
    checkpoint_path: str | Path,
    scaler_path: str | Path,
    task: Task | str,
    source_kind: str,
    model_kind: str = "mlp",
    hidden_dim: int = 512,
    batch_size: int = 256,
    hc_csv: str | Path | None = None,
    h5_dir: str | Path | None = None,
    selection_json: str | Path | None = None,
    npz_dir: str | Path | None = None,
    label_column: str = "y_label",
    device: str | None = None,
) -> dict:
    task = Task(task)
    device = resolve_torch_device(device)
    if source_kind == "h5":
        if hc_csv is None or h5_dir is None or selection_json is None:
            raise ValueError("h5 hybrid inference requires hc_csv, h5_dir, and selection_json")
        meta_frame, X_raw, y_true = _load_hybrid_rows_from_h5(
            hc_csv=hc_csv,
            h5_dir=h5_dir,
            selection_json=selection_json,
            label_column=label_column,
        )
    elif source_kind == "npz":
        if npz_dir is None:
            raise ValueError("npz hybrid inference requires npz_dir")
        meta_frame, X_raw, y_true = _load_hybrid_rows_from_npz(npz_dir)
    else:
        raise ValueError(f"unsupported source_kind: {source_kind}")

    scaler = joblib.load(scaler_path)
    X = scaler.transform(X_raw)
    model = _load_hybrid_model(
        checkpoint_path=checkpoint_path,
        input_dim=X.shape[1],
        task=task,
        model_kind=model_kind,
        hidden_dim=hidden_dim,
        device=device,
    )
    probs = _predict_arrays(task, model, X, batch_size=batch_size, device=device)
    return _write_prediction_outputs(output_csv=output_csv, task=task, base_frame=meta_frame, probs=probs, y_true=y_true)


def summarize_hybrid_predictions_by_slide(
    predictions_csv: str | Path,
    output_json: str | Path,
    *,
    task: Task | str,
    binary_threshold: float = 0.5,
) -> Path:
    task = Task(task)
    df = pd.read_csv(predictions_csv)
    if "slide_id" not in df.columns:
        raise KeyError("predictions CSV needs a slide_id column for hybrid slide summaries")
    if task is Task.BINARY:
        if "prob_unclean" not in df.columns:
            raise KeyError("binary prediction CSV must include prob_unclean")
        grouped = (
            df.groupby("slide_id", as_index=False)["prob_unclean"]
            .agg(["mean", "max", "count"])
            .reset_index()
            .rename(columns={"mean": "mean_prob_unclean", "max": "max_prob_unclean", "count": "n_tiles"})
        )
        grouped["slide_threshold"] = float(binary_threshold)
        grouped["slide_pred_label"] = np.where(grouped["mean_prob_unclean"] >= binary_threshold, "unclean", "clean")
        payload = grouped.to_dict(orient="records")
    else:
        prob_cols = [column for column in df.columns if column.startswith("prob_")]
        grouped = df.groupby("slide_id", as_index=False)[prob_cols].mean()
        pred_cols = grouped[prob_cols].to_numpy()
        labels = task_labels(task)
        grouped["slide_pred_label"] = [labels[int(idx)] for idx in pred_cols.argmax(axis=1)]
        grouped["n_tiles"] = df.groupby("slide_id").size().to_numpy()
        payload = grouped.to_dict(orient="records")
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2))
    return output_json


def _pil_to_u8_chw(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image, dtype=np.uint8)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _ensure_pyramidal_single_wsi(input_wsi: str | Path, output_dir: str | Path, *, quality: int = 90) -> Path:
    from .trident import check_binary
    import subprocess

    input_wsi = Path(input_wsi)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if ".pyr." in input_wsi.name:
        return input_wsi
    output_path = output_dir / f"{input_wsi.stem}.pyr.tif"
    if output_path.exists():
        return output_path
    check_binary("vips")
    subprocess.run(
        [
            "vips",
            "tiffsave",
            str(input_wsi),
            str(output_path),
            "--tile",
            "--pyramid",
            "--bigtiff",
            "--compression",
            "jpeg",
            "--Q",
            str(quality),
        ],
        check=True,
    )
    return output_path


def _find_single_feature_h5(job_dir: str | Path, slide_stem: str) -> Path:
    matches = sorted(Path(job_dir).glob(f"**/{slide_stem}.h5"))
    if not matches:
        raise FileNotFoundError(f"could not find TRIDENT feature H5 for {slide_stem} under {job_dir}")
    if len(matches) > 1:
        raise RuntimeError(f"found multiple TRIDENT feature H5 files for {slide_stem} under {job_dir}: {matches}")
    return matches[0]


def _extract_handcrafted_from_wsi_and_h5(
    *,
    wsi_path: str | Path,
    h5_path: str | Path,
    selection_json: str | Path,
    patch_size_level0: int,
    target_patch_size: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    selection = load_selection_payload(selection_json)
    hc_cols_all = selection["hc_cols_all"]
    hc_keep = selection_hc_keep(selection)
    emb_keep = selection_embedding_keep(selection)
    feature_key = selection_feature_key(selection)
    embeddings, coords = load_h5_features(h5_path, feature_key=feature_key)
    if coords is None:
        raise KeyError(f"{h5_path} does not contain coords; raw WSI hybrid inference needs TRIDENT coords")
    slide = openslide.OpenSlide(str(wsi_path))
    try:
        config = TileCachingConfig(patch_size_level0=patch_size_level0, target_patch_size=target_patch_size)
        level = pick_level_for_patch(slide, patch_size_level0, max_level=config.max_level)
        downsample = float(slide.level_downsamples[level])
        patch_size_level = max(1, int(round(patch_size_level0 / downsample)))
        rows = []
        fused = []
        for patch_id, (x, y) in enumerate(coords):
            region = slide.read_region((int(x), int(y)), level, (patch_size_level, patch_size_level)).convert("RGB")
            region = region.resize((target_patch_size, target_patch_size), resample=Image.BILINEAR)
            tensor = _pil_to_u8_chw(region)
            hc_values, hc_names = extract_kba_features(tensor)
            hc_frame = pd.DataFrame([hc_values], columns=hc_names)
            missing = [column for column in hc_cols_all if column not in hc_frame.columns]
            if missing:
                raise KeyError(f"handcrafted feature mismatch; missing columns: {missing}")
            X_hc = hc_frame[hc_cols_all].to_numpy(dtype=np.float32)
            X_hc = X_hc[:, hc_keep] if hc_keep.size else np.zeros((1, 0), dtype=np.float32)
            X_emb = embeddings[patch_id : patch_id + 1].astype(np.float32)
            X_emb = X_emb[:, emb_keep] if emb_keep.size else np.zeros((1, 0), dtype=np.float32)
            fused.append(np.concatenate([X_hc, X_emb], axis=1))
            rows.append(
                {
                    "slide_id": Path(h5_path).stem,
                    "patch_id": int(patch_id),
                    "feature_row_idx": int(patch_id),
                    "x": int(x),
                    "y": int(y),
                }
            )
    finally:
        slide.close()
    return pd.DataFrame(rows), np.concatenate(fused, axis=0)


def _write_inference_provenance(output_json: str | Path, payload: dict) -> Path:
    return dump_json(output_json, payload)


def predict_hybrid_from_wsi(
    *,
    input_wsi: str | Path,
    output_dir: str | Path,
    trident_dir: str | Path,
    checkpoint_path: str | Path,
    scaler_path: str | Path,
    selection_json: str | Path,
    task: Task | str,
    patch_encoder: str,
    model_kind: str = "mlp",
    hidden_dim: int = 512,
    mpp: float = 0.25,
    mag: int = 10,
    patch_size: int = 512,
    patch_size_level0: int = 3072,
    target_patch_size: int = 512,
    quality: int = 90,
    batch_size: int = 256,
    gpu: int | None = None,
    device: str | None = None,
    slide_threshold: float = 0.5,
) -> dict:
    task = Task(task)
    output_dir = Path(output_dir)
    work_dir = output_dir / "hybrid_inference"
    prepared_wsi_dir = work_dir / "prepared_wsi"
    prepared_wsi = _ensure_pyramidal_single_wsi(input_wsi, prepared_wsi_dir, quality=quality)
    manifest_csv = work_dir / "manifest" / "single_wsi.csv"
    write_custom_wsi_manifest(prepared_wsi.parent, manifest_csv, mpp=mpp)
    trident_job_dir = work_dir / "trident" / f"{patch_encoder}_mag{mag}_ps{patch_size}"
    slide_stem = prepared_wsi.stem
    feature_h5 = _find_single_feature_h5(trident_job_dir, slide_stem) if trident_job_dir.exists() else None
    trident_reused = feature_h5 is not None and feature_h5.exists()
    if feature_h5 is None or not feature_h5.exists():
        run_trident_batch(
            trident_dir,
            wsi_dir=prepared_wsi.parent,
            custom_wsi_csv=manifest_csv,
            job_dir=trident_job_dir,
            patch_encoder=patch_encoder,
            mag=mag,
            patch_size=patch_size,
            task="all",
            gpu=gpu,
        )
        feature_h5 = _find_single_feature_h5(trident_job_dir, slide_stem)
        trident_reused = False
    meta_frame, X_raw = _extract_handcrafted_from_wsi_and_h5(
        wsi_path=prepared_wsi,
        h5_path=feature_h5,
        selection_json=selection_json,
        patch_size_level0=patch_size_level0,
        target_patch_size=target_patch_size,
    )
    scaler = joblib.load(scaler_path)
    X = scaler.transform(X_raw)
    device = resolve_torch_device(device)
    model = _load_hybrid_model(
        checkpoint_path=checkpoint_path,
        input_dim=X.shape[1],
        task=task,
        model_kind=model_kind,
        hidden_dim=hidden_dim,
        device=device,
    )
    probs = _predict_arrays(task, model, X, batch_size=batch_size, device=device)
    predictions_csv = output_dir / "hybrid_tile_predictions.csv"
    payload = _write_prediction_outputs(output_csv=predictions_csv, task=task, base_frame=meta_frame, probs=probs, y_true=None)
    slide_summary_json = output_dir / "hybrid_slide_summary.json"
    summarize_hybrid_predictions_by_slide(
        predictions_csv,
        slide_summary_json,
        task=task,
        binary_threshold=slide_threshold,
    )
    payload["slide_summary_json"] = str(slide_summary_json)
    payload["feature_h5"] = str(feature_h5)
    payload["prepared_wsi"] = str(prepared_wsi)
    payload["torch_device"] = device
    payload["trident_job_dir"] = str(trident_job_dir)
    payload["trident_reused"] = bool(trident_reused)
    payload["slide_threshold"] = float(slide_threshold)
    provenance_json = output_dir / "hybrid_inference_provenance.json"
    _write_inference_provenance(
        provenance_json,
        {
            "input_wsi": str(Path(input_wsi)),
            "prepared_wsi": str(prepared_wsi),
            "predictions_csv": str(predictions_csv),
            "slide_summary_json": str(slide_summary_json),
            "feature_h5": str(feature_h5),
            "trident_dir": str(Path(trident_dir)),
            "trident_job_dir": str(trident_job_dir),
            "trident_gpu_index": gpu,
            "trident_reused": bool(trident_reused),
            "patch_encoder": patch_encoder,
            "mag": int(mag),
            "patch_size": int(patch_size),
            "patch_size_level0": int(patch_size_level0),
            "target_patch_size": int(target_patch_size),
            "mpp": float(mpp),
            "quality": int(quality),
            "checkpoint_path": str(Path(checkpoint_path)),
            "scaler_path": str(Path(scaler_path)),
            "selection_json": str(Path(selection_json)),
            "task": task.value,
            "torch_device": device,
            "slide_threshold": float(slide_threshold),
        },
    )
    payload["provenance_json"] = str(provenance_json)
    return payload
