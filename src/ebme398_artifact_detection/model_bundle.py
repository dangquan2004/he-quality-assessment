from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .labels import Task
from .presets import get_hybrid_inference_preset, resolve_model_dir


MODEL_MANIFEST_FILENAME = "model_manifest.json"


def file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_model_manifest(model_dir: str | Path) -> tuple[Path, dict]:
    model_dir = Path(model_dir)
    manifest_path = model_dir / MODEL_MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"model manifest not found: {manifest_path}")
    return manifest_path, json.loads(manifest_path.read_text())


def resolve_model_bundle(
    *,
    preset_name: str = "s4_new_multiclass",
    model_dir: str | Path | None = None,
    require_manifest: bool = False,
) -> dict:
    preset = get_hybrid_inference_preset(preset_name)
    root = resolve_model_dir(model_dir)

    manifest_path = root / MODEL_MANIFEST_FILENAME
    manifest = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    elif require_manifest:
        raise FileNotFoundError(f"model manifest not found: {manifest_path}")

    files_section = manifest.get("files", {}) if isinstance(manifest, dict) else {}
    checkpoint_relpath = files_section.get("checkpoint", {}).get("path", preset.checkpoint_relpath)
    scaler_relpath = files_section.get("scaler", {}).get("path", preset.scaler_relpath)
    selection_relpath = files_section.get("selection", {}).get("path", preset.selection_relpath)

    checkpoint_path = (root / checkpoint_relpath).resolve()
    scaler_path = (root / scaler_relpath).resolve()
    selection_path = (root / selection_relpath).resolve()
    for path in (checkpoint_path, scaler_path, selection_path):
        if not path.exists():
            raise FileNotFoundError(f"required model file not found: {path}")

    for key, path in {
        "checkpoint": checkpoint_path,
        "scaler": scaler_path,
        "selection": selection_path,
    }.items():
        expected_sha = files_section.get(key, {}).get("sha256")
        if expected_sha and file_sha256(path) != expected_sha:
            raise RuntimeError(f"{key} checksum mismatch for {path}")

    task = Task(manifest.get("task", preset.task.value) if isinstance(manifest, dict) else preset.task)
    patch_encoder = manifest.get("patch_encoder", preset.patch_encoder) if isinstance(manifest, dict) else preset.patch_encoder
    model_kind = manifest.get("model_kind", preset.model_kind) if isinstance(manifest, dict) else preset.model_kind
    hidden_dim = int(manifest.get("hidden_dim", preset.hidden_dim)) if isinstance(manifest, dict) else preset.hidden_dim
    preprocessing = manifest.get("preprocessing", {}) if isinstance(manifest, dict) else {}
    mpp = float(preprocessing.get("mpp", preset.mpp))
    mag = int(preprocessing.get("mag", preset.mag))
    patch_size = int(preprocessing.get("patch_size", preset.patch_size))
    patch_size_level0 = int(preprocessing.get("patch_size_level0", preset.patch_size_level0))
    target_patch_size = int(preprocessing.get("target_patch_size", preset.target_patch_size))
    quality = int(preprocessing.get("quality", preset.quality))
    slide_threshold = float(preprocessing.get("slide_threshold", preset.slide_threshold))

    return {
        "model_dir": root,
        "manifest_path": manifest_path if manifest_path.exists() else None,
        "manifest": manifest,
        "checkpoint_path": checkpoint_path,
        "scaler_path": scaler_path,
        "selection_json": selection_path,
        "task": task,
        "patch_encoder": patch_encoder,
        "model_kind": model_kind,
        "hidden_dim": hidden_dim,
        "mpp": mpp,
        "mag": mag,
        "patch_size": patch_size,
        "patch_size_level0": patch_size_level0,
        "target_patch_size": target_patch_size,
        "quality": quality,
        "slide_threshold": slide_threshold,
    }
