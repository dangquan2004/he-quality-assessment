from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .labels import Task


MODEL_DIR_ENV_VAR = "HE_QUALITY_MODEL_DIR"
LEGACY_ARTIFACT_ROOT_ENV_VAR = "HE_QUALITY_ARTIFACT_ROOT"


@dataclass(frozen=True)
class HybridInferencePreset:
    name: str
    task: Task
    patch_encoder: str
    model_kind: str
    hidden_dim: int
    selection_relpath: str
    scaler_relpath: str
    checkpoint_relpath: str


HYBRID_INFERENCE_PRESETS: dict[str, HybridInferencePreset] = {
    "s4_new_multiclass": HybridInferencePreset(
        name="s4_new_multiclass",
        task=Task.MULTICLASS,
        patch_encoder="uni_v2",
        model_kind="mlp",
        hidden_dim=512,
        selection_relpath="selection.json",
        scaler_relpath="scaler.joblib",
        checkpoint_relpath="checkpoint.pt",
    ),
}


def available_hybrid_inference_presets() -> tuple[str, ...]:
    return tuple(sorted(HYBRID_INFERENCE_PRESETS))


def get_hybrid_inference_preset(name: str) -> HybridInferencePreset:
    try:
        return HYBRID_INFERENCE_PRESETS[name]
    except KeyError as exc:  # pragma: no cover
        raise KeyError(f"unknown hybrid inference preset: {name}") from exc


def repo_default_model_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "models" / "qc"


def resolve_model_dir(model_dir: str | Path | None = None) -> Path:
    if model_dir is not None:
        root = Path(model_dir).expanduser().resolve()
    else:
        env_root = os.environ.get(MODEL_DIR_ENV_VAR) or os.environ.get(LEGACY_ARTIFACT_ROOT_ENV_VAR)
        if env_root:
            root = Path(env_root).expanduser().resolve()
        else:
            root = repo_default_model_dir().resolve()
    if not root.exists():
        raise FileNotFoundError(
            f"model directory does not exist: {root}. "
            f"Pass --model-dir or set {MODEL_DIR_ENV_VAR}."
        )
    return root


def resolve_preset_artifact_path(relative_path: str, model_dir: str | Path | None = None) -> Path:
    root = resolve_model_dir(model_dir)
    path = (root / relative_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"preset artifact not found: {path}")
    return path
