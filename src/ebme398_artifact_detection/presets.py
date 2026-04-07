from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .labels import Task


ARTIFACT_ROOT_ENV_VAR = "HE_QUALITY_ARTIFACT_ROOT"


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
        selection_relpath="10x_512px_0px_overlap/experiments/Multi_class/S4_new/spearman_ovr_select_thr0.04.json",
        scaler_relpath="10x_512px_0px_overlap/experiments/Multi_class/S4_new/results_multiclass/scaler.joblib",
        checkpoint_relpath="10x_512px_0px_overlap/experiments/Multi_class/S4_new/results_multiclass/best_pt_mlp_multiclass.pt",
    ),
}


def available_hybrid_inference_presets() -> tuple[str, ...]:
    return tuple(sorted(HYBRID_INFERENCE_PRESETS))


def get_hybrid_inference_preset(name: str) -> HybridInferencePreset:
    try:
        return HYBRID_INFERENCE_PRESETS[name]
    except KeyError as exc:  # pragma: no cover
        raise KeyError(f"unknown hybrid inference preset: {name}") from exc


def repo_default_artifact_root() -> Path:
    return Path(__file__).resolve().parents[2] / "source" / "working_dir"


def resolve_artifact_root(artifact_root: str | Path | None = None) -> Path:
    if artifact_root is not None:
        root = Path(artifact_root).expanduser().resolve()
    else:
        env_root = os.environ.get(ARTIFACT_ROOT_ENV_VAR)
        if env_root:
            root = Path(env_root).expanduser().resolve()
        else:
            root = repo_default_artifact_root().resolve()
    if not root.exists():
        raise FileNotFoundError(
            f"artifact root does not exist: {root}. "
            f"Pass --artifact-root or set {ARTIFACT_ROOT_ENV_VAR}."
        )
    return root


def resolve_preset_artifact_path(relative_path: str, artifact_root: str | Path | None = None) -> Path:
    root = resolve_artifact_root(artifact_root)
    path = (root / relative_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"preset artifact not found: {path}")
    return path
