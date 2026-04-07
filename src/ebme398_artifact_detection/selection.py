from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_selection_payload(selection_json: str | Path) -> dict:
    payload = json.loads(Path(selection_json).read_text())
    if "embedding_keep_idx" not in payload and "uni_keep_idx" in payload:
        payload["embedding_keep_idx"] = payload["uni_keep_idx"]
    return payload


def selection_hc_keep(selection: dict) -> np.ndarray:
    return np.asarray(selection["hc_keep_idx"], dtype=int)


def selection_embedding_keep(selection: dict) -> np.ndarray:
    if "embedding_keep_idx" not in selection:
        raise KeyError("selection JSON must contain embedding_keep_idx or legacy uni_keep_idx")
    return np.asarray(selection["embedding_keep_idx"], dtype=int)


def selection_feature_key(selection: dict) -> str:
    return selection.get("feature_key", "features")
