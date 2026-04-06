from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .paths import parse_patch_id, parse_wsi_stem_from_patch_path


def ensure_slide_id_column(df: pd.DataFrame) -> pd.DataFrame:
    if "slide_id" in df.columns:
        out = df.copy()
        out["slide_id"] = out["slide_id"].astype(str)
        return out
    if "path" not in df.columns:
        raise KeyError("expected either a slide_id column or a path column")
    out = df.copy()
    out["slide_id"] = out["path"].map(parse_wsi_stem_from_patch_path)
    return out


def _extract_coordinate_keys(group: pd.DataFrame) -> list[tuple[int, int]] | None:
    x_col = "x" if "x" in group.columns else None
    y_col = "y" if "y" in group.columns else "y0" if "y0" in group.columns else None
    if x_col is None or y_col is None:
        return None
    x = pd.to_numeric(group[x_col], errors="coerce")
    y = pd.to_numeric(group[y_col], errors="coerce")
    if x.isna().any() or y.isna().any():
        bad_rows = group.index[(x.isna() | y.isna())].tolist()[:5]
        raise RuntimeError(f"non-numeric handcrafted coordinates at rows {bad_rows}")
    return [(int(xv), int(yv)) for xv, yv in zip(x.to_numpy(), y.to_numpy())]


def _extract_patch_ids(group: pd.DataFrame) -> np.ndarray | None:
    if "patch_id" in group.columns:
        patch_ids = pd.to_numeric(group["patch_id"], errors="coerce")
    elif "tile_idx" in group.columns:
        patch_ids = pd.to_numeric(group["tile_idx"], errors="coerce")
    elif "path" in group.columns:
        try:
            patch_ids = group["path"].map(parse_patch_id)
        except ValueError:
            return None
    else:
        return None
    if isinstance(patch_ids, pd.Series):
        if patch_ids.isna().any():
            return None
        patch_ids = patch_ids.to_numpy()
    return np.asarray(patch_ids, dtype=int)


def _validate_patch_ids(feature_row_idx: np.ndarray, n_features: int, *, context: str) -> None:
    if feature_row_idx.size == 0:
        raise RuntimeError(f"no handcrafted rows were available to align for {context}")
    if np.any(feature_row_idx < 0):
        bad = feature_row_idx[feature_row_idx < 0][:5].tolist()
        raise RuntimeError(f"negative patch indices found for {context}: {bad}")
    if np.any(feature_row_idx >= n_features):
        bad = feature_row_idx[feature_row_idx >= n_features][:5].tolist()
        raise RuntimeError(
            f"patch indices exceed embedding rows for {context}: {bad}; n_features={n_features}"
        )


def align_handcrafted_rows_to_feature_rows(
    group: pd.DataFrame,
    *,
    coords: np.ndarray | None,
    n_features: int,
    context: str,
) -> tuple[pd.DataFrame, np.ndarray, str]:
    group = group.copy().reset_index(drop=True)
    coord_keys = _extract_coordinate_keys(group)
    patch_ids = _extract_patch_ids(group)

    if coords is not None and len(coords):
        coord_map: dict[tuple[int, int], int] = {}
        for idx, coord in enumerate(np.asarray(coords)):
            if len(coord) < 2:
                raise RuntimeError(f"coords array for {context} does not contain x/y pairs")
            key = (int(coord[0]), int(coord[1]))
            if key in coord_map:
                raise RuntimeError(f"duplicate TRIDENT coordinates for {context}: {key}")
            coord_map[key] = idx
        if coord_keys is not None:
            missing = [key for key in coord_keys if key not in coord_map]
            if missing:
                sample = missing[:5]
                raise RuntimeError(
                    f"handcrafted rows do not align to TRIDENT coords for {context}; "
                    f"missing coordinate keys such as {sample}"
                )
            feature_row_idx = np.asarray([coord_map[key] for key in coord_keys], dtype=int)
            if patch_ids is not None and not np.array_equal(patch_ids, feature_row_idx):
                mismatch = np.flatnonzero(patch_ids != feature_row_idx)[:5]
                sample = [
                    {
                        "row": int(idx),
                        "coord": coord_keys[int(idx)],
                        "patch_id": int(patch_ids[int(idx)]),
                        "feature_row_idx": int(feature_row_idx[int(idx)]),
                    }
                    for idx in mismatch
                ]
                raise RuntimeError(
                    f"patch-id alignment disagrees with coordinate alignment for {context}: {sample}"
                )
            mode = "coords"
        elif patch_ids is not None:
            feature_row_idx = patch_ids
            mode = "patch_id"
        else:
            raise RuntimeError(
                f"cannot align handcrafted rows for {context}; provide x/y coordinates or patch indices"
            )
    elif patch_ids is not None:
        feature_row_idx = patch_ids
        mode = "patch_id"
    else:
        raise RuntimeError(
            f"cannot align handcrafted rows for {context}; H5 coords are missing and no patch indices were found"
        )

    _validate_patch_ids(feature_row_idx, n_features, context=context)
    group["feature_row_idx"] = feature_row_idx
    group = group.sort_values("feature_row_idx").reset_index(drop=True)
    return group, group["feature_row_idx"].to_numpy(dtype=int), mode


def summarize_alignment_requirements() -> str:
    return (
        "Fusion workflows require either explicit x/y coordinates that match TRIDENT coords "
        "or patch indices that exactly match TRIDENT feature row indices."
    )
