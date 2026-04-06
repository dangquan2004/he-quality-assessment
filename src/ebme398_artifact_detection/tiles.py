from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import openslide
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .labels import Task, to_task_label
from .paths import normalize_slide_id_from_wsi


@dataclass
class TileCachingConfig:
    patch_size_level0: int = 3072
    target_patch_size: int = 512
    max_level: int = 2


def _require_columns(df: pd.DataFrame, cols: list[str], path: Path) -> None:
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise KeyError(f"missing columns {missing} in {path}; columns={list(df.columns)}")


def discover_wsi_label_pairs(wsi_dir: str | Path, label_dir: str | Path) -> list[tuple[str, Path, Path]]:
    wsi_dir = Path(wsi_dir)
    label_dir = Path(label_dir)
    label_map = {path.stem: path for path in sorted(label_dir.glob("*.csv"))}
    wsi_map: dict[str, Path] = {}
    for pattern in ("**/*.tif", "**/*.tiff", "**/*.svs", "**/*.ndpi", "**/*.mrxs"):
        for path in wsi_dir.glob(pattern):
            slide_id = normalize_slide_id_from_wsi(path)
            wsi_map.setdefault(slide_id, path)
    common = sorted(set(label_map) & set(wsi_map))
    return [(slide_id, wsi_map[slide_id], label_map[slide_id]) for slide_id in common]


def build_tile_dataframe(
    wsi_dir: str | Path,
    label_dir: str | Path,
    *,
    task: Task | str,
    splits_json: str | Path | None = None,
    patch_size_level0: int = 3072,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    task = Task(task)
    records: list[dict] = []
    pairs = discover_wsi_label_pairs(wsi_dir, label_dir)
    for slide_id, wsi_path, label_path in pairs:
        df = pd.read_csv(label_path)
        y_col = "y" if "y" in df.columns else "y0" if "y0" in df.columns else None
        label_col = "label" if "label" in df.columns else "label_collapsed" if "label_collapsed" in df.columns else None
        if y_col is None or label_col is None:
            raise KeyError(f"could not infer y/label columns for {label_path}")
        _require_columns(df, ["x", y_col, label_col], label_path)
        if "idx" not in df.columns:
            df["idx"] = np.arange(len(df), dtype=int)
        df["x"] = pd.to_numeric(df["x"], errors="coerce")
        df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
        df = df[df["x"].notna() & df[y_col].notna()].copy()
        df["y_label"] = [to_task_label(value, task) for value in df[label_col]]
        for row in df.itertuples(index=False):
            records.append(
                {
                    "slide_id": slide_id,
                    "tile_idx": int(row.idx),
                    "x": int(row.x),
                    "y0": int(getattr(row, y_col)),
                    "y_label": int(row.y_label),
                    "wsi_path": str(wsi_path),
                    "ps0": int(patch_size_level0),
                }
            )
    frame = pd.DataFrame(records)
    if frame.empty:
        raise RuntimeError(
            f"no matched tiles were built from wsi_dir={Path(wsi_dir)} and label_dir={Path(label_dir)}; "
            "check that slide stems and CSV names match"
        )
    if splits_json is None:
        return frame, {}
    splits_path = Path(splits_json)
    splits = json.loads(splits_path.read_text())
    slide_ids = set(frame["slide_id"].unique())
    filtered = {
        split_name: [slide_id for slide_id in slide_ids_for_split if slide_id in slide_ids]
        for split_name, slide_ids_for_split in splits.items()
    }
    return frame, filtered


def split_tile_dataframe(frame: pd.DataFrame, splits: dict[str, list[str]]) -> dict[str, pd.DataFrame]:
    output: dict[str, pd.DataFrame] = {}
    for split_name, slide_ids in splits.items():
        output[split_name] = frame[frame["slide_id"].isin(slide_ids)].reset_index(drop=True)
    return output


def ensure_local_wsi_copy(drive_path: str | Path, cache_dir: str | Path) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    drive_path = Path(drive_path)
    local_path = cache_dir / drive_path.name
    if local_path.exists():
        try:
            if local_path.stat().st_size == drive_path.stat().st_size:
                return local_path
        except FileNotFoundError:
            pass
        local_path.unlink(missing_ok=True)
    shutil.copy2(drive_path, local_path)
    return local_path


def pick_level_for_patch(slide: openslide.OpenSlide, patch_size_level0: int, max_level: int = 2) -> int:
    if slide.level_count <= 1:
        return 0
    level = 1
    if patch_size_level0 >= 2048 and slide.level_count > 2:
        level = 2
    return min(level, slide.level_count - 1, max_level)


def _pil_to_u8_chw(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image, dtype=np.uint8)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def cache_tiles_to_disk(
    split_frame: pd.DataFrame,
    *,
    split_name: str,
    tile_cache_dir: str | Path,
    wsi_cache_dir: str | Path,
    config: TileCachingConfig,
) -> Path:
    tile_cache_dir = Path(tile_cache_dir)
    out_dir = tile_cache_dir / split_name
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_csv = tile_cache_dir / f"{split_name}_meta.csv"
    if meta_csv.exists():
        return meta_csv
    rows: list[dict] = []
    for drive_wsi_path, group in split_frame.groupby("wsi_path", sort=False):
        local_wsi_path = ensure_local_wsi_copy(drive_wsi_path, wsi_cache_dir)
        slide = openslide.OpenSlide(str(local_wsi_path))
        try:
            patch_size_level0 = int(group["ps0"].iloc[0]) if "ps0" in group.columns else config.patch_size_level0
            level = pick_level_for_patch(slide, patch_size_level0, max_level=config.max_level)
            downsample = float(slide.level_downsamples[level])
            patch_size_level = max(1, int(round(patch_size_level0 / downsample)))
            for row in group.itertuples(index=False):
                region = slide.read_region((int(row.x), int(row.y0)), level, (patch_size_level, patch_size_level)).convert("RGB")
                region = region.resize((config.target_patch_size, config.target_patch_size), resample=Image.BILINEAR)
                tensor = _pil_to_u8_chw(region)
                save_path = out_dir / f"{row.slide_id}_{int(row.tile_idx):08d}.pt"
                torch.save(tensor, save_path)
                rows.append(
                    {
                        "path": str(save_path),
                        "y_label": int(row.y_label),
                        "slide_id": row.slide_id,
                        "tile_idx": int(row.tile_idx),
                        "x": int(row.x),
                        "y0": int(row.y0),
                    }
                )
        finally:
            slide.close()
    pd.DataFrame(rows).to_csv(meta_csv, index=False)
    return meta_csv


class CachedTileDataset(Dataset):
    def __init__(self, meta_csv: str | Path):
        self.meta = pd.read_csv(meta_csv)
        _require_columns(self.meta, ["path", "y_label"], Path(meta_csv))

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.meta.iloc[index]
        image = torch.load(row["path"]).float() / 255.0
        return image, int(row["y_label"])
