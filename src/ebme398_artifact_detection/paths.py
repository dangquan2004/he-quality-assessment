from __future__ import annotations

import re
from pathlib import Path


PATCH_ID_RE = re.compile(r"_(\d+)\.pt$")
WSI_STEM_RE = re.compile(r"(.+\.ome\.pyr)_\d+\.pt$")


def normalize_slide_id_from_wsi(path: str | Path) -> str:
    name = Path(path).name
    name = re.sub(r"(\.ome\.pyr)\.tiff?$", r"\1", name, flags=re.IGNORECASE)
    name = re.sub(r"\.(svs|tiff?|ndpi|mrxs)$", "", name, flags=re.IGNORECASE)
    return name


def parse_patch_id(patch_path: str | Path) -> int:
    match = PATCH_ID_RE.search(str(patch_path))
    if not match:
        raise ValueError(f"could not parse patch id from {patch_path}")
    return int(match.group(1))


def parse_wsi_stem_from_patch_path(patch_path: str | Path) -> str:
    match = WSI_STEM_RE.search(str(patch_path))
    if not match:
        raise ValueError(f"could not parse WSI stem from {patch_path}")
    return Path(match.group(1)).name
