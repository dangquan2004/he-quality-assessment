from __future__ import annotations

import csv
import shutil
import subprocess
import sys
from pathlib import Path


WSI_PATTERNS = ("*.svs", "*.tif", "*.tiff", "*.ome.tif", "*.ome.tiff", "*.ndpi", "*.mrxs")
RAW_TIFF_PATTERNS = ("*.tif", "*.tiff", "*.ome.tif", "*.ome.tiff")


def convert_to_pyramidal_tiffs(dataset_dir: str | Path, output_dir: str | Path, *, quality: int = 90) -> list[Path]:
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    check_binary("vips")
    candidates: dict[str, Path] = {}
    for pattern in RAW_TIFF_PATTERNS:
        for path in dataset_dir.rglob(pattern):
            candidates[str(path.resolve())] = path.resolve()
    written: list[Path] = []
    for path in sorted(candidates.values()):
        out = output_dir / f"{path.stem}.pyr.tif"
        if out.exists():
            written.append(out)
            continue
        subprocess.run(
            [
                "vips",
                "tiffsave",
                str(path),
                str(out),
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
        written.append(out)
    return written


def write_custom_wsi_manifest(wsi_dir: str | Path, out_csv: str | Path, *, mpp: float) -> Path:
    wsi_dir = Path(wsi_dir)
    out_csv = Path(out_csv)
    paths: list[Path] = []
    for pattern in WSI_PATTERNS:
        paths.extend(wsi_dir.glob(pattern))
    rows = sorted({str(path.resolve()): path.resolve() for path in paths}.values())
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["wsi", "mpp"])
        writer.writeheader()
        for path in rows:
            writer.writerow({"wsi": str(path), "mpp": mpp})
    return out_csv


def run_trident_batch(
    trident_repo: str | Path,
    *,
    wsi_dir: str | Path,
    custom_wsi_csv: str | Path,
    job_dir: str | Path,
    patch_encoder: str,
    mag: int = 10,
    patch_size: int = 512,
    task: str = "all",
    gpu: int | None = None,
) -> None:
    trident_repo = Path(trident_repo)
    script = trident_repo / "run_batch_of_slides.py"
    if not script.exists():
        raise FileNotFoundError(f"TRIDENT entrypoint not found: {script}")
    cmd = [
        sys.executable,
        str(script),
        "--task",
        task,
        "--wsi_dir",
        str(Path(wsi_dir)),
        "--custom_list_of_wsis",
        str(Path(custom_wsi_csv)),
        "--job_dir",
        str(Path(job_dir)),
        "--patch_encoder",
        patch_encoder,
        "--mag",
        str(mag),
        "--patch_size",
        str(patch_size),
    ]
    if gpu is not None:
        cmd.extend(["--gpu", str(gpu)])
    subprocess.run(cmd, check=True, cwd=trident_repo)


def merge_feature_h5(feature_dir_a: str | Path, feature_dir_b: str | Path, output_dir: str | Path) -> list[Path]:
    import h5py
    import numpy as np

    feature_dir_a = Path(feature_dir_a)
    feature_dir_b = Path(feature_dir_b)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    common = sorted({path.name for path in feature_dir_a.glob("*.h5")} & {path.name for path in feature_dir_b.glob("*.h5")})
    written: list[Path] = []
    for name in common:
        path_a = feature_dir_a / name
        path_b = feature_dir_b / name
        out = output_dir / name
        with h5py.File(path_a, "r") as fa, h5py.File(path_b, "r") as fb, h5py.File(out, "w") as fout:
            merged = np.concatenate([fa["features"][:], fb["features"][:]], axis=1)
            fout.create_dataset("features", data=merged)
            if "coords" in fa:
                fout.create_dataset("coords", data=fa["coords"][:])
        written.append(out)
    return written


def check_binary(name: str) -> str:
    resolved = shutil.which(name)
    if resolved is None:
        raise FileNotFoundError(f"required executable not found on PATH: {name}")
    return resolved
