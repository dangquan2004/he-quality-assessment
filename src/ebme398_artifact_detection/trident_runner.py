from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CPU-safe TRIDENT wrapper for he-quality.")
    parser.add_argument("--trident-repo", required=True)
    parser.add_argument("--wsi-dir", "--wsi_dir", dest="wsi_dir", required=True)
    parser.add_argument(
        "--custom-wsi-csv",
        "--custom_wsi_csv",
        "--custom-list-of-wsis",
        "--custom_list_of_wsis",
        dest="custom_wsi_csv",
        required=True,
    )
    parser.add_argument("--job-dir", "--job_dir", dest="job_dir", required=True)
    parser.add_argument("--patch-encoder", "--patch_encoder", dest="patch_encoder", required=True)
    parser.add_argument("--mag", type=int, default=10)
    parser.add_argument("--patch-size", "--patch_size", dest="patch_size", type=int, default=512)
    parser.add_argument("--task", choices=["seg", "coords", "feat", "all"], default="all")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=64)
    parser.add_argument("--max-workers", "--max_workers", dest="max_workers", type=int, default=None)
    parser.add_argument("--segmenter", default="hest", choices=["hest", "grandqc", "otsu"])
    parser.add_argument("--seg-conf-thresh", "--seg_conf_thresh", dest="seg_conf_thresh", type=float, default=0.5)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument(
        "--min-tissue-proportion",
        "--min_tissue_proportion",
        dest="min_tissue_proportion",
        type=float,
        default=0.0,
    )
    return parser


def _resolve_device(torch_module, gpu: int | None) -> str:
    if torch_module.cuda.is_available():
        return f"cuda:{0 if gpu is None else gpu}"
    mps_backend = getattr(torch_module.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


def _install_safe_worker_policy(device: str) -> None:
    if device.startswith("cuda"):
        return
    import trident.IO as trident_io
    import trident.wsi_objects.WSI as trident_wsi_module

    def safe_get_num_workers(
        batch_size: int,
        factor: float = 0.75,
        fallback: int = 16,
        max_workers: int | None = None,
    ) -> int:
        if os.name == "nt" or max_workers == 0:
            return 0
        num_cores = os.cpu_count() or fallback
        num_workers = int(factor * num_cores)
        resolved_max_workers = (2 * batch_size) if max_workers is None else max_workers
        if resolved_max_workers <= 0:
            return 0
        return int(min(max(num_workers, 1), resolved_max_workers))

    trident_io.get_num_workers = safe_get_num_workers
    trident_wsi_module.get_num_workers = safe_get_num_workers


def _coords_dir(args: argparse.Namespace) -> str:
    return f"{args.mag}x_{args.patch_size}px_{args.overlap}px_overlap"


def main() -> None:
    args = _build_parser().parse_args()
    trident_repo = Path(args.trident_repo).resolve()
    if not trident_repo.exists():
        raise FileNotFoundError(f"TRIDENT repo not found: {trident_repo}")

    sys.path.insert(0, str(trident_repo))

    import torch
    from trident import Processor
    from trident.patch_encoder_models.load import encoder_factory as patch_encoder_factory
    from trident.segmentation_models.load import segmentation_model_factory

    device = _resolve_device(torch, args.gpu)
    worker_cap = 1 if not device.startswith("cuda") and args.max_workers in (None, 0) else args.max_workers
    wsi_worker_cap = 0 if not device.startswith("cuda") else args.max_workers
    _install_safe_worker_policy(device)

    processor = Processor(
        job_dir=args.job_dir,
        wsi_source=args.wsi_dir,
        custom_list_of_wsis=args.custom_wsi_csv,
        max_workers=worker_cap,
    )
    if wsi_worker_cap == 0:
        for wsi in processor.wsis:
            wsi.max_workers = 0

    coords_dir = _coords_dir(args)
    if args.task in {"seg", "all"}:
        segmentation_model = segmentation_model_factory(args.segmenter, confidence_thresh=args.seg_conf_thresh)
        processor.run_segmentation_job(
            segmentation_model,
            seg_mag=segmentation_model.target_mag,
            holes_are_tissue=True,
            batch_size=args.batch_size,
            device="cpu" if args.segmenter == "otsu" else device,
        )

    if args.task in {"coords", "all"}:
        processor.run_patching_job(
            target_magnification=args.mag,
            patch_size=args.patch_size,
            overlap=args.overlap,
            saveto=coords_dir,
            min_tissue_proportion=args.min_tissue_proportion,
        )

    if args.task in {"feat", "all"}:
        encoder = patch_encoder_factory(args.patch_encoder)
        if not device.startswith("cuda"):
            encoder.precision = torch.float32
        processor.run_patch_feature_extraction_job(
            coords_dir=coords_dir,
            patch_encoder=encoder,
            device=device,
            saveas="h5",
            batch_limit=args.batch_size,
        )


if __name__ == "__main__":  # pragma: no cover
    main()
