from __future__ import annotations

import argparse
from pathlib import Path
from .labels import Task
from .presets import available_hybrid_inference_presets, get_hybrid_inference_preset, resolve_preset_artifact_path


def _task(value: str) -> Task:
    return Task(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="he-quality",
        description="Installable CLI for H&E quality assessment and artifact detection.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert_wsi = subparsers.add_parser("convert-wsi", help="Convert raw TIFF slides into pyramidal TIFFs with libvips.")
    convert_wsi.add_argument("--dataset-dir", required=True)
    convert_wsi.add_argument("--output-dir", required=True)
    convert_wsi.add_argument("--quality", type=int, default=90)

    manifest = subparsers.add_parser("build-manifest", help="Write a TRIDENT custom WSI manifest CSV.")
    manifest.add_argument("--wsi-dir", required=True)
    manifest.add_argument("--output-csv", required=True)
    manifest.add_argument("--mpp", type=float, required=True)

    trident = subparsers.add_parser("run-trident", help="Run TRIDENT over a WSI manifest.")
    trident.add_argument("--trident-dir", required=True)
    trident.add_argument("--wsi-dir", required=True)
    trident.add_argument("--custom-wsi-csv", required=True)
    trident.add_argument("--job-dir", required=True)
    trident.add_argument("--patch-encoder", required=True, choices=["uni_v2", "conch_v1"])
    trident.add_argument("--mag", type=int, default=10)
    trident.add_argument("--patch-size", type=int, default=512)
    trident.add_argument("--task", default="all")
    trident.add_argument("--gpu", type=int)

    merge = subparsers.add_parser("merge-embeddings", help="Concatenate matching UNI and CONCH H5 features.")
    merge.add_argument("--feature-dir-a", required=True)
    merge.add_argument("--feature-dir-b", required=True)
    merge.add_argument("--output-dir", required=True)

    cache = subparsers.add_parser("cache-tiles", help="Cache labeled tiles to local .pt tensors for downstream training.")
    cache.add_argument("--wsi-dir", required=True)
    cache.add_argument("--label-dir", required=True)
    cache.add_argument("--splits-json", required=True)
    cache.add_argument("--task", required=True, type=_task, choices=list(Task))
    cache.add_argument("--tile-cache-dir", required=True)
    cache.add_argument("--wsi-cache-dir", required=True)
    cache.add_argument("--patch-size-level0", type=int, default=3072)
    cache.add_argument("--target-patch-size", type=int, default=512)

    handcrafted = subparsers.add_parser("extract-handcrafted", help="Extract handcrafted KBA features from cached tiles.")
    handcrafted.add_argument("--meta-csv", required=True)
    handcrafted.add_argument("--output-csv", required=True)

    select = subparsers.add_parser("fit-fusion-selection", help="Fit Spearman feature selection for handcrafted + embedding fusion.")
    select.add_argument("--hc-csv", required=True)
    select.add_argument("--h5-dir", required=True)
    select.add_argument("--selection-json", required=True)
    select.add_argument("--task", required=True, type=_task, choices=list(Task))
    select.add_argument("--threshold", type=float, default=0.08)

    apply_selection = subparsers.add_parser("apply-fusion-selection", help="Apply saved selection indices and write fused NPZ files.")
    apply_selection.add_argument("--hc-csv", required=True)
    apply_selection.add_argument("--h5-dir", required=True)
    apply_selection.add_argument("--selection-json", required=True)
    apply_selection.add_argument("--output-dir", required=True)

    sklearn_train = subparsers.add_parser("train-sklearn", help="Train an sklearn baseline from handcrafted CSV features.")
    sklearn_train.add_argument("--train-csv", required=True)
    sklearn_train.add_argument("--val-csv", required=True)
    sklearn_train.add_argument("--test-csv", required=True)
    sklearn_train.add_argument("--output-dir", required=True)
    sklearn_train.add_argument("--task", required=True, type=_task, choices=list(Task))
    sklearn_train.add_argument("--estimator", default="svm", choices=["svm", "xgb"])
    sklearn_train.add_argument("--balance-train", action="store_true")
    sklearn_train.add_argument("--max-train-per-class", type=int)
    sklearn_train.add_argument("--experiment-name", default="feature_classifier")

    resnet = subparsers.add_parser("train-resnet", help="Train a ResNet classifier from cached tile tensors.")
    resnet.add_argument("--train-meta-csv", required=True)
    resnet.add_argument("--val-meta-csv", required=True)
    resnet.add_argument("--test-meta-csv", required=True)
    resnet.add_argument("--output-dir", required=True)
    resnet.add_argument("--task", required=True, type=_task, choices=list(Task))
    resnet.add_argument("--arch", default="resnet50", choices=["resnet18", "resnet34", "resnet50"])
    resnet.add_argument("--batch-size", type=int, default=32)
    resnet.add_argument("--epochs", type=int, default=10)
    resnet.add_argument("--lr", type=float, default=1e-4)
    resnet.add_argument("--balance-train", action="store_true")
    resnet.add_argument("--pretrained", action="store_true")
    resnet.add_argument("--experiment-name", default="resnet_classifier")

    embedding = subparsers.add_parser("train-embedding", help="Train an MLP or KAN classifier from H5 or fused NPZ features.")
    embedding.add_argument("--output-dir", required=True)
    embedding.add_argument("--task", required=True, type=_task, choices=list(Task))
    embedding.add_argument("--source-kind", required=True, choices=["h5", "npz"])
    embedding.add_argument("--model-kind", default="mlp", choices=["mlp", "kan"])
    embedding.add_argument("--hidden-dim", type=int, default=512)
    embedding.add_argument("--batch-size", type=int, default=256)
    embedding.add_argument("--epochs", type=int, default=20)
    embedding.add_argument("--lr", type=float, default=1e-3)
    embedding.add_argument("--balance-train", action="store_true")
    embedding.add_argument("--experiment-name", default="embedding_classifier")
    embedding.add_argument("--feature-dir")
    embedding.add_argument("--label-dir")
    embedding.add_argument("--splits-json")
    embedding.add_argument("--train-dir")
    embedding.add_argument("--val-dir")
    embedding.add_argument("--test-dir")

    infer_hybrid = subparsers.add_parser(
        "infer-hybrid-wsi",
        help="Run fused handcrafted+embedding inference on one WSI or a folder of WSIs, converting raw .ome.tiff to pyramidal TIFF as needed.",
    )
    infer_hybrid.add_argument(
        "--input-path",
        "--input-wsi",
        dest="input_path",
        required=True,
        help="Path to one WSI or a directory of WSIs.",
    )
    infer_hybrid.add_argument("--output-dir", required=True)
    infer_hybrid.add_argument("--trident-dir", required=True)
    infer_hybrid.add_argument("--preset", choices=available_hybrid_inference_presets())
    infer_hybrid.add_argument("--artifact-root")
    infer_hybrid.add_argument("--checkpoint-path")
    infer_hybrid.add_argument("--scaler-path")
    infer_hybrid.add_argument("--selection-json")
    infer_hybrid.add_argument("--task", type=_task, choices=list(Task))
    infer_hybrid.add_argument("--patch-encoder", choices=["uni_v2", "conch_v1"])
    infer_hybrid.add_argument("--model-kind", choices=["mlp", "kan"])
    infer_hybrid.add_argument("--hidden-dim", type=int)
    infer_hybrid.add_argument("--mpp", type=float, default=0.25)
    infer_hybrid.add_argument("--mag", type=int, default=10)
    infer_hybrid.add_argument("--patch-size", type=int, default=512)
    infer_hybrid.add_argument("--patch-size-level0", type=int, default=3072)
    infer_hybrid.add_argument("--target-patch-size", type=int, default=512)
    infer_hybrid.add_argument("--quality", type=int, default=90)
    infer_hybrid.add_argument("--batch-size", type=int, default=256)
    infer_hybrid.add_argument("--gpu", type=int)
    infer_hybrid.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    infer_hybrid.add_argument("--slide-threshold", type=float, default=0.5)

    run_qc = subparsers.add_parser(
        "run-qc",
        help="Recommended user-facing entrypoint: feed one WSI or a folder of WSIs in and get quality-control results out.",
    )
    run_qc.add_argument(
        "--input-path",
        required=True,
        help="Path to one WSI or a directory of WSIs.",
    )
    run_qc.add_argument("--output-dir", required=True)
    run_qc.add_argument("--trident-dir", default="external/TRIDENT")
    run_qc.add_argument("--artifact-root")
    run_qc.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    run_qc.add_argument("--gpu", type=int)

    return parser


def _handle_convert_wsi(args: argparse.Namespace) -> None:
    from .trident import convert_to_pyramidal_tiffs

    convert_to_pyramidal_tiffs(args.dataset_dir, args.output_dir, quality=args.quality)


def _handle_build_manifest(args: argparse.Namespace) -> None:
    from .trident import write_custom_wsi_manifest

    write_custom_wsi_manifest(args.wsi_dir, args.output_csv, mpp=args.mpp)


def _handle_run_trident(args: argparse.Namespace) -> None:
    from .trident import run_trident_batch

    run_trident_batch(
        args.trident_dir,
        wsi_dir=args.wsi_dir,
        custom_wsi_csv=args.custom_wsi_csv,
        job_dir=args.job_dir,
        patch_encoder=args.patch_encoder,
        mag=args.mag,
        patch_size=args.patch_size,
        task=args.task,
        gpu=args.gpu,
    )


def _handle_merge_embeddings(args: argparse.Namespace) -> None:
    from .trident import merge_feature_h5

    merge_feature_h5(args.feature_dir_a, args.feature_dir_b, args.output_dir)


def _handle_cache_tiles(args: argparse.Namespace) -> None:
    from .tiles import TileCachingConfig, build_tile_dataframe, cache_tiles_to_disk, split_tile_dataframe

    frame, splits = build_tile_dataframe(
        args.wsi_dir,
        args.label_dir,
        task=args.task,
        splits_json=args.splits_json,
        patch_size_level0=args.patch_size_level0,
    )
    split_frames = split_tile_dataframe(frame, splits)
    config = TileCachingConfig(
        patch_size_level0=args.patch_size_level0,
        target_patch_size=args.target_patch_size,
    )
    for split_name, split_frame in split_frames.items():
        cache_tiles_to_disk(
            split_frame,
            split_name=split_name,
            tile_cache_dir=args.tile_cache_dir,
            wsi_cache_dir=args.wsi_cache_dir,
            config=config,
        )


def _handle_extract_handcrafted(args: argparse.Namespace) -> None:
    from .handcrafted import featurize_meta_csv

    featurize_meta_csv(args.meta_csv, args.output_csv)


def _handle_fit_fusion_selection(args: argparse.Namespace) -> None:
    from .fusion import fit_spearman_selection

    fit_spearman_selection(
        args.hc_csv,
        args.h5_dir,
        args.selection_json,
        threshold=args.threshold,
        task=args.task,
    )


def _handle_apply_fusion_selection(args: argparse.Namespace) -> None:
    from .fusion import apply_selection_and_write_npz

    apply_selection_and_write_npz(args.hc_csv, args.h5_dir, args.output_dir, args.selection_json)


def _handle_train_sklearn(args: argparse.Namespace) -> None:
    from .train_sklearn import train_feature_classifier

    train_feature_classifier(
        args.train_csv,
        args.val_csv,
        args.test_csv,
        output_dir=args.output_dir,
        task=args.task,
        estimator=args.estimator,
        balance_train=args.balance_train,
        max_train_per_class=args.max_train_per_class,
        experiment_name=args.experiment_name,
    )


def _handle_train_resnet(args: argparse.Namespace) -> None:
    from .train_torch import train_resnet_classifier

    train_resnet_classifier(
        train_meta_csv=args.train_meta_csv,
        val_meta_csv=args.val_meta_csv,
        test_meta_csv=args.test_meta_csv,
        output_dir=args.output_dir,
        task=args.task,
        arch=args.arch,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        balance_train=args.balance_train,
        pretrained=args.pretrained,
        experiment_name=args.experiment_name,
    )


def _handle_train_embedding(args: argparse.Namespace) -> None:
    from .train_torch import train_embedding_classifier

    train_embedding_classifier(
        output_dir=args.output_dir,
        task=args.task,
        source_kind=args.source_kind,
        hidden_dim=args.hidden_dim,
        model_kind=args.model_kind,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        balance_train=args.balance_train,
        experiment_name=args.experiment_name,
        feature_dir=args.feature_dir,
        label_dir=args.label_dir,
        splits_json=args.splits_json,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        test_dir=args.test_dir,
    )


def _handle_infer_hybrid_wsi(args: argparse.Namespace) -> None:
    from .infer import predict_hybrid_from_path

    checkpoint_path = args.checkpoint_path
    scaler_path = args.scaler_path
    selection_json = args.selection_json
    task = args.task
    patch_encoder = args.patch_encoder
    model_kind = args.model_kind
    hidden_dim = args.hidden_dim

    if args.preset:
        preset = get_hybrid_inference_preset(args.preset)
        checkpoint_path = checkpoint_path or str(resolve_preset_artifact_path(preset.checkpoint_relpath, args.artifact_root))
        scaler_path = scaler_path or str(resolve_preset_artifact_path(preset.scaler_relpath, args.artifact_root))
        selection_json = selection_json or str(resolve_preset_artifact_path(preset.selection_relpath, args.artifact_root))
        task = task or preset.task
        patch_encoder = patch_encoder or preset.patch_encoder
        model_kind = model_kind or preset.model_kind
        hidden_dim = hidden_dim if hidden_dim is not None else preset.hidden_dim

    model_kind = model_kind or "mlp"
    hidden_dim = 512 if hidden_dim is None else hidden_dim

    missing = []
    if checkpoint_path is None:
        missing.append("--checkpoint-path")
    if scaler_path is None:
        missing.append("--scaler-path")
    if selection_json is None:
        missing.append("--selection-json")
    if task is None:
        missing.append("--task")
    if patch_encoder is None:
        missing.append("--patch-encoder")
    if missing:
        joined = ", ".join(missing)
        raise SystemExit(f"infer-hybrid-wsi is missing required arguments: {joined}. Use --preset or pass them explicitly.")

    payload = predict_hybrid_from_path(
        input_path=args.input_path,
        output_dir=args.output_dir,
        trident_dir=args.trident_dir,
        checkpoint_path=checkpoint_path,
        scaler_path=scaler_path,
        selection_json=selection_json,
        task=task,
        patch_encoder=patch_encoder,
        model_kind=model_kind,
        hidden_dim=hidden_dim,
        mpp=args.mpp,
        mag=args.mag,
        patch_size=args.patch_size,
        patch_size_level0=args.patch_size_level0,
        target_patch_size=args.target_patch_size,
        quality=args.quality,
        batch_size=args.batch_size,
        gpu=args.gpu,
        device=None if args.device == "auto" else args.device,
        slide_threshold=args.slide_threshold,
    )
    qc_results_json = payload.get("qc_results_json")
    if qc_results_json:
        print(f"Quality-control results written to {qc_results_json}")


def _handle_run_qc(args: argparse.Namespace) -> None:
    from .infer import predict_hybrid_from_path

    preset = get_hybrid_inference_preset("s4_new_multiclass")
    trident_dir = Path(args.trident_dir)
    payload = predict_hybrid_from_path(
        input_path=args.input_path,
        output_dir=args.output_dir,
        trident_dir=trident_dir,
        checkpoint_path=resolve_preset_artifact_path(preset.checkpoint_relpath, args.artifact_root),
        scaler_path=resolve_preset_artifact_path(preset.scaler_relpath, args.artifact_root),
        selection_json=resolve_preset_artifact_path(preset.selection_relpath, args.artifact_root),
        task=preset.task,
        patch_encoder=preset.patch_encoder,
        model_kind=preset.model_kind,
        hidden_dim=preset.hidden_dim,
        device=None if args.device == "auto" else args.device,
        gpu=args.gpu,
    )
    qc_results_json = payload.get("qc_results_json")
    if qc_results_json:
        print(f"Quality-control results written to {qc_results_json}")


COMMAND_HANDLERS = {
    "convert-wsi": _handle_convert_wsi,
    "build-manifest": _handle_build_manifest,
    "run-trident": _handle_run_trident,
    "merge-embeddings": _handle_merge_embeddings,
    "cache-tiles": _handle_cache_tiles,
    "extract-handcrafted": _handle_extract_handcrafted,
    "fit-fusion-selection": _handle_fit_fusion_selection,
    "apply-fusion-selection": _handle_apply_fusion_selection,
    "train-sklearn": _handle_train_sklearn,
    "train-resnet": _handle_train_resnet,
    "train-embedding": _handle_train_embedding,
    "infer-hybrid-wsi": _handle_infer_hybrid_wsi,
    "run-qc": _handle_run_qc,
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = COMMAND_HANDLERS.get(args.command)
    if handler is None:  # pragma: no cover
        parser.error(f"unknown command: {args.command}")
    handler(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
