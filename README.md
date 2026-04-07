# H&E Quality Assessment

Python package and CLI for the recovered `EBME398_ArtifactDetection` workflow.

This repo currently has one clear deployment path:

- `S4_new` multiclass hybrid inference

That path takes either:

- one raw WSI such as `.ome.tiff`
- or a folder of WSIs

and produces tile-level predictions plus a slide-level summary for each slide.

## Start Here: I Only Have WSI TIFF Files

If you are a new user and all you have is one WSI TIFF or a folder of WSI TIFFs, this is the path you want.

You do not need to:

- train a model
- prepare label CSVs
- manually choose a checkpoint, scaler, or selection file

You do need to:

1. install this repo
2. install TRIDENT
3. authenticate to Hugging Face for `uni_v2`
4. run `run-qc` on your slide or folder

What the repo handles for you:

- converts raw `.tif`, `.tiff`, `.ome.tif`, or `.ome.tiff` to pyramidal TIFF if needed
- runs TRIDENT feature extraction
- applies the recovered `S4_new` multiclass hybrid model
- writes predictions for each slide

## Quick Start

If you only want inference, do these four things:

1. install the package and system dependencies
2. clone TRIDENT
3. authenticate to Hugging Face for `uni_v2`
4. run `run-qc`

Single-slide example:

```bash
he-quality run-qc \
  --input-path data/inference/SR999.ome.tiff \
  --output-dir outputs/inference/SR999
```

Folder-of-slides example:

```bash
he-quality run-qc \
  --input-path data/inference_wsis \
  --output-dir outputs/inference_batch
```

For folder input, the CLI writes:

- `output_dir/<slide_id>/quality_control_results.json`
- `output_dir/<slide_id>/hybrid_tile_predictions.csv`
- `output_dir/<slide_id>/hybrid_slide_summary.json`
- `output_dir/<slide_id>/hybrid_inference_provenance.json`
- `output_dir/quality_control_results.json`
- `output_dir/hybrid_batch_summary.json`

If TRIDENT is not at `external/TRIDENT`, add:

```bash
--trident-dir /path/to/TRIDENT
```

If the recovered artifacts are not at `source/working_dir`, add:

```bash
--artifact-root /path/to/working_dir
```

## What This Repo Does

- converts raw TIFF-like WSIs into pyramidal TIFF when needed
- runs TRIDENT feature extraction with `uni_v2` or `conch_v1`
- caches labeled tiles for image-model training
- extracts handcrafted features
- trains handcrafted, CNN, frozen-feature, and fusion models
- runs hybrid inference on one WSI or a folder of WSIs

This repo does not pretrain UNI or CONCH. Those remain external dependencies.

## Installation

Recommended Python:

- `3.10+`

System dependencies:

- `openslide`
- `libvips`

Examples:

- macOS: `brew install openslide libvips`
- Ubuntu/Debian: `sudo apt-get install libopenslide-dev openslide-tools libvips-tools`

Clone and install:

```bash
git clone https://github.com/dangquan2004/he-quality-assessment.git
cd he-quality-assessment
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

Optional extras:

```bash
python -m pip install '.[xgb]'
python -m pip install '.[kan]'
python -m pip install '.[dev]'
```

Basic checks:

```bash
python -c "import openslide; print(openslide.__library_version__)"
vips --version
he-quality --help
python scripts/he_quality.py --help
```

Important:

- `pip install .` installs only this Python package
- TRIDENT must be installed separately
- `uni_v2` requires gated Hugging Face access to `MahmoodLab/UNI2-h`
- `conch_v1` also depends on gated external model access
- `train-sklearn --estimator xgb` needs `.[xgb]`
- `train-embedding --model-kind kan` needs `.[kan]`

## Before You Run Inference

Make sure all of these are true:

- your WSI is a supported file such as `.tif`, `.tiff`, `.ome.tif`, `.ome.tiff`, `.svs`, `.ndpi`, or `.mrxs`
- `vips` works in the terminal
- `openslide` imports in Python
- you have a local TRIDENT checkout
- you have Hugging Face access to `MahmoodLab/UNI2-h` if you use `uni_v2`
- the recovered `working_dir` artifacts are available either at `source/working_dir` or through `--artifact-root`

## Inference

The recommended user-facing command is:

- `run-qc`

The lower-level advanced command is:

- `infer-hybrid-wsi`

### Deployment Model

The current deployable model is the recovered `S4_new` multiclass hybrid pipeline.

Why this is the default:

- `S4_new` has a recoverable checkpoint, scaler, and selection file
- the old `G4` binary hybrid path does not currently have a recoverable matching selection file

The matching recovered `S4_new` artifact trio is:

- selection: `10x_512px_0px_overlap/experiments/Multi_class/S4_new/spearman_ovr_select_thr0.04.json`
- scaler: `10x_512px_0px_overlap/experiments/Multi_class/S4_new/results_multiclass/scaler.joblib`
- checkpoint: `10x_512px_0px_overlap/experiments/Multi_class/S4_new/results_multiclass/best_pt_mlp_multiclass.pt`

`run-qc` uses the `s4_new_multiclass` preset internally, so a new user does not need to manually wire these files.

### What `run-qc` Does

For each slide, the command:

1. converts raw WSI to pyramidal TIFF if needed
2. runs TRIDENT on that slide
3. reads TRIDENT coordinates
4. extracts handcrafted features from the same coordinates
5. applies the saved feature selection
6. applies the saved scaler
7. runs the downstream classifier
8. writes tile predictions, slide summary, and provenance

### What You Need To Provide

As a user, you only need to provide:

- one WSI file or one folder of WSIs
- a TRIDENT checkout
- OpenSlide and `vips`
- Hugging Face authentication if you use `uni_v2`

If you use the preset, the repo resolves the recovered checkpoint, scaler, and selection JSON automatically.

### Artifact Root

The preset expects an artifact root containing the recovered `working_dir` tree.

Default location inside this repo:

```text
source/working_dir
```

If your recovered artifacts live elsewhere, either pass:

```bash
--artifact-root /path/to/working_dir
```

or set:

```bash
export HE_QUALITY_ARTIFACT_ROOT=/path/to/working_dir
```

### Hugging Face Authentication For `uni_v2`

If you use:

```bash
--patch-encoder uni_v2
```

then TRIDENT needs access to the gated Hugging Face model `MahmoodLab/UNI2-h`.

Typical setup:

```bash
python -m pip install -U huggingface_hub
huggingface-cli login
```

or:

```bash
export HF_TOKEN=your_hugging_face_token
```

Without Hugging Face authentication, `uni_v2` feature extraction will fail.

### Recommended Commands

Single slide:

```bash
he-quality run-qc \
  --input-path data/inference/SR999.ome.tiff \
  --output-dir outputs/inference/SR999
```

Folder of slides:

```bash
he-quality run-qc \
  --input-path data/inference_wsis \
  --output-dir outputs/inference_batch
```

If TRIDENT is not checked out at `external/TRIDENT`:

```bash
he-quality run-qc \
  --input-path data/inference_wsis \
  --output-dir outputs/inference_batch \
  --trident-dir /path/to/TRIDENT
```

Advanced manual artifact wiring is still supported through `infer-hybrid-wsi`:

```bash
he-quality infer-hybrid-wsi \
  --input-path data/inference/SR999.ome.tiff \
  --output-dir outputs/inference/SR999 \
  --trident-dir external/TRIDENT \
  --checkpoint-path source/working_dir/10x_512px_0px_overlap/experiments/Multi_class/S4_new/results_multiclass/best_pt_mlp_multiclass.pt \
  --scaler-path source/working_dir/10x_512px_0px_overlap/experiments/Multi_class/S4_new/results_multiclass/scaler.joblib \
  --selection-json source/working_dir/10x_512px_0px_overlap/experiments/Multi_class/S4_new/spearman_ovr_select_thr0.04.json \
  --task multiclass \
  --patch-encoder uni_v2 \
  --model-kind mlp \
  --device auto
```

Use `best_pt_mlp_multiclass.pt`, not the multihead checkpoint, for the current CLI.

### Outputs

Single-slide run:

- `quality_control_results.json`
- `hybrid_tile_predictions.csv`
- `hybrid_slide_summary.json`
- `hybrid_inference_provenance.json`
- prepared pyramidal WSI under `hybrid_inference/prepared_wsi/`
- TRIDENT features under `hybrid_inference/trident/<encoder>_mag<mag>_ps<patch_size>/`

The file most users care about first is:

- `quality_control_results.json`

Folder run:

- one root-level `quality_control_results.json`
- one subfolder per slide under `output_dir/<slide_id>/`
- each subfolder contains the same single-slide outputs
- one root-level `hybrid_batch_summary.json`

For a folder run, the main top-level summary is:

- `hybrid_batch_summary.json`

### Common Failure Points

- `vips` missing from `PATH`
- OpenSlide system libraries missing
- wrong or missing TRIDENT checkout
- missing Hugging Face auth for `uni_v2`
- missing `source/working_dir` artifacts when using `--preset` without `--artifact-root`
- multiple files in the input folder resolving to the same slide ID
- mismatched checkpoint, scaler, and selection JSON
- using the multihead `S4_new` checkpoint with the current single-head CLI

## Training Overview

Inference is the main deployable surface. Training is still available, but it is easier to think about in four phases.

### 1. Preprocess WSI And Run TRIDENT

```bash
he-quality convert-wsi \
  --dataset-dir data/raw_wsi \
  --output-dir data/wsi_pyr

he-quality build-manifest \
  --wsi-dir data/wsi_pyr \
  --output-csv data/manifests/custom_wsi.csv \
  --mpp 0.25

git clone https://github.com/mahmoodlab/TRIDENT.git external/TRIDENT

he-quality run-trident \
  --trident-dir external/TRIDENT \
  --wsi-dir data/wsi_pyr \
  --custom-wsi-csv data/manifests/custom_wsi.csv \
  --job-dir outputs/trident_uni \
  --patch-encoder uni_v2 \
  --mag 10 \
  --patch-size 512
```

### 2. Build Tile And Handcrafted Features

Expected per-slide label CSV:

- filename stem matches the WSI stem
- contains `x`
- contains `y` or `y0`
- contains `label` or `label_collapsed`
- optional `idx`

```bash
he-quality cache-tiles \
  --wsi-dir data/wsi_pyr \
  --label-dir data/labels \
  --splits-json configs/splits/sr040_seed42_split.json \
  --task binary \
  --tile-cache-dir artifacts/tile_cache \
  --wsi-cache-dir artifacts/wsi_cache

he-quality extract-handcrafted \
  --meta-csv artifacts/tile_cache/train_meta.csv \
  --output-csv artifacts/features/g1_kba_train.csv
```

### 3. Train Baselines

Handcrafted:

```bash
he-quality train-sklearn \
  --train-csv artifacts/features/g1_kba_train.csv \
  --val-csv artifacts/features/g1_kba_val.csv \
  --test-csv artifacts/features/g1_kba_test.csv \
  --output-dir outputs/handcrafted_svm \
  --task binary \
  --balance-train
```

ResNet:

```bash
he-quality train-resnet \
  --train-meta-csv artifacts/tile_cache/train_meta.csv \
  --val-meta-csv artifacts/tile_cache/val_meta.csv \
  --test-meta-csv artifacts/tile_cache/test_meta.csv \
  --output-dir outputs/resnet_scratch \
  --task binary \
  --arch resnet50 \
  --epochs 20
```

Frozen embeddings:

```bash
he-quality train-embedding \
  --output-dir outputs/embedding_mlp \
  --task binary \
  --source-kind h5 \
  --feature-dir outputs/trident_uni/10x_512px_0px_overlap/features_uni_v2 \
  --label-dir data/labels \
  --splits-json configs/splits/sr040_seed42_split.json
```

### 4. Train A Fusion Model

Fit feature selection:

```bash
he-quality fit-fusion-selection \
  --hc-csv artifacts/features/g1_kba_train.csv \
  --h5-dir outputs/trident_uni/10x_512px_0px_overlap/features_uni_v2 \
  --selection-json artifacts/fusion/selection.json \
  --task binary
```

Apply it:

```bash
he-quality apply-fusion-selection \
  --hc-csv artifacts/features/g1_kba_train.csv \
  --h5-dir outputs/trident_uni/10x_512px_0px_overlap/features_uni_v2 \
  --selection-json artifacts/fusion/selection.json \
  --output-dir artifacts/fusion/train
```

Train the downstream head:

```bash
he-quality train-embedding \
  --output-dir outputs/fusion_mlp \
  --task binary \
  --source-kind npz \
  --train-dir artifacts/fusion/train \
  --val-dir artifacts/fusion/val \
  --test-dir artifacts/fusion/test
```

## Labels

- binary: `clean` vs `unclean`
- multiclass: `clean`, `tissue_damage`, `blurry+fold`

The code normalizes notebook-era variants such as `tissue_damge` and `fold+blur`.

## Repository Layout

```text
src/ebme398_artifact_detection/   package code
scripts/he_quality.py             script entrypoint
configs/splits/                   reusable split JSON
docs/recovered_workflow.md        notebook-to-package mapping
source/                           recovered local artifacts, ignored by git
analysis/                         scratch outputs, ignored by git
```

## External References

- TRIDENT: <https://github.com/mahmoodlab/TRIDENT>
- UNI2-h model card: <https://huggingface.co/MahmoodLab/UNI2-h>
- UNI code repo: <https://github.com/mahmoodlab/UNI>
- CONCH model card: <https://huggingface.co/MahmoodLab/CONCH>
- CONCH code repo: <https://github.com/mahmoodlab/CONCH>

Both UNI and CONCH are gated for non-commercial academic research use and should not be redistributed.

## Recovery Notes

See `docs/recovered_workflow.md` for the notebook-to-package mapping and recovered data contracts.
