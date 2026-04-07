# H&E Quality Assessment

Installable Python tooling for the recovered `EBME398_ArtifactDetection` workflow. This repo translates the original notebook pipeline into a normal package and CLI for preprocessing, training, and single-slide hybrid inference.

## What This Repo Is For

- WSI preprocessing into pyramidal TIFF
- TRIDENT feature extraction with `uni_v2` or `conch_v1`
- tile caching for image-model training
- handcrafted feature extraction
- frozen-feature and fusion-model training
- single-slide hybrid inference from raw WSI input

This repo does not re-pretrain UNI or CONCH. Those remain external dependencies.

## Current Deployment Target

The recommended deployment path is now the recovered `S4_new` multiclass hybrid model, not the old `G4` binary path.

Reason:

- `S4_new` has a recoverable checkpoint, scaler, and feature-selection artifact set
- `G4` does not currently have a recoverable matching selection file

For the recovered local artifacts, the matching `S4_new` trio is:

- selection: `source/working_dir/10x_512px_0px_overlap/experiments/Multi_class/S4_new/spearman_ovr_select_thr0.04.json`
- scaler: `source/working_dir/10x_512px_0px_overlap/experiments/Multi_class/S4_new/results_multiclass/scaler.joblib`
- checkpoint: `source/working_dir/10x_512px_0px_overlap/experiments/Multi_class/S4_new/results_multiclass/best_pt_mlp_multiclass.pt`

The repo now supports both the current `embedding_keep_idx` format and the older recovered `uni_keep_idx` format in selection JSON files.

## Installation

Recommended Python: `3.10+`

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

- `pip install .` installs only the Python package
- TRIDENT must be checked out separately
- `uni_v2` requires gated Hugging Face access to `MahmoodLab/UNI2-h`
- `conch_v1` also depends on gated external model access
- `train-sklearn --estimator xgb` needs `.[xgb]`
- `train-embedding --model-kind kan` needs `.[kan]`

## Inference

### What The Inference Command Does

`infer-hybrid-wsi` performs single-slide hybrid inference:

1. convert raw WSI to pyramidal TIFF if needed
2. run TRIDENT on that slide
3. extract handcrafted features from the same TRIDENT coordinates
4. apply the saved fusion selection
5. apply the saved scaler
6. run the saved downstream classifier
7. write tile predictions, slide summary, and provenance metadata

### Required Inputs

You need:

- one raw WSI such as `.ome.tiff`
- a local TRIDENT checkout
- a matching checkpoint `.pt`
- a matching scaler `.joblib`
- a matching fusion selection JSON
- OpenSlide and `vips`

The checkpoint, scaler, selection JSON, and encoder choice must come from the same model family.

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

Without Hugging Face authentication, `uni_v2` extraction will fail even if this repo is installed correctly.

### Recommended `S4_new` Multiclass Example

Generic command shape:

```bash
he-quality infer-hybrid-wsi \
  --input-wsi data/inference/SR999.ome.tiff \
  --output-dir outputs/inference/SR999 \
  --trident-dir external/TRIDENT \
  --checkpoint-path path/to/best_pt_mlp_multiclass.pt \
  --scaler-path path/to/scaler.joblib \
  --selection-json path/to/spearman_ovr_select_thr0.04.json \
  --task multiclass \
  --patch-encoder uni_v2 \
  --model-kind mlp \
  --device auto
```

For the recovered local artifacts in this repo clone, the concrete files are:

```bash
he-quality infer-hybrid-wsi \
  --input-wsi data/inference/SR999.ome.tiff \
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

Use `best_pt_mlp_multiclass.pt`, not the multihead checkpoint, for the current CLI path.

### Outputs

- `hybrid_tile_predictions.csv`
- `hybrid_slide_summary.json`
- `hybrid_inference_provenance.json`
- prepared pyramidal WSI under `hybrid_inference/prepared_wsi/`
- TRIDENT features under `hybrid_inference/trident/<encoder>_mag<mag>_ps<patch_size>/`

### Common Failure Points

- `vips` missing from `PATH`
- OpenSlide system libraries missing
- wrong or missing TRIDENT checkout
- missing Hugging Face auth for `uni_v2`
- mismatched checkpoint / scaler / selection JSON
- using the multihead `S4_new` checkpoint with the current single-head CLI
- using artifacts from different tasks or thresholds

## Training Overview

Inference is the main deployable surface. Training remains available, but the workflow is easier to follow as phases rather than one long numbered list.

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

### 2. Build Tile And Handcrafted Feature Data

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

## Repository Layout

```text
src/ebme398_artifact_detection/   package code
scripts/he_quality.py             script entrypoint
configs/splits/                   reusable split JSON
docs/recovered_workflow.md        notebook-to-package mapping
source/                           recovered local artifacts, ignored by git
analysis/                         scratch outputs, ignored by git
```

## Labels

- binary: `clean` vs `unclean`
- multiclass: `clean`, `tissue_damage`, `blurry+fold`

The code normalizes notebook-era variants such as `tissue_damge` and `fold+blur`.

## External References

- TRIDENT: <https://github.com/mahmoodlab/TRIDENT>
- UNI2-h model card: <https://huggingface.co/MahmoodLab/UNI2-h>
- UNI code repo: <https://github.com/mahmoodlab/UNI>
- CONCH model card: <https://huggingface.co/MahmoodLab/CONCH>
- CONCH code repo: <https://github.com/mahmoodlab/CONCH>

Both UNI and CONCH are gated for non-commercial academic research use and should not be redistributed.

## Recovery Notes

See `docs/recovered_workflow.md` for the notebook-to-package mapping and recovered data contracts.
