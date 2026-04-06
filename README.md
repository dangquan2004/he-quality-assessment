# H&E quality assesment

Installable Python scripts for the recovered EBME398 artifact-detection workflow, translated out of notebooks into a normal package and CLI.

This repo keeps the original scientific intent, but it does not pretend to fully replace every notebook interaction. In particular, the manual tile-labeling notebook logic was not kept as a notebook dependency; the package assumes you already have tile-level CSV labels for training.

## What This Repo Covers

- Whole-slide preprocessing for pyramidal TIFF export.
- TRIDENT-based patch extraction and feature generation with `uni_v2` or `conch_v1`.
- Tile caching for image-level training.
- Handcrafted feature extraction from cached tiles.
- Frozen-feature training from H5 embeddings or fused NPZ features.
- End-to-end ResNet training from cached tiles.
- Binary and multiclass label normalization recovered from the original notebooks.

## What "Train From Scratch" Means Here

- For image training, `train-resnet` trains a downstream classifier from scratch by default. Add `--pretrained` only if you want ImageNet initialization.
- For embedding workflows, `train-embedding` trains the downstream head from scratch on frozen UNI, CONCH, or fused features.
- This repo does **not** re-pretrain UNI or CONCH themselves. Those foundation models remain external dependencies.

## External Dependencies

Recommended Python: `3.10+`.

System packages:

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

You can then run either the installed CLI:

```bash
he-quality --help
```

or the plain Python script entrypoint:

```bash
python scripts/he_quality.py --help
```

Quick preflight checks:

```bash
python -c "import openslide; print(openslide.__library_version__)"
vips --version
he-quality --help
python scripts/he_quality.py --help
```

Important install note:

- `pip install .` installs the Python package only.
- TRIDENT-based commands still require a separate TRIDENT checkout.
- `uni_v2` requires approved Hugging Face access to `MahmoodLab/UNI2-h` plus user authentication before TRIDENT can download weights.
- CONCH weights are also external gated dependencies and are not installed by this repo.
- `train-sklearn --estimator xgb` requires `python -m pip install '.[xgb]'`.
- `train-embedding --model-kind kan` requires `python -m pip install '.[kan]'`.

## Repository Layout

```text
src/ebme398_artifact_detection/   package code
scripts/he_quality.py             script entrypoint
configs/splits/                   reusable split JSON recovered from source runs
docs/recovered_workflow.md        mapping from old notebooks to new commands
source/                           downloaded raw project material, ignored by git
analysis/                         notebook extraction scratch space, ignored by git
```

## Inference

### What This Inference Pipeline Does

The deployed inference path in this repo is the fused handcrafted + embedding pipeline only.

Given one raw WSI such as `.ome.tiff`, the command:

- converts the slide to pyramidal TIFF when needed
- runs TRIDENT on that single slide
- extracts handcrafted features from the same TRIDENT coordinates
- applies the saved fusion feature selection
- loads the saved hybrid classifier and scaler
- writes tile-level predictions, slide-level summary output, and a provenance JSON

### Required Inputs Before Inference

You need all of the following:

- a raw WSI file such as `sample.ome.tiff`
- a trained hybrid checkpoint such as `embedding_classifier_best.pt`
- the matching scaler `.joblib`
- the matching fusion `selection.json`
- a local TRIDENT checkout
- system access to `openslide` and `vips`

The checkpoint, scaler, selection JSON, and embedding encoder must all come from the same trained hybrid model family.

### Hugging Face Authentication For `uni_v2`

If you run inference with:

```bash
--patch-encoder uni_v2
```

then TRIDENT will need access to the gated Hugging Face model `MahmoodLab/UNI2-h`.

That means the user must have:

- a Hugging Face account
- approved access to `MahmoodLab/UNI2-h`
- an authenticated environment before running TRIDENT

Typical setup:

```bash
python -m pip install -U huggingface_hub
huggingface-cli login
```

or:

```bash
export HF_TOKEN=your_hugging_face_token
```

If that authentication step is missing, TRIDENT-based `uni_v2` extraction will fail even if this repo itself installs correctly.

### Single-Slide Hybrid Inference Example

```bash
he-quality infer-hybrid-wsi \
  --input-wsi data/inference/SR999.ome.tiff \
  --output-dir outputs/inference/SR999 \
  --trident-dir external/TRIDENT \
  --checkpoint-path outputs/fusion_mlp/embedding_classifier_best.pt \
  --scaler-path outputs/fusion_mlp/embedding_classifier_scaler.joblib \
  --selection-json artifacts/fusion/selection.json \
  --task binary \
  --patch-encoder uni_v2 \
  --device auto \
  --slide-threshold 0.5
```

### Inference Outputs

- `outputs/inference/SR999/hybrid_tile_predictions.csv`
- `outputs/inference/SR999/hybrid_slide_summary.json`
- `outputs/inference/SR999/hybrid_inference_provenance.json`
- `outputs/inference/SR999/hybrid_inference/prepared_wsi/*.pyr.tif`
- `outputs/inference/SR999/hybrid_inference/trident/<encoder>_mag<mag>_ps<patch_size>/**/<slide>.h5`

### Inference Constraints

- The embedding source used at inference must match the encoder used during training.
- Fusion alignment is validated against TRIDENT coordinates when available.
- `--device` controls the downstream torch classifier only.
- TRIDENT extraction is still controlled by TRIDENT itself and the optional `--gpu` flag.
- `--slide-threshold` controls the binary slide-level summary threshold explicitly.
- This command does not run dual-encoder UNI+CONCH fusion directly from a raw slide in one step.

### Common Inference Failure Points

- missing `vips` on `PATH`
- missing OpenSlide system libraries
- missing or wrong TRIDENT checkout
- missing Hugging Face auth for `uni_v2`
- mismatched checkpoint, scaler, and selection JSON
- wrong encoder choice at inference time relative to training

## Quickstart

This section covers training and feature-preparation workflow. Inference is documented in the dedicated section above.

### 1. Convert raw WSI files to pyramidal TIFF

```bash
he-quality convert-wsi \
  --dataset-dir data/raw_wsi \
  --output-dir data/wsi_pyr
```

### 2. Build the TRIDENT manifest

```bash
he-quality build-manifest \
  --wsi-dir data/wsi_pyr \
  --output-csv data/manifests/custom_wsi.csv \
  --mpp 0.25
```

### 3. Run TRIDENT with UNI or CONCH

```bash
git clone https://github.com/mahmoodlab/TRIDENT.git external/TRIDENT
```

UNI:

```bash
he-quality run-trident \
  --trident-dir external/TRIDENT \
  --wsi-dir data/wsi_pyr \
  --custom-wsi-csv data/manifests/custom_wsi.csv \
  --job-dir outputs/trident_uni \
  --patch-encoder uni_v2 \
  --mag 10 \
  --patch-size 512
```

CONCH:

```bash
he-quality run-trident \
  --trident-dir external/TRIDENT \
  --wsi-dir data/wsi_pyr \
  --custom-wsi-csv data/manifests/custom_wsi.csv \
  --job-dir outputs/trident_conch \
  --patch-encoder conch_v1 \
  --mag 10 \
  --patch-size 512
```

If you extracted both encoders on the same tiles, merge them:

```bash
he-quality merge-embeddings \
  --feature-dir-a outputs/trident_uni/10x_512px_0px_overlap/features_uni_v2 \
  --feature-dir-b outputs/trident_conch/10x_512px_0px_overlap/features_conch_v1 \
  --output-dir outputs/features_uni_conch
```

### 4. Cache labeled tiles for image training

Expected label CSV contract per slide:

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
```

This produces `train_meta.csv`, `val_meta.csv`, and `test_meta.csv`.

### 5. Handcrafted baseline

```bash
he-quality extract-handcrafted \
  --meta-csv artifacts/tile_cache/train_meta.csv \
  --output-csv artifacts/features/g1_kba_train.csv

he-quality extract-handcrafted \
  --meta-csv artifacts/tile_cache/val_meta.csv \
  --output-csv artifacts/features/g1_kba_val.csv

he-quality extract-handcrafted \
  --meta-csv artifacts/tile_cache/test_meta.csv \
  --output-csv artifacts/features/g1_kba_test.csv
```

```bash
he-quality train-sklearn \
  --train-csv artifacts/features/g1_kba_train.csv \
  --val-csv artifacts/features/g1_kba_val.csv \
  --test-csv artifacts/features/g1_kba_test.csv \
  --output-dir outputs/handcrafted_svm \
  --task binary \
  --balance-train
```

### 6. ResNet image model

Train from scratch:

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

Fine-tune from ImageNet initialization:

```bash
he-quality train-resnet \
  --train-meta-csv artifacts/tile_cache/train_meta.csv \
  --val-meta-csv artifacts/tile_cache/val_meta.csv \
  --test-meta-csv artifacts/tile_cache/test_meta.csv \
  --output-dir outputs/resnet_imagenet \
  --task binary \
  --arch resnet50 \
  --epochs 20 \
  --pretrained
```

### 7. Frozen-feature models

Train directly from H5 embeddings:

```bash
he-quality train-embedding \
  --output-dir outputs/embedding_mlp \
  --task binary \
  --source-kind h5 \
  --feature-dir outputs/trident_uni/10x_512px_0px_overlap/features_uni_v2 \
  --label-dir data/labels \
  --splits-json configs/splits/sr040_seed42_split.json
```

Use KAN instead of MLP:

```bash
he-quality train-embedding \
  --output-dir outputs/embedding_kan \
  --task binary \
  --source-kind h5 \
  --model-kind kan \
  --feature-dir outputs/trident_uni/10x_512px_0px_overlap/features_uni_v2 \
  --label-dir data/labels \
  --splits-json configs/splits/sr040_seed42_split.json
```

### 8. Handcrafted + embedding fusion

Fit feature selection on the training set:

```bash
he-quality fit-fusion-selection \
  --hc-csv artifacts/features/g1_kba_train.csv \
  --h5-dir outputs/trident_uni/10x_512px_0px_overlap/features_uni_v2 \
  --selection-json artifacts/fusion/selection.json \
  --task binary
```

Apply it to train, val, and test:

```bash
he-quality apply-fusion-selection \
  --hc-csv artifacts/features/g1_kba_train.csv \
  --h5-dir outputs/trident_uni/10x_512px_0px_overlap/features_uni_v2 \
  --selection-json artifacts/fusion/selection.json \
  --output-dir artifacts/fusion/train
```

Then train on the fused NPZ files:

```bash
he-quality train-embedding \
  --output-dir outputs/fusion_mlp \
  --task binary \
  --source-kind npz \
  --train-dir artifacts/fusion/train \
  --val-dir artifacts/fusion/val \
  --test-dir artifacts/fusion/test
```

## Label Semantics Recovered From The Source

- Binary: `clean` vs `unclean`
- Multiclass: `clean`, `tissue_damage`, `blurry+fold`

Supported normalization includes notebook typos such as `tissue_damge` and order variants such as `fold+blur`.

## TRIDENT, UNI, and CONCH References

- TRIDENT framework: <https://github.com/mahmoodlab/TRIDENT>
- UNI2-h model card: <https://huggingface.co/MahmoodLab/UNI2-h>
- UNI code repo: <https://github.com/mahmoodlab/UNI>
- CONCH model card: <https://huggingface.co/MahmoodLab/CONCH>
- CONCH code repo: <https://github.com/mahmoodlab/CONCH>

Important constraint from the official model cards:

- `uni_v2` in this workflow depends on gated Hugging Face access to `MahmoodLab/UNI2-h`.
- UNI and CONCH weights are gated on Hugging Face.
- Both are released for non-commercial academic research use.
- Users should request access with an institutional email and should not redistribute the weights.

## Recovery Notes

See `docs/recovered_workflow.md` for how the original notebooks were translated into package modules and CLI commands.
