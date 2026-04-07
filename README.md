# H&E Quality Assessment

Python package and CLI for H&E whole-slide quality control, plus the recovered `EBME398_ArtifactDetection` research workflow.

For most users, this repo has one main job:

- run quality control on whole-slide images

The user-facing inference command is:

```bash
he-quality run-qc --input-path /path/to/wsi_or_folder --output-dir /path/to/output
```

It accepts either:

- one raw WSI such as `.ome.tiff`
- or a folder of WSIs

and writes tile-level predictions plus a slide-level summary for each slide.

If you only want QC results from WSI files, start with the quick start below and stop after Step 7.

## Quick Start

### Step 1. Install System Dependencies

You need both:

- `OpenSlide`
- `libvips`

macOS:

```bash
brew install openslide libvips
```

Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install libopenslide-dev openslide-tools libvips-tools
```

### Step 2. Clone This Repo And Install It

Recommended Python:

- `3.10` or `3.11`

Do not use Python `3.12+` for the default inference workflow. `he-quality` and TRIDENT need to run in the same environment, and TRIDENT currently supports Python `<3.12`.

```bash
git clone https://github.com/dangquan2004/he-quality-assessment.git
cd he-quality-assessment
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

If `python3.11` is not available on your machine, use `python3.10` instead.

Optional extras:

```bash
python -m pip install '.[xgb]'
python -m pip install '.[kan]'
python -m pip install '.[dev]'
```

### Step 3. Clone TRIDENT And Install It Into The Same Environment

`he-quality` launches TRIDENT using the current Python interpreter, so TRIDENT must be installed into the same active environment as this repo.

```bash
git clone https://github.com/mahmoodlab/TRIDENT.git external/TRIDENT
python -m pip install -e external/TRIDENT
```

If you want to inspect the TRIDENT install itself, the TRIDENT repo also exposes `trident-doctor`.

### Step 4. Authenticate To Hugging Face For `uni_v2`

The default QC path uses `uni_v2`, which requires approved access to:

- `MahmoodLab/UNI2-h`

Install the Hugging Face CLI and log in:

```bash
python -m pip install -U huggingface_hub
hf auth login
```

If you prefer environment variables:

```bash
export HF_TOKEN=your_hugging_face_token
```

### Step 5. Use The Bundled QC Model Files

This repo already ships the QC inference bundle in:

```text
models/qc
```

By default, `run-qc` uses those bundled files automatically. You do not need to download a separate checkpoint bundle.

That bundle includes:

- `model_manifest.json`
- `checkpoint.pt`
- `scaler.joblib`
- `selection.json`

The manifest stores the preprocessing contract for the bundled model, so `run-qc` automatically uses the model's own:

- `mpp`
- `mag`
- `patch_size`
- `patch_size_level0`
- `target_patch_size`
- `quality`
- `slide_threshold`

Most users do not need to touch these files directly. You only need `--model-dir` if you want to override the bundled model with a different compatible model directory.

If you prefer an environment variable, you can also set:

```bash
export HE_QUALITY_MODEL_DIR=/path/to/model_dir
```

### Step 6. Run The Gate Check

Before your first inference run, check that the full stack is available:

```bash
he-quality doctor
```

If TRIDENT is not at `external/TRIDENT`:

```bash
he-quality doctor --trident-dir /path/to/TRIDENT
```

If you want to override the bundled model files:

```bash
he-quality doctor --model-dir /path/to/model_dir
```

`he-quality doctor` checks:

- Python version compatibility for the full inference stack
- `openslide` import
- `vips` on `PATH`
- TRIDENT callable from the current Python environment
- verified access to `MahmoodLab/UNI2-h`
- bundled QC model artifacts
- model-manifest checksums when `model_manifest.json` is present

### Step 7. Run Quality Control

Single slide:

```bash
he-quality run-qc \
  --input-path /path/to/slide.ome.tiff \
  --output-dir /path/to/output/slide_name
```

Folder of slides:

```bash
he-quality run-qc \
  --input-path /path/to/wsi_folder \
  --output-dir /path/to/output_folder
```

If TRIDENT is not at `external/TRIDENT`, add:

```bash
--trident-dir /path/to/TRIDENT
```

If you want to override the bundled model files, add:

```bash
--model-dir /path/to/model_dir
```

That is the standard deployment path. Most users do not need anything else in this README.

## What This Repo Does

- converts raw TIFF-like WSIs into pyramidal TIFF when needed
- runs TRIDENT feature extraction
- applies the packaged quality-control model
- writes tile-level and slide-level QC outputs
- still exposes training utilities for the recovered research workflow

The default QC path uses `uni_v2`. Advanced workflows can switch to other supported patch encoders such as `conch_v1`.

This repo does not pretrain UNI or CONCH. Those remain external dependencies.

## What `run-qc` Does Internally

For each slide, the command:

1. converts raw WSI to pyramidal TIFF if needed
2. runs TRIDENT on that slide
3. reads TRIDENT coordinates
4. extracts handcrafted features from the same coordinates
5. applies the model-bundle preprocessing contract automatically
6. applies the saved feature selection
7. applies the saved scaler
8. runs the downstream classifier
9. writes tile predictions, slide summary, and provenance

## Most Important Outputs

Single-slide run:

- `quality_control_results.json`
- `hybrid_tile_predictions.csv`
- `hybrid_slide_summary.json`
- `hybrid_inference_provenance.json`

Folder run:

- `quality_control_results.json`
- `batch_results.csv`
- one subfolder per slide under `output_dir/<slide_id>/`
- one root-level `hybrid_batch_summary.json`

The first file most users should open is:

- `quality_control_results.json`

If you want a spreadsheet-friendly summary for a folder run, use:

- `batch_results.csv`

## Troubleshooting

Common failure points:

- `vips` missing from `PATH`
- OpenSlide system libraries missing
- TRIDENT cloned but not installed into the current Python environment
- missing Hugging Face auth for `uni_v2`
- missing bundled or overridden QC model files
- multiple files in the input folder resolving to the same slide ID
- mixing checkpoint, scaler, and selection files from different runs in advanced mode

When in doubt, rerun:

```bash
he-quality doctor
```

## Advanced Inference

Use this only if you need full manual control over the inference artifacts. Most users should stay with `run-qc`.

- `infer-hybrid-wsi`

Advanced manual artifact wiring:

```bash
he-quality infer-hybrid-wsi \
  --input-path /path/to/slide.ome.tiff \
  --output-dir /path/to/output/slide_name \
  --trident-dir external/TRIDENT \
  --checkpoint-path /path/to/checkpoint.pt \
  --scaler-path /path/to/scaler.joblib \
  --selection-json /path/to/selection.json \
  --task multiclass \
  --patch-encoder uni_v2 \
  --model-kind mlp \
  --device auto
```

Use matching artifacts from the same model run. Do not mix a checkpoint, scaler, and feature-selection file from different runs.

## Training Overview

Inference is the main deployable surface. Most users can ignore this section.

For research use, training follows four phases:

1. Preprocess WSI and run TRIDENT
   - convert raw slides with `convert-wsi`
   - build a manifest with `build-manifest`
   - extract embeddings with `run-trident`
2. Build tile and handcrafted feature data
   - cache tiles with `cache-tiles`
   - extract KBA features with `extract-handcrafted`
3. Train baselines
   - handcrafted model with `train-sklearn`
   - CNN baseline with `train-resnet`
   - frozen-embedding baseline with `train-embedding --source-kind h5`
4. Train a fusion model
   - fit feature selection with `fit-fusion-selection`
   - materialize fused inputs with `apply-fusion-selection`
   - train the downstream head with `train-embedding --source-kind npz`

Representative commands:

```bash
# 1. Preprocess and run TRIDENT
he-quality convert-wsi --dataset-dir data/raw_wsi --output-dir data/wsi_pyr
he-quality build-manifest --wsi-dir data/wsi_pyr --output-csv data/manifests/custom_wsi.csv --mpp 0.25
he-quality run-trident --trident-dir external/TRIDENT --wsi-dir data/wsi_pyr --custom-wsi-csv data/manifests/custom_wsi.csv --job-dir outputs/trident_uni --patch-encoder uni_v2 --mag 10 --patch-size 512

# 2. Build handcrafted features
he-quality cache-tiles --wsi-dir data/wsi_pyr --label-dir data/labels --splits-json configs/splits/sr040_seed42_split.json --task binary --tile-cache-dir artifacts/tile_cache --wsi-cache-dir artifacts/wsi_cache
he-quality extract-handcrafted --meta-csv artifacts/tile_cache/train_meta.csv --output-csv artifacts/features/g1_kba_train.csv

# 3. Train a baseline
he-quality train-embedding --output-dir outputs/embedding_mlp --task binary --source-kind h5 --feature-dir outputs/trident_uni/10x_512px_0px_overlap/features_uni_v2 --label-dir data/labels --splits-json configs/splits/sr040_seed42_split.json

# 4. Train a fusion model
he-quality fit-fusion-selection --hc-csv artifacts/features/g1_kba_train.csv --h5-dir outputs/trident_uni/10x_512px_0px_overlap/features_uni_v2 --selection-json artifacts/fusion/selection.json --task binary
he-quality apply-fusion-selection --hc-csv artifacts/features/g1_kba_train.csv --h5-dir outputs/trident_uni/10x_512px_0px_overlap/features_uni_v2 --selection-json artifacts/fusion/selection.json --output-dir artifacts/fusion/train
# repeat apply-fusion-selection for each split you plan to use downstream
# for example: artifacts/fusion/val and artifacts/fusion/test
he-quality train-embedding --output-dir outputs/fusion_mlp --task binary --source-kind npz --train-dir artifacts/fusion/train --val-dir artifacts/fusion/val --test-dir artifacts/fusion/test
```

Expected per-slide label CSV for training:

- filename stem matches the WSI stem
- contains `x`
- contains `y` or `y0`
- contains `label` or `label_collapsed`
- optional `idx`

The main alignment contract is spatial: the CSV should provide coordinates that correspond to the extracted slide patches.

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
models/qc/                       bundled QC inference files
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
