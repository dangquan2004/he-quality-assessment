# Change Report

## Scope

This document summarizes the recent changes made to the H&E quality-assessment repository to make the inference workflow usable, reproducible, and easier for a new user to run from GitHub.

Repository:

- `https://github.com/dangquan2004/he-quality-assessment`

Most recent related commits:

- `728b179` Bind QC preprocessing to model manifest
- `d363a5d` Stabilize TRIDENT doctor check
- `03d19ca` Harden hybrid inference smoke test path
- `9989579` Harden bundled QC model workflow
- `10f2253` Rename artifact override to model-dir

## What Changed

### 1. Simplified user-facing inference

The repository now exposes a clear deployment entrypoint:

```bash
he-quality run-qc --input-path /path/to/wsi_or_folder --output-dir /path/to/output
```

The goal was to hide model-internal details from normal users. A user no longer needs to manually provide:

- checkpoint path
- scaler path
- selection JSON path
- patch encoder
- preprocessing settings

Those are now resolved automatically from the bundled QC model.

### 2. Bundled QC model artifacts

The repository now ships the default QC model directly in:

- `models/qc/`

That bundle includes:

- `checkpoint.pt`
- `scaler.joblib`
- `selection.json`
- `model_manifest.json`

This makes the default inference path self-contained at the repo level, aside from external dependencies like TRIDENT, OpenSlide, libvips, and Hugging Face access for `UNI2-h`.

### 3. Bound preprocessing to the model bundle

The preprocessing contract is now stored in:

- `models/qc/model_manifest.json`

The manifest now records:

- `mpp`
- `mag`
- `patch_size`
- `patch_size_level0`
- `target_patch_size`
- `quality`
- `slide_threshold`

`run-qc` now reads those values from the bundled model instead of relying on hidden code defaults. This reduces the risk of silent drift between the trained model and the deployed preprocessing pipeline.

### 4. Hardened hybrid WSI inference

The raw-slide hybrid inference path was strengthened in several ways:

- raw `.ome.tiff` slides are converted to pyramidal TIFF when needed
- the pipeline handles one WSI or a folder of WSIs
- batch runs now emit `batch_results.csv`
- per-run provenance is written to `hybrid_inference_provenance.json`
- no-tissue slides now return a structured result instead of crashing
- stale or partial TRIDENT output directories no longer break reruns as easily

The code also now handles the recovered legacy selection format where the saved JSON used `uni_keep_idx` instead of `embedding_keep_idx`.

### 5. Safer TRIDENT execution on non-CUDA systems

Inference was tested on Apple Silicon with `mps`, not CUDA.

To support that path more safely, the repo now uses a wrapper:

- `src/ebme398_artifact_detection/trident_runner.py`

This wrapper avoids depending blindly on stock TRIDENT batch behavior in a non-CUDA environment and makes the repo’s own inference path more robust on macOS.

### 6. Better gate checks

The `doctor` command was expanded and hardened. It now checks:

- Python version compatibility
- OpenSlide import
- `vips` availability
- TRIDENT importability from the active environment
- verified access to `MahmoodLab/UNI2-h`
- bundled model presence and checksum validation

The TRIDENT check was changed from a fragile `run_batch_of_slides.py --help` probe to a direct import-based validation because the help-based check could fail in a fresh environment even when actual inference worked.

### 7. Installation and README cleanup

The README was rewritten to be more deployment-first and easier to follow for a user who only has WSI TIFF files.

The current flow is:

1. install system dependencies
2. install this repo
3. install TRIDENT in the same environment
4. authenticate to Hugging Face
5. run `he-quality doctor`
6. run `he-quality run-qc`

The docs now emphasize Python `3.10` or `3.11`, since the full inference stack depends on TRIDENT and TRIDENT currently does not support Python `3.12+`.

## Smoke Tests Performed

### Real-slide smoke test

The inference path was run successfully on a real slide from:

- `REVA_GUI/segmentation/input/original_HE/SR007-CR2-07d09d-100um-HE-20231031-s1.ome.tiff`

Observed result from the fresh clean test:

- `slide_pred_label`: `clean`
- `n_tiles`: `6`
- `prob_clean`: `0.8825645666666667`
- `prob_tissue_damage`: `0.11697974238814925`
- `prob_blurry+fold`: `0.00045571495478348624`

The run completed on:

- `mps`

### Clean-environment validation

The pipeline was rerun from scratch in a fresh Python `3.11` environment with:

- a fresh virtual environment
- a fresh TRIDENT clone
- a fresh install of this repo
- a fresh `doctor` run
- a real-slide `run-qc` execution

The manifest-driven preprocessing values were confirmed to match the values recorded in the emitted provenance JSON.

### Synthetic edge-case validation

Synthetic slides were also used during debugging to validate failure handling, especially:

- slides with no detected tissue
- reruns into partially populated output directories

Those temporary files were removed after testing and were not committed.

## Important Fixes To Silent Failure Modes

The most important fixes were not cosmetic. They were aimed at silent result corruption or misleading user behavior:

- bundled model resolution now works reliably in clone/install layouts
- no-tissue slides no longer crash with a missing-feature-H5 error
- partial TRIDENT cache directories are handled more safely
- the QC model now carries its own preprocessing contract
- doctor false negatives were reduced
- scikit-learn version mismatch warnings were fixed by pinning the compatible version family for the bundled scaler

## Remaining External Requirements

The repo is more self-contained than before, but these still remain external requirements:

- `OpenSlide`
- `libvips`
- TRIDENT installed in the same environment
- Hugging Face authentication with approved access to `MahmoodLab/UNI2-h`

## Current Outcome

The repository is now in a materially better state for deployment:

- a new user can run QC with a much simpler interface
- the shipped model bundle is internally tied to its preprocessing settings
- the inference path has been smoke-tested on a real WSI
- the main failure modes discovered during testing were fixed in code, not just documented
