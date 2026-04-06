# Recovered Workflow

This package was reconstructed from the notebook-based `EBME398_ArtifactDetection` folder. The goal was to preserve the operational pipeline while removing notebook-only execution as a dependency.

## Notebook To Package Mapping

- `Pre-processing/Pre01_tissueSeg+Patch+Grading`
  - recovered as:
    - `convert-wsi`
    - `build-manifest`
    - `run-trident`
  - note:
    - the interactive grading UI was not kept as a notebook dependency
    - this repo now expects tile labels as CSV inputs

- `Pre-processing/Pre02_embedding`
  - recovered as:
    - `run-trident --patch-encoder conch_v1`

- `Pre-processing/UNI+CONCH.ipynb`
  - recovered as:
    - `merge-embeddings`

- `Pre-processing/Pre03_FeatureSelection_embedding.ipynb`
  - recovered as:
    - `fit-fusion-selection`
    - `apply-fusion-selection`

- `Pre-processing/Pre04_HC`
  - recovered as:
    - `extract-handcrafted`

- `Binary/G1/G1_handcraft*.ipynb`
  - recovered as:
    - `extract-handcrafted`
    - `train-sklearn`

- `Binary/G2/G2_FinetuneResNet50*.ipynb`
  - recovered as:
    - `cache-tiles`
    - `train-resnet`

- `Binary/G3/G3a_FrozenBac+MLP*.ipynb`
  - recovered as:
    - `train-embedding --source-kind h5 --model-kind mlp`

- `Binary/G3/G3b_FrozenBac+KAN*.ipynb`
  - recovered as:
    - `train-embedding --source-kind h5 --model-kind kan`

- `Multi_Class/S3_Frozen+MLP.ipynb`
  - recovered as:
    - `train-embedding --task multiclass`

## Recovered Split Policy

The original run artifacts included `sr040_seed42_split.json` with this rule:

- test set: `SR >= 040`
- train and validation: `SR < 040`
- validation fraction: `25%`
- random state: `42`

That split has been copied into `configs/splits/sr040_seed42_split.json` for reuse.

## Data Contracts That Matter

### WSI manifests

- `build-manifest` writes a CSV with columns:
  - `wsi`
  - `mpp`

### Tile label CSVs

Per-slide annotation CSVs should have:

- `x`
- `y` or `y0`
- `label` or `label_collapsed`
- optional `idx`

The CSV filename stem should match the slide stem derived from the WSI filename.

### Feature assumptions

- H5 embeddings are expected under dataset key `features`
- coordinates are read from `coords` when present
- fusion workflows should preserve handcrafted row coordinates so they can be validated against TRIDENT `coords`
- fused NPZ files are written with:
  - `X_fused`
  - `y`
  - `paths`
  - `coords`
  - `feature_row_idx`

## Scientific Limits

- This translation preserves pipeline mechanics, not every exploratory notebook branch.
- The package does not recreate notebook-era manual curation decisions unless those were encoded in files such as split JSON or label CSVs.
- The repo can reproduce downstream training workflows, but it does not claim to reproduce foundation-model pretraining for UNI or CONCH.
