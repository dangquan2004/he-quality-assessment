# QC Model Bundle

This directory contains the bundled quality-control inference files used by `he-quality run-qc`.

Files:

- `model_manifest.json`: bundle metadata and file checksums
- `checkpoint.pt`: downstream PyTorch classifier
- `scaler.joblib`: fitted feature scaler for inference
- `selection.json`: saved handcrafted-plus-embedding feature selection

`model_manifest.json` also stores the preprocessing contract used for inference, including:

- `mpp`
- `mag`
- `patch_size`
- `patch_size_level0`
- `target_patch_size`
- `quality`
- `slide_threshold`

`he-quality run-qc` reads those values automatically, so normal users only need `--input-path` and `--output-dir`.

If you want to override this bundled model, pass:

```bash
--model-dir /path/to/other/model_dir
```

where that directory contains the same three files.
