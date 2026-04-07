# QC Model Bundle

This directory contains the bundled quality-control inference files used by `he-quality run-qc`.

Files:

- `model_manifest.json`: bundle metadata and file checksums
- `checkpoint.pt`: downstream PyTorch classifier
- `scaler.joblib`: fitted feature scaler for inference
- `selection.json`: saved handcrafted-plus-embedding feature selection

If you want to override this bundled model, pass:

```bash
--model-dir /path/to/other/model_dir
```

where that directory contains the same three files.
