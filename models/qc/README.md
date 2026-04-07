# QC Model Bundle

This directory contains the bundled quality-control inference files used by `he-quality run-qc`.

Files:

- `checkpoint.pt`: downstream PyTorch classifier
- `scaler.joblib`: fitted feature scaler for inference
- `selection.json`: saved handcrafted-plus-embedding feature selection

If you want to override this bundled model, pass:

```bash
--artifact-root /path/to/other/model_dir
```

where that directory contains the same three files.
