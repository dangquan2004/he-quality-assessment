import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ebme398_artifact_detection.labels import Task

try:
    from ebme398_artifact_detection.infer import _find_single_feature_h5, _write_no_tissue_outputs
except Exception as exc:  # pragma: no cover - environment-dependent fallback
    _find_single_feature_h5 = None
    _write_no_tissue_outputs = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@unittest.skipIf(_IMPORT_ERROR is not None, f"inference helpers unavailable: {_IMPORT_ERROR}")
class InferHelpersTests(unittest.TestCase):
    def test_find_single_feature_h5_optional_returns_none_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self.assertIsNone(_find_single_feature_h5(root, "missing-slide", required=False))
            with self.assertRaises(FileNotFoundError):
                _find_single_feature_h5(root, "missing-slide")

    def test_write_no_tissue_outputs_creates_structured_multiclass_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            payload = _write_no_tissue_outputs(
                output_dir=tmpdir,
                slide_id="slide-001",
                task=Task.MULTICLASS,
                slide_threshold=0.5,
                reason="no coords",
            )
            self.assertEqual(payload["status"], "no_tissue_detected")
            self.assertTrue(Path(payload["predictions_csv"]).exists())
            self.assertTrue(Path(payload["slide_summary_json"]).exists())
            self.assertTrue(Path(payload["qc_results_json"]).exists())

            frame = pd.read_csv(payload["predictions_csv"])
            self.assertEqual(
                list(frame.columns),
                [
                    "slide_id",
                    "patch_id",
                    "feature_row_idx",
                    "x",
                    "y",
                    "prob_clean",
                    "prob_tissue_damage",
                    "prob_blurry+fold",
                    "pred_idx",
                    "pred_label",
                ],
            )
            self.assertTrue(frame.empty)

            summary = json.loads(Path(payload["qc_results_json"]).read_text())
            self.assertEqual(summary[0]["slide_pred_label"], "no_tissue_detected")
            self.assertEqual(summary[0]["n_tiles"], 0)
            self.assertEqual(summary[0]["status"], "no_tissue_detected")
            self.assertEqual(summary[0]["reason"], "no coords")
            self.assertIsNone(summary[0]["prob_clean"])
            self.assertIsNone(summary[0]["prob_tissue_damage"])
            self.assertIsNone(summary[0]["prob_blurry+fold"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
