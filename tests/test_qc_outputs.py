import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ebme398_artifact_detection.qc_outputs import load_single_slide_qc_row, write_batch_results_csv


class QCOutputsTests(unittest.TestCase):
    def test_load_single_slide_qc_row_requires_single_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            payload_path = Path(tmpdir) / "quality_control_results.json"
            payload_path.write_text(json.dumps([{"slide_id": "SR001", "slide_pred_label": "clean"}]))

            row = load_single_slide_qc_row(payload_path)

            self.assertEqual(row["slide_id"], "SR001")
            self.assertEqual(row["slide_pred_label"], "clean")

    def test_write_batch_results_csv_flattens_and_sorts_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first_json = root / "b.json"
            second_json = root / "a.json"
            first_json.write_text(json.dumps([{"slide_id": "SR002", "slide_pred_label": "clean", "n_tiles": 2}]))
            second_json.write_text(json.dumps([{"slide_id": "SR001", "slide_pred_label": "tissue_damage", "n_tiles": 3}]))

            output_csv = root / "batch_results.csv"
            write_batch_results_csv(
                [
                    {
                        "input_wsi": "/tmp/SR002.ome.tiff",
                        "output_dir": "/tmp/out/SR002",
                        "predictions_csv": "/tmp/out/SR002/hybrid_tile_predictions.csv",
                        "qc_results_json": str(first_json),
                    },
                    {
                        "input_wsi": "/tmp/SR001.ome.tiff",
                        "output_dir": "/tmp/out/SR001",
                        "predictions_csv": "/tmp/out/SR001/hybrid_tile_predictions.csv",
                        "qc_results_json": str(second_json),
                    },
                ],
                output_csv,
            )

            frame = pd.read_csv(output_csv)

            self.assertEqual(frame["slide_id"].tolist(), ["SR001", "SR002"])
            self.assertIn("input_wsi", frame.columns)
            self.assertIn("predictions_csv", frame.columns)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
