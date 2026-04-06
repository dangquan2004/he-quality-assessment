import unittest

import numpy as np
import pandas as pd

from ebme398_artifact_detection.alignment import (
    align_handcrafted_rows_to_feature_rows,
    ensure_slide_id_column,
)


class AlignmentTests(unittest.TestCase):
    def test_ensure_slide_id_column_uses_patch_path(self) -> None:
        df = pd.DataFrame({"path": ["/tmp/SR040.ome.pyr_00000003.pt"]})
        out = ensure_slide_id_column(df)
        self.assertEqual(out["slide_id"].tolist(), ["SR040.ome.pyr"])

    def test_coordinate_alignment_sorts_to_feature_rows(self) -> None:
        df = pd.DataFrame(
            {
                "path": ["/tmp/SR040.ome.pyr_00000001.pt", "/tmp/SR040.ome.pyr_00000000.pt"],
                "x": [30, 10],
                "y0": [40, 20],
            }
        )
        coords = np.asarray([[10, 20], [30, 40]], dtype=int)
        aligned, feature_row_idx, mode = align_handcrafted_rows_to_feature_rows(
            df,
            coords=coords,
            n_features=2,
            context="unit-test",
        )
        self.assertEqual(mode, "coords")
        self.assertEqual(feature_row_idx.tolist(), [0, 1])
        self.assertEqual(aligned["feature_row_idx"].tolist(), [0, 1])

    def test_coordinate_and_patch_id_mismatch_raises(self) -> None:
        df = pd.DataFrame(
            {
                "path": ["/tmp/SR040.ome.pyr_00000000.pt", "/tmp/SR040.ome.pyr_00000001.pt"],
                "x": [30, 10],
                "y0": [40, 20],
            }
        )
        coords = np.asarray([[10, 20], [30, 40]], dtype=int)
        with self.assertRaises(RuntimeError):
            align_handcrafted_rows_to_feature_rows(
                df,
                coords=coords,
                n_features=2,
                context="unit-test",
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
