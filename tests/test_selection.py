import json
import tempfile
import unittest
from pathlib import Path

from ebme398_artifact_detection.selection import (
    load_selection_payload,
    selection_embedding_keep,
    selection_feature_key,
    selection_hc_keep,
)


class SelectionTests(unittest.TestCase):
    def test_load_selection_payload_supports_legacy_uni_keep_idx(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "legacy_selection.json"
            path.write_text(
                json.dumps(
                    {
                        "hc_cols_all": ["a", "b"],
                        "hc_keep_idx": [0],
                        "uni_keep_idx": [1, 3, 5],
                    }
                )
            )
            payload = load_selection_payload(path)
            self.assertEqual(payload["embedding_keep_idx"], [1, 3, 5])
            self.assertEqual(selection_hc_keep(payload).tolist(), [0])
            self.assertEqual(selection_embedding_keep(payload).tolist(), [1, 3, 5])
            self.assertEqual(selection_feature_key(payload), "features")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
