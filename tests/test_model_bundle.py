import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from ebme398_artifact_detection.model_bundle import file_sha256, resolve_model_bundle


class ModelBundleTests(unittest.TestCase):
    def test_file_sha256_matches_known_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            path.write_bytes(b"abc123")
            self.assertEqual(file_sha256(path), hashlib.sha256(b"abc123").hexdigest())

    def test_resolve_model_bundle_uses_manifest_and_verifies_checksums(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint = root / "weights.pt"
            scaler = root / "scaler.joblib"
            selection = root / "selection.json"
            checkpoint.write_bytes(b"checkpoint")
            scaler.write_bytes(b"scaler")
            selection.write_text('{"hc_cols_all": [], "hc_keep_idx": [], "embedding_keep_idx": []}')
            manifest = {
                "task": "multiclass",
                "patch_encoder": "uni_v2",
                "model_kind": "mlp",
                "hidden_dim": 256,
                "preprocessing": {
                    "mpp": 0.5,
                    "mag": 20,
                    "patch_size": 256,
                    "patch_size_level0": 2048,
                    "target_patch_size": 256,
                    "quality": 85,
                    "slide_threshold": 0.4,
                },
                "files": {
                    "checkpoint": {"path": checkpoint.name, "sha256": file_sha256(checkpoint)},
                    "scaler": {"path": scaler.name, "sha256": file_sha256(scaler)},
                    "selection": {"path": selection.name, "sha256": file_sha256(selection)},
                },
            }
            (root / "model_manifest.json").write_text(json.dumps(manifest))

            bundle = resolve_model_bundle(model_dir=root)

            self.assertEqual(bundle["checkpoint_path"], checkpoint.resolve())
            self.assertEqual(bundle["scaler_path"], scaler.resolve())
            self.assertEqual(bundle["selection_json"], selection.resolve())
            self.assertEqual(bundle["patch_encoder"], "uni_v2")
            self.assertEqual(bundle["model_kind"], "mlp")
            self.assertEqual(bundle["hidden_dim"], 256)
            self.assertEqual(bundle["task"].value, "multiclass")
            self.assertEqual(bundle["mpp"], 0.5)
            self.assertEqual(bundle["mag"], 20)
            self.assertEqual(bundle["patch_size"], 256)
            self.assertEqual(bundle["patch_size_level0"], 2048)
            self.assertEqual(bundle["target_patch_size"], 256)
            self.assertEqual(bundle["quality"], 85)
            self.assertEqual(bundle["slide_threshold"], 0.4)

    def test_resolve_model_bundle_raises_on_checksum_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint = root / "checkpoint.pt"
            scaler = root / "scaler.joblib"
            selection = root / "selection.json"
            checkpoint.write_bytes(b"checkpoint")
            scaler.write_bytes(b"scaler")
            selection.write_text("{}")
            manifest = {
                "files": {
                    "checkpoint": {"path": checkpoint.name, "sha256": "0" * 64},
                    "scaler": {"path": scaler.name, "sha256": file_sha256(scaler)},
                    "selection": {"path": selection.name, "sha256": file_sha256(selection)},
                }
            }
            (root / "model_manifest.json").write_text(json.dumps(manifest))

            with self.assertRaises(RuntimeError):
                resolve_model_bundle(model_dir=root)

    def test_resolve_model_bundle_falls_back_to_preset_preprocessing_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint = root / "checkpoint.pt"
            scaler = root / "scaler.joblib"
            selection = root / "selection.json"
            checkpoint.write_bytes(b"checkpoint")
            scaler.write_bytes(b"scaler")
            selection.write_text("{}")
            manifest = {
                "files": {
                    "checkpoint": {"path": checkpoint.name, "sha256": file_sha256(checkpoint)},
                    "scaler": {"path": scaler.name, "sha256": file_sha256(scaler)},
                    "selection": {"path": selection.name, "sha256": file_sha256(selection)},
                }
            }
            (root / "model_manifest.json").write_text(json.dumps(manifest))

            bundle = resolve_model_bundle(model_dir=root)

            self.assertEqual(bundle["mpp"], 0.25)
            self.assertEqual(bundle["mag"], 10)
            self.assertEqual(bundle["patch_size"], 512)
            self.assertEqual(bundle["patch_size_level0"], 3072)
            self.assertEqual(bundle["target_patch_size"], 512)
            self.assertEqual(bundle["quality"], 90)
            self.assertEqual(bundle["slide_threshold"], 0.5)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
