import tempfile
import unittest
from pathlib import Path

from ebme398_artifact_detection.labels import Task
from ebme398_artifact_detection.presets import get_hybrid_inference_preset, resolve_preset_artifact_path


class PresetTests(unittest.TestCase):
    def test_s4_new_preset_metadata(self) -> None:
        preset = get_hybrid_inference_preset("s4_new_multiclass")
        self.assertEqual(preset.task, Task.MULTICLASS)
        self.assertEqual(preset.patch_encoder, "uni_v2")
        self.assertEqual(preset.model_kind, "mlp")

    def test_resolve_preset_artifact_path_from_custom_root(self) -> None:
        preset = get_hybrid_inference_preset("s4_new_multiclass")
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target = root / preset.selection_relpath
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("{}")
            resolved = resolve_preset_artifact_path(preset.selection_relpath, root)
            self.assertEqual(resolved, target.resolve())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
