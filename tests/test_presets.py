import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

from ebme398_artifact_detection.labels import Task
from ebme398_artifact_detection.presets import (
    get_hybrid_inference_preset,
    resolve_model_dir,
    resolve_preset_artifact_path,
)


@contextmanager
def chdir(path: Path):
    old = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(old)


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

    def test_resolve_model_dir_prefers_cwd_clone_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "models" / "qc"
            model_dir.mkdir(parents=True)
            with chdir(root):
                self.assertEqual(resolve_model_dir(), model_dir.resolve())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
