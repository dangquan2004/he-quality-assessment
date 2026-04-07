import unittest
import subprocess
import sys
from pathlib import Path

from ebme398_artifact_detection.cli import build_parser


class CLITests(unittest.TestCase):
    def test_help_lists_recovered_commands(self) -> None:
        help_text = build_parser().format_help()
        self.assertIn("train-embedding", help_text)
        self.assertIn("train-resnet", help_text)
        self.assertIn("run-trident", help_text)
        self.assertIn("infer-hybrid-wsi", help_text)
        self.assertIn("run-qc", help_text)

    def test_infer_hybrid_wsi_accepts_device_and_threshold(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "infer-hybrid-wsi",
                "--input-path",
                "slide.ome.tiff",
                "--output-dir",
                "out",
                "--trident-dir",
                "external/TRIDENT",
                "--preset",
                "s4_new_multiclass",
                "--device",
                "cpu",
                "--slide-threshold",
                "0.7",
            ]
        )
        self.assertEqual(args.input_path, "slide.ome.tiff")
        self.assertEqual(args.preset, "s4_new_multiclass")
        self.assertEqual(args.device, "cpu")
        self.assertAlmostEqual(args.slide_threshold, 0.7)

    def test_run_qc_defaults_to_external_trident(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "run-qc",
                "--input-path",
                "slides",
                "--output-dir",
                "out",
            ]
        )
        self.assertEqual(args.trident_dir, "external/TRIDENT")
        self.assertEqual(args.device, "auto")

    def test_repo_script_help_works_from_clone_layout(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "scripts" / "he_quality.py"
        result = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("infer-hybrid-wsi", result.stdout)
        self.assertIn("run-qc", result.stdout)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
