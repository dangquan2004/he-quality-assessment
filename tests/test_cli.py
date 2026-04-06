import unittest

from ebme398_artifact_detection.cli import build_parser


class CLITests(unittest.TestCase):
    def test_help_lists_recovered_commands(self) -> None:
        help_text = build_parser().format_help()
        self.assertIn("train-embedding", help_text)
        self.assertIn("train-resnet", help_text)
        self.assertIn("run-trident", help_text)
        self.assertIn("infer-hybrid-wsi", help_text)

    def test_infer_hybrid_wsi_accepts_device_and_threshold(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "infer-hybrid-wsi",
                "--input-wsi",
                "slide.ome.tiff",
                "--output-dir",
                "out",
                "--trident-dir",
                "external/TRIDENT",
                "--checkpoint-path",
                "model.pt",
                "--scaler-path",
                "scaler.joblib",
                "--selection-json",
                "selection.json",
                "--task",
                "binary",
                "--patch-encoder",
                "uni_v2",
                "--device",
                "cpu",
                "--slide-threshold",
                "0.7",
            ]
        )
        self.assertEqual(args.device, "cpu")
        self.assertAlmostEqual(args.slide_threshold, 0.7)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
