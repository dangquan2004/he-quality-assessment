import unittest

from ebme398_artifact_detection.cli import build_parser


class CLITests(unittest.TestCase):
    def test_help_lists_recovered_commands(self) -> None:
        help_text = build_parser().format_help()
        self.assertIn("train-embedding", help_text)
        self.assertIn("train-resnet", help_text)
        self.assertIn("run-trident", help_text)
        self.assertIn("infer-hybrid-wsi", help_text)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
