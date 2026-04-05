import unittest

from ebme398_artifact_detection.labels import Task, normalize_label, to_task_label


class LabelTests(unittest.TestCase):
    def test_binary_labels(self) -> None:
        self.assertEqual(to_task_label("clean", Task.BINARY), 0)
        self.assertEqual(to_task_label("blurry+fold", Task.BINARY), 1)
        self.assertEqual(to_task_label("tissue_damge", Task.BINARY), 1)

    def test_multiclass_labels(self) -> None:
        self.assertEqual(to_task_label("clean", Task.MULTICLASS), 0)
        self.assertEqual(to_task_label("tissue damage", Task.MULTICLASS), 1)
        self.assertEqual(to_task_label("fold+blur", Task.MULTICLASS), 2)

    def test_normalization(self) -> None:
        self.assertEqual(normalize_label(" Tissue-Damge "), "tissue_damage")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
