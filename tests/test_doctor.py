import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from ebme398_artifact_detection import doctor


class DoctorTests(unittest.TestCase):
    def test_hugging_face_auth_timeout_returns_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            token_path = Path(tmpdir) / "token"
            token_path.write_text("fake-token")

            def fake_run_command(cmd, *, cwd=None, timeout=30):
                if cmd[:2] == [doctor.sys.executable, "-c"]:
                    self.assertEqual(timeout, 20)
                    return False, "Command timed out after 20 seconds"
                return True, "test-user"

            with mock.patch.dict(os.environ, {}, clear=True):
                with mock.patch.object(doctor, "_default_hf_token_path", return_value=token_path):
                    with mock.patch.dict("sys.modules", {"huggingface_hub": types.SimpleNamespace(HfApi=object)}):
                        with mock.patch.object(doctor, "_run_command", side_effect=fake_run_command):
                            result = doctor.check_hugging_face_auth()

        self.assertFalse(result.ok)
        self.assertIn("could not verify access to MahmoodLab/UNI2-h", result.summary)
        self.assertIn("timed out", result.summary)
        self.assertIsNotNone(result.fix)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
