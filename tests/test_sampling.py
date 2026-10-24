"""Exercise the public sampling CLI and its file handling contracts."""

from contextlib import redirect_stderr, redirect_stdout
import io
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import tempfile
import unittest

import sample


ROOT = Path(__file__).resolve().parents[1]
PRETRAINED = ROOT / "pretrained"
CHECKPOINT = PRETRAINED / "checkpoints" / "checkpoint_200000.ckpt"


class SamplingTests(unittest.TestCase):
    def setUp(self):
        # Keep all temporary files inside the checkout, including in sandboxes.
        temporary_root = ROOT / ".test-output"
        temporary_root.mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=temporary_root)
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name) / "nested" / "samples.txt"

    def arguments(self, *extra):
        return [
            "-i", str(PRETRAINED), "-c", str(CHECKPOINT),
            "-o", str(self.output), "-b", "4", *extra,
        ]

    def invoke(self, *extra):
        stderr = io.StringIO()
        with redirect_stdout(io.StringIO()), redirect_stderr(stderr):
            status = sample.main(self.arguments(*extra))
        self.assertEqual(status, 0, stderr.getvalue())

    def assert_cli_error(self, arguments, expected_text):
        stderr = io.StringIO()
        with redirect_stdout(io.StringIO()), redirect_stderr(stderr):
            try:
                status = sample.main(arguments)
            except SystemExit as raised:
                status = raised.code
        self.assertIsInstance(status, int)
        self.assertNotEqual(status, 0)
        self.assertIn(expected_text.lower(), stderr.getvalue().lower())

    def test_exact_counts_including_partial_batches_and_zero(self):
        for count in (1, 3, 5, 0):
            with self.subTest(count=count):
                self.invoke("-n", str(count))
                self.assertTrue(self.output.is_file())
                self.assertEqual(
                    len(self.output.read_text(encoding="utf-8").splitlines()), count
                )

    def test_overwrite_and_explicit_append(self):
        self.output.parent.mkdir(parents=True)
        self.output.write_text("既存の行\n", encoding="utf-8")
        self.invoke("-n", "1", "--append")
        appended = self.output.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(appended), 2)
        self.assertEqual(appended[0], "既存の行")
        self.invoke("-n", "1")
        overwritten = self.output.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(overwritten), 1)
        self.assertNotIn("既存の行", overwritten)

    def test_invalid_numeric_arguments_are_rejected(self):
        for flag, value in (("-n", "-1"), ("-b", "0"), ("-l", "0"), ("-d", "-1")):
            with self.subTest(flag=flag, value=value):
                self.assert_cli_error(self.arguments(flag, value), "error")
        self.assertFalse(self.output.exists())

    def test_missing_checkpoint_and_wrong_model_shape_are_rejected(self):
        self.assert_cli_error(
            self.arguments("-n", "1", "-c", str(Path(self.directory.name) / "missing")),
            "checkpoint",
        )
        self.assert_cli_error(self.arguments("-n", "1", "-d", "8"), "shape")
        self.assertFalse(self.output.exists())

    def test_checkpoint_does_not_require_saved_meta_graph(self):
        prefix = Path(self.directory.name) / "checkpoint-without-meta.ckpt"
        for suffix in (".index", ".data-00000-of-00001"):
            shutil.copyfile(str(CHECKPOINT) + suffix, str(prefix) + suffix)
        self.assertFalse(Path(str(prefix) + ".meta").exists())
        self.invoke("-n", "1", "-c", str(prefix))
        self.assertEqual(len(self.output.read_text(encoding="utf-8").splitlines()), 1)

    def test_inconsistent_character_dictionary_is_rejected(self):
        bad_model = Path(self.directory.name) / "bad-model"
        bad_model.mkdir()
        for name in ("charmap.pickle", "charmap_inv.pickle"):
            (bad_model / name).write_bytes((PRETRAINED / name).read_bytes())
        charmap = pickle.loads((bad_model / "charmap.pickle").read_bytes(), encoding="latin1")
        charmap["a"] = charmap["e"]
        (bad_model / "charmap.pickle").write_bytes(pickle.dumps(charmap))
        self.assert_cli_error(
            self.arguments("-n", "1", "-i", str(bad_model)), "character map"
        )
        self.assertFalse(self.output.exists())

    def test_import_does_not_parse_arguments_build_graph_or_write_files(self):
        code = (
            "import sys; sys.argv = ['sample.py', '--invalid-import-argument']; "
            "import sample; from tflib.tf_compat import tf; "
            "assert not tf.get_default_graph().get_operations(); "
            "print('import-ok')"
        )
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(ROOT)
        environment["TF_CPP_MIN_LOG_LEVEL"] = "2"
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=self.directory.name,
            env=environment, capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("import-ok", result.stdout)
        self.assertEqual(list(Path(self.directory.name).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
