"""Check the data preparation workflow using small, isolated artificial data."""

from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
DATASET_URL = (
    "https://raw.githubusercontent.com/danielmiessler/SecLists/"
    "5e1dc9cc79aac54b373349e2a97bbb22f1b63bb3/"
    "Passwords/Leaked-Databases/rockyou-withcount.txt.tar.gz"
)


class DataPreparationTests(unittest.TestCase):
    def setUp(self):
        temporary_root = ROOT / ".test-output"
        temporary_root.mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=temporary_root)
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        self.bin = self.path / "bin"
        self.data = self.path / "data"
        self.bin.mkdir()
        self.data.mkdir()
        for name in ("make-rockyou-full.py", "prep-data.py", "get-dataset.sh"):
            shutil.copyfile(ROOT / "bin" / name, self.bin / name)

    def run_python(self, name, environment=None):
        variables = os.environ.copy()
        if environment:
            variables.update(environment)
        result = subprocess.run(
            [sys.executable, str(self.bin / name)], cwd=self.bin,
            env=variables,
            capture_output=True, text=True, encoding="utf-8", errors="backslashreplace",
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_count_expansion_preserves_existing_ascii_rules_and_overwrites(self):
        source = self.data / "rockyou-withcount.txt"
        output = self.data / "rockyou-full.txt"
        source.write_bytes(
            b"  2 alpha\r\n1 pa ss \n0 zero\n-1 negative\nbad invalid\n"
            b"1\n\n1  \n1\ttab-separated\n1 toolongpassword\n1 final"
        )
        output.write_text("old-output\n", encoding="ascii")
        self.run_python("make-rockyou-full.py")
        self.assertEqual(
            output.read_text(encoding="ascii").splitlines(),
            ["alpha", "alpha", "pass", "toolongpassword", "final"],
        )
        # A rerun replaces the destination rather than accumulating old samples.
        source.write_bytes(b"1 replacement\n")
        self.run_python("make-rockyou-full.py")
        self.assertEqual(output.read_text(encoding="ascii"), "replacement\n")

    def test_split_is_reproducible_and_preserves_every_complete_input_line(self):
        records = [
            "alpha\n", "alpha\n", "a b\n", "  lead\n", "trail  \n",
            "\n", "toolongpassword\n", "p7\n", "p8\n", "p9\n",
        ]
        source = self.data / "rockyou-full.txt"
        original = "".join(records).encode("ascii")
        source.write_bytes(original)
        destinations = [self.data / "train.txt", self.data / "test.txt"]
        for destination in destinations:
            destination.write_text("old-output\n", encoding="ascii")
        self.run_python("prep-data.py")
        first = [destination.read_bytes() for destination in destinations]
        training, testing = [
            destination.read_text(encoding="ascii").splitlines(keepends=True)
            for destination in destinations
        ]
        self.assertEqual((len(training), len(testing)), (8, 2))
        self.assertEqual(Counter(training + testing), Counter(records))
        self.assertEqual(source.read_bytes(), original)
        self.run_python("prep-data.py")
        self.assertEqual([destination.read_bytes() for destination in destinations], first)

    def test_latin1_source_produces_utf8_files_even_without_python_utf8_mode(self):
        source = self.data / "rockyou-withcount.txt"
        original = b"4 caf\xe9\n4 ni\xf1o\n2 plain\n"
        source.write_bytes(original)
        environment = {"PYTHONUTF8": "0"}
        self.run_python("make-rockyou-full.py", environment)
        full = self.data / "rockyou-full.txt"
        expected = Counter({"caf\u00e9": 4, "ni\u00f1o": 4, "plain": 2})
        full_bytes = full.read_bytes()
        self.assertEqual(Counter(full_bytes.decode("utf-8").splitlines()), expected)
        self.run_python("prep-data.py", environment)
        training, testing = [
            (self.data / name).read_bytes().decode("utf-8").splitlines()
            for name in ("train.txt", "test.txt")
        ]
        self.assertEqual((len(training), len(testing)), (8, 2))
        self.assertEqual(Counter(training + testing), expected)
        self.assertEqual(full.read_bytes(), full_bytes)
        self.assertEqual(source.read_bytes(), original)

    def test_split_does_not_join_an_unterminated_final_line_to_another_sample(self):
        records = [f"p{index}" for index in range(8)]
        source = self.data / "rockyou-full.txt"
        original = "\n".join(records).encode("ascii")
        source.write_bytes(original)
        self.run_python("prep-data.py")
        training, testing = [
            (self.data / name).read_text(encoding="utf-8").splitlines()
            for name in ("train.txt", "test.txt")
        ]
        self.assertEqual((len(training), len(testing)), (6, 2))
        self.assertEqual(Counter(training + testing), Counter(records))
        self.assertEqual(source.read_bytes(), original)

    def shell_environment(self, mode="success"):
        stubs = self.path / "stubs"
        stubs.mkdir(exist_ok=True)
        (self.path / "commands.jsonl").write_text("", encoding="utf-8")
        helper = stubs / "simulate_tools.py"
        helper.write_text("""
import io
import json
import os
from pathlib import Path
import sys
import tarfile

root = Path(os.environ['PASSGAN_TEST_ROOT']).resolve()
command, arguments = sys.argv[1], sys.argv[2:]
mode = os.environ['PASSGAN_TEST_MODE']
with (root / 'commands.jsonl').open('a', encoding='utf-8') as stream:
    stream.write(json.dumps([command, arguments]) + '\\n')
archive = root / 'data' / 'rockyou-withcount.txt.tar.gz'
if command == 'wget':
    assert arguments == [os.environ['PASSGAN_TEST_URL'], '-P', '../data']
    if mode == 'wget-fail':
        print('intentional wget failure', file=sys.stderr)
        sys.exit(8)
    if mode == 'tar-fail':
        archive.write_bytes(b'not a gzip archive')
    else:
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode='w:gz') as bundle:
            for name, content in (
                ('rockyou-withcount.txt', b'2 alpha\\n1 bravo\\n'),
                ('unrelated.txt', b'do not extract this file'),
            ):
                member = tarfile.TarInfo(name)
                member.size = len(content)
                bundle.addfile(member, io.BytesIO(content))
        payload = buffer.getvalue()
        if mode == 'truncated-archive':
            payload = payload[:-8]
        archive.write_bytes(payload)
else:
    raise AssertionError(command)
""", encoding="utf-8")
        for command in ("wget",):
            executable = stubs / command
            executable.write_text(
                '#!/bin/sh\nexec "$PASSGAN_TEST_PYTHON" "$PASSGAN_TEST_HELPER" '
                f'{command} "$@"\n', encoding="utf-8",
            )
            executable.chmod(0o755)
        environment = os.environ.copy()
        environment.update({
            # Stub wget; use the system tar/gzip to exercise real extraction.
            "PATH": str(stubs) + os.pathsep + os.defpath,
            "PASSGAN_TEST_ROOT": str(self.path),
            "PASSGAN_TEST_PYTHON": sys.executable,
            "PASSGAN_TEST_HELPER": str(helper), "PASSGAN_TEST_URL": DATASET_URL,
            "PASSGAN_TEST_MODE": mode,
        })
        return environment

    def run_shell(self, environment):
        return subprocess.run(
            ["/bin/sh", str(self.bin / "get-dataset.sh")], cwd=self.bin,
            env=environment, capture_output=True, text=True, encoding="utf-8",
            errors="backslashreplace", timeout=30,
        )

    @unittest.skipUnless(sys.platform.startswith("linux"), "Requires the Linux POSIX shell")
    def test_download_and_extraction_follow_existing_paths_without_network(self):
        result = self.run_shell(self.shell_environment())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("...Done", result.stdout)
        self.assertEqual(
            (self.data / "rockyou-withcount.txt").read_bytes(), b"2 alpha\n1 bravo\n",
        )
        self.assertFalse((self.data / "rockyou-withcount.txt.tar.gz").exists())
        self.assertFalse((self.data / "unrelated.txt").exists())
        commands = [json.loads(line) for line in (self.path / "commands.jsonl").read_text().splitlines()]
        self.assertEqual(commands, [
            ["wget", [DATASET_URL, "-P", "../data"]],
        ])

    @unittest.skipUnless(sys.platform.startswith("linux"), "Requires the Linux POSIX shell")
    def test_download_or_extraction_failure_stops_without_reporting_done(self):
        for mode in ("wget-fail", "tar-fail", "truncated-archive"):
            with self.subTest(mode=mode):
                result = self.run_shell(self.shell_environment(mode))
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertNotIn("...Done", result.stdout)
                self.assertFalse((self.data / "rockyou-withcount.txt").exists())
                commands = [
                    json.loads(line)[0]
                    for line in (self.path / "commands.jsonl").read_text().splitlines()
                ]
                self.assertEqual(commands, ["wget"])
                self.assertEqual(
                    (self.data / "rockyou-withcount.txt.tar.gz").exists(), mode != "wget-fail",
                )

    @unittest.skipUnless(sys.platform.startswith("linux"), "Requires the Linux POSIX shell")
    def test_extraction_does_not_overwrite_an_existing_file_or_symbolic_link(self):
        output = self.data / "rockyou-withcount.txt"
        for kind in ("file", "symbolic-link"):
            with self.subTest(kind=kind):
                if kind == "file":
                    output.write_bytes(b"keep this existing file")
                else:
                    output.symlink_to("missing.txt")
                result = self.run_shell(self.shell_environment())
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertNotIn("...Done", result.stdout)
                if kind == "file":
                    self.assertEqual(output.read_bytes(), b"keep this existing file")
                else:
                    self.assertTrue(output.is_symlink())
                    self.assertEqual(os.readlink(output), "missing.txt")
                    self.assertFalse((self.data / "missing.txt").exists())
                self.assertTrue((self.data / "rockyou-withcount.txt.tar.gz").exists())
                output.unlink()


if __name__ == "__main__":
    unittest.main()
