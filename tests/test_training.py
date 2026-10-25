"""Exercise real CPU training, checkpoint state, and the existing sampling CLI."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import pickle
import re
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import sample
import train
from tflib.tf_compat import tf
import utils


ROOT = Path(__file__).resolve().parents[1]
SESSION_CONFIG = tf.ConfigProto(device_count={"GPU": 0})


class TrainingTests(unittest.TestCase):
    def setUp(self):
        temporary_root = ROOT / ".test-output"
        temporary_root.mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=temporary_root)
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)

    def arguments(self, *extra):
        return train.parse_args([
            "-i", str(self.path / "train.txt"),
            "-o", str(self.path / "nested" / "model"),
            "-n", "100", "-s", "100", "-b", "2", "-l", "4",
            "-d", "4", "-c", "1", *extra,
        ])

    def run_cli(self, script, *arguments):
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = "-1"
        environment["TF_CPP_MIN_LOG_LEVEL"] = "2"
        result = subprocess.run(
            [sys.executable, str(ROOT / script), *map(str, arguments)],
            cwd=self.path, env=environment, capture_output=True, text=True,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_real_optimizer_updates_and_complete_checkpoint_restore(self):
        args = self.arguments()
        batch = np.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=np.int32)
        graph = tf.Graph()
        with graph.as_default():
            tf.set_random_seed(2026)
            operations = train.build_training_graph(args, vocab_size=5)
            generator = [variable for variable in tf.trainable_variables()
                         if variable.op.name.startswith("Generator.")]
            discriminator = [variable for variable in tf.trainable_variables()
                             if variable.op.name.startswith("Discriminator.")]
            variables = tf.global_variables()
            initializer = tf.global_variables_initializer()
            saver = tf.train.Saver()
        self.assertTrue(generator)
        self.assertTrue(discriminator)
        self.assertTrue(any("Adam" in variable.op.name for variable in variables))
        feed = {operations["real_inputs_discrete"]: batch}
        checkpoint = str(self.path / "complete.ckpt")
        with tf.Session(graph=graph, config=SESSION_CONFIG) as session:
            session.run(initializer)
            before_generator, before_discriminator = session.run(
                [generator, discriminator]
            )
            session.run(operations["gen_train_op"])
            after_generator, unchanged_discriminator = session.run(
                [generator, discriminator]
            )
            self.assertTrue(any(
                not np.array_equal(before, after)
                for before, after in zip(before_generator, after_generator)
            ), "The generator optimizer must update real model weights")
            for before, after in zip(before_discriminator, unchanged_discriminator):
                np.testing.assert_array_equal(before, after)
            session.run(operations["disc_train_op"], feed_dict=feed)
            unchanged_generator, after_discriminator = session.run(
                [generator, discriminator]
            )
            self.assertTrue(any(
                not np.array_equal(before, after)
                for before, after in zip(before_discriminator, after_discriminator)
            ), "The discriminator optimizer must update real model weights")
            for before, after in zip(after_generator, unchanged_generator):
                np.testing.assert_array_equal(before, after)
            costs = session.run(
                [operations["gen_cost"], operations["disc_cost"]], feed_dict=feed
            )
            self.assertTrue(np.isfinite(costs).all())
            expected = {variable.op.name: value for variable, value in zip(
                variables, session.run(variables)
            )}
            self.assertTrue(all(np.isfinite(value).all() for value in expected.values()))
            saver.save(session, checkpoint)

        restored_graph = tf.Graph()
        with restored_graph.as_default():
            train.build_training_graph(args, vocab_size=5)
            restored_variables = tf.global_variables()
            restored_saver = tf.train.Saver()
            uninitialized = tf.report_uninitialized_variables()
        self.assertEqual({variable.op.name for variable in restored_variables}, set(expected))
        with tf.Session(graph=restored_graph, config=SESSION_CONFIG) as session:
            restored_saver.restore(session, checkpoint)
            self.assertEqual(session.run(uninitialized).tolist(), [])
            for variable, value in zip(restored_variables, session.run(restored_variables)):
                with self.subTest(variable=variable.op.name):
                    np.testing.assert_array_equal(value, expected[variable.op.name])

    def test_training_cli_saves_legacy_artifacts_and_sampling_reads_them(self):
        dataset = self.path / "train.txt"
        # More than ten batches keeps the original held-out JSD split nonempty.
        passwords = ["a", "ab", "abc", "abcd", "bc", "bcd", "cd", "d", "da", "dab"] * 3
        dataset.write_text("\n".join(passwords) + "\n", encoding="ascii")
        output = self.path / "nested" / "model"
        self.run_cli(
            "train.py", "-i", dataset, "-o", output, "-n", 101, "-s", 10,
            "-b", 2, "-l", 4, "-d", 4, "-c", 1,
        )
        charmap, inverse = sample.load_charmaps(output)
        self.assertEqual(set(charmap), {"unk", "`", "a", "b", "c", "d"})
        self.assertEqual({character: index for index, character in enumerate(inverse)}, charmap)
        checkpoint = output / "checkpoints" / "checkpoint_100.ckpt"
        for suffix in (".index", ".meta", ".data-00000-of-00001"):
            self.assertTrue(Path(str(checkpoint) + suffix).is_file(), suffix)
        self.assertEqual(
            sorted(path.name for path in checkpoint.parent.glob("*.index")),
            sorted(f"checkpoint_{iteration}.ckpt.index" for iteration in range(10, 101, 10)),
        )
        # Retain more than five checkpoints and do not add an off-cadence final save.
        self.assertEqual(Path(tf.train.latest_checkpoint(str(checkpoint.parent))), checkpoint)
        self.assertFalse(Path(str(checkpoint).replace('_100.', '_101.') + '.index').exists())
        reader = tf.train.load_checkpoint(str(checkpoint))
        names = reader.get_variable_to_shape_map()
        self.assertTrue(any(name.startswith("Generator.") for name in names))
        self.assertTrue(any(name.startswith("Discriminator.") for name in names))
        self.assertTrue(any("Adam" in name for name in names))
        self.assertTrue(all(np.isfinite(reader.get_tensor(name)).all() for name in names))

        preview = output / "samples" / "samples_100.txt"
        preview_lines = preview.read_text().splitlines()
        self.assertEqual(len(preview_lines), 20)
        tokens = "|".join(re.escape(character) for character in inverse)
        for line in preview_lines:
            self.assertRegex(line, rf"^(?:{tokens}){{4}}$")
        self.assertFalse((preview.parent / "samples_0.txt").exists())
        with (output / "log.pkl").open("rb") as stream:
            log = pickle.load(stream)
        self.assertEqual(set(log), {"time", "train disc cost", "js1", "js2", "js3", "js4"})
        for metric, values in log.items():
            self.assertEqual(set(values), set(range(101)) if metric in ("time", "train disc cost") else {100})
            self.assertTrue(np.isfinite(list(values.values())).all(), metric)
            self.assertTrue((output / (metric.replace(" ", "_") + ".jpg")).is_file())

        generated = self.path / "generated" / "samples.txt"
        self.run_cli(
            "sample.py", "-i", output, "-c", checkpoint, "-o", generated,
            "-n", 5, "-b", 2, "-l", 4, "-d", 4,
        )
        generated_lines = generated.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(generated_lines), 5)
        decoded_tokens = "|".join(re.escape(character) for character in inverse if character != "`")
        for line in generated_lines:
            self.assertRegex(line, rf"^(?:{decoded_tokens}){{0,4}}$")

    def test_dataset_preserves_content_without_a_final_newline(self):
        dataset = self.path / "line-endings.txt"
        dataset.write_bytes(b"a\r\nb \ncd\nabcde\nxy")
        with redirect_stdout(io.StringIO()):
            lines, charmap, inverse = utils.load_dataset(dataset, max_length=4)
        # Keep trailing spaces and padding; exclude overlength samples as before.
        self.assertCountEqual(lines, [tuple("a```"), tuple("b ``"), tuple("cd``"), tuple("xy``")])
        self.assertNotIn("\r", charmap)
        self.assertNotIn("\n", charmap)
        self.assertEqual({character: index for index, character in enumerate(inverse)}, charmap)

    def test_prepared_utf8_data_loads_with_utf8_mode_disabled(self):
        working_bin = self.path / "bin"
        working_data = self.path / "data"
        working_bin.mkdir()
        working_data.mkdir()
        passwords = ["caf\u00e9", "\u00e9t\u00e9", "\u00f8"]
        (working_data / "rockyou-withcount.txt").write_bytes(
            "".join(f"10 {password}\n" for password in passwords).encode("iso-8859-1")
        )
        environment = os.environ.copy()
        environment.update({
            "PYTHONUTF8": "0", "PYTHONCOERCECLOCALE": "0",
            "LC_ALL": "C", "PYTHONPATH": str(ROOT),
        })
        for script in ("make-rockyou-full.py", "prep-data.py"):
            result = subprocess.run(
                [sys.executable, "-X", "utf8=0", str(ROOT / "bin" / script)],
                cwd=working_bin, env=environment, capture_output=True,
                text=True, encoding="ascii", errors="backslashreplace", timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

        code = "\n".join([
            "from contextlib import redirect_stdout",
            "import io, sys, utils",
            "assert sys.flags.utf8_mode == 0",
            "with redirect_stdout(io.StringIO()):",
            "    lines, charmap, inverse = utils.load_dataset(sys.argv[1], max_length=4)",
            "words = [''.join(line).rstrip('`') for line in lines]",
            "assert len(words) == 24, len(words)",
            f"assert set(words) == set({json.dumps(passwords)}), ascii(words)",
            "print('utf8-data-ok')",
        ])
        result = subprocess.run(
            [sys.executable, "-X", "utf8=0", "-c", code, str(working_data / "train.txt")],
            cwd=working_bin, env=environment, capture_output=True,
            text=True, encoding="ascii", errors="backslashreplace", timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("utf8-data-ok", result.stdout)

    def test_invalid_arguments_and_too_small_dataset_fail_without_hanging(self):
        for flag in ("-s", "-b", "-l", "-d", "-c"):
            with self.subTest(flag=flag), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    self.arguments(flag, "0")
                self.assertEqual(raised.exception.code, 2)
        (self.path / "train.txt").write_text("a\n", encoding="ascii")
        stderr = io.StringIO()
        with redirect_stdout(io.StringIO()), redirect_stderr(stderr):
            status = train.main([
                "-i", str(self.path / "train.txt"), "-o", str(self.path / "small"),
                "-n", "1", "-b", "2", "-l", "4", "-d", "4", "-c", "1",
            ])
        self.assertEqual(status, 2, stderr.getvalue())
        self.assertIn("error", stderr.getvalue().lower())

    def test_import_does_not_parse_arguments_build_graph_or_write_files(self):
        code = (
            "import sys; sys.argv = ['train.py', '--invalid-import-argument']; "
            "import train; from tflib.tf_compat import tf; "
            "assert not tf.get_default_graph().get_operations(); print('import-ok')"
        )
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(ROOT)
        environment["TF_CPP_MIN_LOG_LEVEL"] = "2"
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=self.path,
            env=environment, capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("import-ok", result.stdout)
        self.assertEqual(list(self.path.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
