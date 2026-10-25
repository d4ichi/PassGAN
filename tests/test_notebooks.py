"""Execute the debugging notebooks in real, isolated CPU Jupyter kernels."""

import ast
from contextlib import contextmanager
from importlib.util import find_spec
import json
import os
from pathlib import Path
import pickle
import re
import subprocess
import sys
import tempfile
import unittest

try:
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    if find_spec("ipykernel") is None:
        raise ImportError("ipykernel is required to execute notebook tests")
except ImportError:
    if os.environ.get("PASSGAN_REQUIRE_NOTEBOOK_TESTS") == "1":
        raise
    nbformat = None


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipIf(nbformat is None, "Install requirements-notebooks.txt to execute notebook tests")
class NotebookTests(unittest.TestCase):
    def setUp(self):
        temporary_root = ROOT / ".test-output"
        temporary_root.mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=temporary_root)
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        self.environment = os.environ.copy()
        self.environment.update({
            "CUDA_VISIBLE_DEVICES": "-1",
            "TF_CPP_MIN_LOG_LEVEL": "2",
            "MPLCONFIGDIR": str(self.path / "matplotlib"),
            "IPYTHONDIR": str(self.path / "ipython"),
            "JUPYTER_RUNTIME_DIR": str(self.path / "runtime"),
        })

    def notebook(self, filename, **overrides):
        notebook = nbformat.read(ROOT / filename, as_version=4)
        self.assertEqual(notebook.metadata.kernelspec.name, "python3")
        self.assertEqual(notebook.metadata.language_info.version, "3.12.14")
        parameters = None
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            self.assertIsNone(cell.execution_count)
            self.assertEqual(cell.outputs, [])
            if "class VirtualArgparse:" in cell.source:
                parameters = cell
        self.assertIsNotNone(parameters)
        # These defaults are part of the existing interactive notebook workflow.
        expected = {
            "notebook-sample.ipynb": {
                "input_dir": "pretrained",
                "checkpoint": "pretrained/checkpoints/checkpoint_200000.ckpt",
                "output": "generated_pass.txt", "batch_size": 1024,
                "num_samples": 1000000, "seq_length": 10, "layer_dim": 128,
            },
            "notebook-train.ipynb": {
                "training_data": "data/train.txt", "output_dir": "pretrained",
                "save_every": 10, "iters": 100, "batch_size": 64,
                "seq_length": 10, "layer_dim": 128, "critic_iters": 10,
                "lamb": 10,
            },
        }[filename]
        definition = next(node for node in ast.parse(parameters.source).body
                          if isinstance(node, ast.ClassDef))
        defaults = {node.targets[0].id: ast.literal_eval(node.value)
                    for node in definition.body if isinstance(node, ast.Assign)}
        self.assertEqual(defaults, expected)
        parameters.source += "\n" + "\n".join(
            f"VirtualArgparse.{name} = {value!r}" for name, value in overrides.items()
        )
        return notebook

    @contextmanager
    def kernel(self):
        specification = self.path / "kernels" / "passgan-tests"
        specification.mkdir(parents=True)
        (specification / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "PassGAN notebook tests", "language": "python",
        }), encoding="utf-8")
        manager = KernelManager(
            kernel_name="passgan-tests",
            kernel_spec_manager=KernelSpecManager(
                kernel_dirs=[str(specification.parent)], ensure_native_kernel=False,
            ),
            connection_file=str(self.path / "connection.json"),
        )
        client = NotebookClient(
            nbformat.v4.new_notebook(), km=manager, timeout=180,
            allow_errors=False, record_timing=False,
            resources={"metadata": {"path": str(ROOT)}},
        )
        try:
            with client.setup_kernel(cleanup_kc=True, env=self.environment):
                yield client
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
            manager.cleanup_resources()

    def execute_notebook(self, client, notebook):
        for cell in notebook.cells:
            index = len(client.nb.cells)
            client.nb.cells.append(cell)
            client.execute_cell(cell, index)

    def execute_code(self, client, source):
        cell = nbformat.v4.new_code_cell(source)
        index = len(client.nb.cells)
        client.nb.cells.append(cell)
        client.execute_cell(cell, index)

    def assert_sampling_graph(self, client):
        self.execute_code(client, """
assert len(tf.global_variables()) == 24
assert all(variable.op.name.startswith('Generator.') for variable in tf.global_variables())
assert not any('Adam' in variable.op.name for variable in tf.global_variables())
""")

    def test_pretrained_sampling_preserves_append_and_complete_batch_behavior(self):
        output = self.path / "generated.txt"
        output.write_text("existing-line\n", encoding="ascii")
        with self.kernel() as client:
            for batch_size, num_samples, expected_count in ((2, 5, 4), (1, 1, 5), (2, 1, 5), (2, 0, 5)):
                with self.subTest(batch_size=batch_size, num_samples=num_samples):
                    notebook = self.notebook(
                        "notebook-sample.ipynb", output=str(output),
                        batch_size=batch_size, num_samples=num_samples,
                    )
                    self.execute_notebook(client, notebook)
                    self.assert_sampling_graph(client)
                    lines = output.read_text().splitlines()
                    self.assertEqual(lines[0], "existing-line")
                    self.assertEqual(len(lines), expected_count + 1)
                    self.assertTrue(all("`" not in line for line in lines[1:]))

    def test_training_checkpoint_works_in_cli_and_sampling_notebook(self):
        dataset = self.path / "train.txt"
        passwords = ["a", "ab", "abc", "abcd", "bc", "bcd", "cd", "d"] * 4
        dataset.write_text("\n".join(passwords) + "\n", encoding="ascii")
        output = self.path / "model"
        checkpoint = output / "checkpoints" / "checkpoint_100.ckpt"
        with self.kernel() as client:
            notebook = self.notebook(
                "notebook-train.ipynb", training_data=str(dataset), output_dir=str(output),
                batch_size=2, seq_length=4, layer_dim=4, critic_iters=1,
                iters=101, save_every=10,
            )
            self.execute_notebook(client, notebook)
            self.execute_code(client, f"""
assert len(tf.global_variables()) == 148
reader = tf.train.load_checkpoint({str(checkpoint)!r})
saved_names = reader.get_variable_to_shape_map()
assert len(saved_names) == 148
assert set(saved_names) == {{variable.op.name for variable in tf.global_variables()}}
assert all(np.isfinite(reader.get_tensor(name)).all() for name in saved_names)
# At checkpoint 100, Adam has performed 100 generator and 101 discriminator updates.
assert float(reader.get_tensor('beta1_power')) == 2.0 ** -101
assert float(reader.get_tensor('beta1_power_1')) == 2.0 ** -102
assert os.path.normpath(tf.train.latest_checkpoint({str(checkpoint.parent)!r})) == os.path.normpath({str(checkpoint)!r})
""")
            self.assertEqual(
                sorted(path.name for path in checkpoint.parent.glob("*.index")),
                sorted(f"checkpoint_{iteration}.ckpt.index" for iteration in range(10, 101, 10)),
            )
            for iteration in range(10, 101, 10):
                prefix = checkpoint.parent / f"checkpoint_{iteration}.ckpt"
                for suffix in (".index", ".meta", ".data-00000-of-00001"):
                    self.assertTrue(Path(str(prefix) + suffix).is_file(), (iteration, suffix))
            with (output / "charmap.pickle").open("rb") as stream:
                charmap = pickle.load(stream)
            with (output / "charmap_inv.pickle").open("rb") as stream:
                inverse = pickle.load(stream)
            self.assertIsInstance(inverse, list)
            self.assertEqual({character: index for index, character in enumerate(inverse)}, charmap)
            preview = (output / "samples" / "samples_100.txt").read_text().splitlines()
            self.assertEqual(len(preview), 20)
            tokens = "|".join(re.escape(character) for character in inverse)
            for line in preview:
                self.assertRegex(line, rf"^(?:{tokens}){{4}}$")
            self.assertEqual(list((output / "samples").iterdir()), [output / "samples" / "samples_100.txt"])
            self.assertFalse(list(output.rglob("*.jpg")))
            self.assertFalse(list(output.rglob("log.pkl")))

            cli_output = self.path / "cli-generated.txt"
            result = subprocess.run(
                [sys.executable, str(ROOT / "sample.py"), "-i", str(output),
                 "-c", str(checkpoint), "-o", str(cli_output), "-n", "5", "-b", "2",
                 "-l", "4", "-d", "4"],
                cwd=self.path, env=self.environment, capture_output=True, text=True, timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(len(cli_output.read_text().splitlines()), 5)

            # Reuse the training kernel to expose stale optimizer and parameter caches.
            notebook_output = self.path / "notebook-generated.txt"
            notebook_output.write_text("existing-line\n", encoding="ascii")
            sampling = self.notebook(
                "notebook-sample.ipynb", input_dir=str(output), checkpoint=str(checkpoint),
                output=str(notebook_output), batch_size=2, num_samples=5,
                seq_length=4, layer_dim=4,
            )
            self.execute_notebook(client, sampling)
            self.assert_sampling_graph(client)
            generated = notebook_output.read_text().splitlines()
            self.assertEqual(generated[0], "existing-line")
            self.assertEqual(len(generated), 5)
            decoded = "|".join(re.escape(character) for character in inverse if character != "`")
            for line in generated[1:]:
                self.assertRegex(line, rf"^(?:{decoded}){{0,4}}$")


if __name__ == "__main__":
    unittest.main()
