# Notebook migration

## Scope and dependencies

`notebook-sample.ipynb` and `notebook-train.ipynb` use `tflib/tf_compat.py` for TensorFlow 2 graph execution, as do the command-line scripts. The notebook cell structure, `VirtualArgparse` defaults, model architecture, checkpoint formats, and input/output behavior are retained.

`requirements-notebooks.txt` includes `requirements-training.txt` and adds pinned Jupyter Notebook 7.6.3, nbclient 0.11.0, nbformat 5.11.1, ipykernel 7.4.0, and their additional dependencies. It uses the same Python 3.12.14, TensorFlow 2.21.0, and NumPy 2.5.4 environment as the command-line scripts.

## Running the notebooks

Follow [the notebook setup commands in the README](../README.md#jupyter-notebooks) from the repository root with the environment active. Select the Python kernel for that environment, edit the parameters in the `VirtualArgparse` cell, and run the cells in order. Paths are relative to the repository root.

The notebooks configure parameters in `VirtualArgparse` rather than parsing command-line arguments. When sampling from a custom checkpoint, use its matching character dictionaries and the `seq_length` and `layer_dim` values used during training.

The generator-modeling cell in the sampling notebook and the first modeling cell in the training notebook reset the default TensorFlow graph. When changing model parameters, rerun the notebook in order from the parameter cell so that later cells use the new graph and values. The training session also resets in-memory plotting state before starting.

## Sampling behavior

The sampling notebook retains these defaults:

| Parameter | Default |
| --- | --- |
| `input_dir` | `pretrained` |
| `checkpoint` | `pretrained/checkpoints/checkpoint_200000.ckpt` |
| `output` | `generated_pass.txt` |
| `batch_size` | `1024` |
| `num_samples` | `1000000` |
| `seq_length` | `10` |
| `layer_dim` | `128` |

The notebook appends samples to the output file using Python's default text encoding and newline handling. It generates only complete batches, so the number of added lines is `floor(num_samples / batch_size) * batch_size`. Repeating the session cell appends another set of samples. Padding backticks are removed from generated strings.

These are the original notebook semantics. The command-line `sample.py` instead writes exactly the requested number of samples, including a partial final batch, uses UTF-8, and overwrites the output file by default unless `--append` is supplied.

## Training behavior

Prepare the training data as described in [Training your own models](../README.md#training-your-own-models). The dataset must contain at least one full batch of usable lines after the sequence-length filter.

The training notebook retains these defaults:

| Parameter | Default |
| --- | --- |
| `training_data` | `data/train.txt` |
| `output_dir` | `pretrained` |
| `save_every` | `10` |
| `iters` | `100` |
| `batch_size` | `64` |
| `seq_length` | `10` |
| `layer_dim` | `128` |
| `critic_iters` | `10` |
| `lamb` | `10` |

Set `output_dir` before running the directory and dictionary cells. Its original default, `pretrained`, writes character dictionaries and new checkpoints into the bundled model directory. Choose another directory to keep the bundled files. Training initializes a new model; the output directory does not select a checkpoint to restore.

The loop visits iterations `0` through `iters`, with no generator update at iteration zero. Checkpoints are saved at positive multiples of `save_every`, with no additional forced save at the end. Training previews are written every 100 positive iterations as ten batches to `samples/samples_<iteration>.txt`. As before, preview strings retain padding backticks and use Python's default text-file behavior.

Character dictionaries remain `charmap.pickle` and `charmap_inv.pickle`, with checkpoints under `checkpoints/checkpoint_<iteration>.ckpt.*`. Use the dictionaries from the same training run when sampling from its checkpoints.

The original `lib.plot.flush()` and `lib.plot.tick()` calls remain commented out. Metrics are recorded in memory, but the notebook does not write the command-line training script's `log.pkl` or JPEG plots by default.

## Notebook tests

After installing `requirements-notebooks.txt`, run the notebook tests with:

```powershell
python -m unittest tests.test_notebooks -v
```

The notebook tests are skipped when their optional dependencies are not installed. The existing command-line tests can still run with `requirements-training.txt` alone. Installing the notebook dependencies also enables the notebook tests when running `python -m unittest discover -s tests -v`.
