# Training migration

## Scope and environment

This migration extends the pretrained inference work merged at `b9101d53dc80e26a67af6d036e52553a8fe44710` to command-line training. It retains the TensorFlow graph model and WGAN-GP training procedure from the [original TensorFlow 1.13 revision](https://github.com/d4ichi/PassGAN/tree/8ce9ce09a6b915a981e6244c22d34c36cd3bccb4), using `tflib/tf_compat.py` for TensorFlow 2 graph execution. The bundled pretrained assets are unchanged.

The target is Windows x64 CPU with Python 3.12.14, TensorFlow 2.21.0, and NumPy 2.5.4, as described in [the inference migration notes](inference-migration.md). `requirements-training.txt` includes the unchanged `requirements.txt` and adds pinned Matplotlib 3.11.2 and its additional dependencies for the existing training plots. Matplotlib uses the existing `Agg` backend, so training does not need a graphical display.

In the active environment, install the training dependencies before training or running the full test suite:

```powershell
python -m pip install -r requirements-training.txt
python -m pip check
python -m unittest discover -s tests -v
```

GPU execution, other operating systems, and a command-line option for resuming training are outside this migration. Notebook setup and its preserved input/output behavior are described in [the notebook migration notes](notebook-migration.md). No training-time or password-quality claim is made.

## Preserved commands and training behavior

The existing training command remains:

```powershell
python train.py --output-dir output --training-data data/train.txt
```

All existing option names, short options, and defaults are retained:

| Option | Short option | Default |
| --- | --- | --- |
| `--training-data` | `-i` | `data/train.txt` |
| `--output-dir` | `-o` | Required |
| `--save-every` | `-s` | `5000` |
| `--iters` | `-n` | `200000` |
| `--batch-size` | `-b` | `64` |
| `--seq-length` | `-l` | `10` |
| `--layer-dim` | `-d` | `128` |
| `--critic-iters` | `-c` | `10` |
| `--lambda` | `-p` | `10` |

The loop remains inclusive: `--iters N` visits iterations `0` through `N`. Iteration zero updates only the discriminator (critic); iterations `1` through `N` update the generator once and the critic `--critic-iters` times. Consequently, the default still performs 200,000 generator updates and 200,001 groups of critic updates.

The generator and discriminator architecture, variable names, WGAN losses, gradient penalty, and Adam settings (`learning_rate=1e-4`, `beta1=0.5`, `beta2=0.9`) are retained. Input batches are shuffled and incomplete batches are omitted as before. The n-gram validation split and diagnostic calculations are unchanged.

`train.py` now exposes `parse_args`, `build_training_graph`, and `main` so tests can exercise the actual training graph. Importing it does not parse arguments, build a graph, or write output files. Separate calls use separate graphs and reset the plotting state.

## Input handling

Training still reads one password per line using Python's default text encoding and newline handling. Character tokenization, spaces, the maximum-length filter, character vocabulary construction, the `unk` token, and right padding with backticks are unchanged. Lines longer than `--seq-length` remain excluded.

One input correction was explicitly approved: when the last line has no terminating newline, its last character is retained. Previously, unconditional `line[:-1]` removed that character. Lines with normal line endings retain the same interpretation.

The output directory and its `samples` and `checkpoints` subdirectories are created when missing. A dataset containing fewer usable lines than `--batch-size` now produces an error instead of waiting indefinitely for a batch. Nonpositive batch sizes, sequence lengths, layer dimensions, critic counts, or save intervals are rejected; `--iters` may be zero.

## Saved artifacts and generation

The output layout and formats remain:

| Path relative to `--output-dir` | Contents |
| --- | --- |
| `charmap.pickle` | Pickled character-to-ID dictionary |
| `charmap_inv.pickle` | Pickled list mapping IDs to characters |
| `checkpoints/checkpoint_<iteration>.ckpt.*` | TensorFlow checkpoint files and saved graph |
| `checkpoints/checkpoint` | TensorFlow checkpoint state file |
| `samples/samples_<iteration>.txt` | Training preview samples |
| `log.pkl` | Pickled metric history |
| `time.jpg`, `train_disc_cost.jpg`, `js1.jpg` through `js4.jpg` | Existing metric plots |

Checkpoints include generator and discriminator weights plus Adam optimizer state. Saving occurs at positive iteration numbers divisible by `--save-every`, with the existing checkpoint prefix. Old checkpoints are retained. There is no forced final save: for example, `--iters 101 --save-every 100` saves iteration 100, and a run ending before its first positive save interval creates no checkpoint.

Training previews are written every 100 positive iterations, with ten batches per preview file. They retain padding backticks and the original text-file behavior. The sampling command retains its existing behavior of removing padding and writing UTF-8 strings, one per line.

For example, after the default training command has saved iteration 200,000:

```powershell
python sample.py `
	--input-dir output `
	--checkpoint output/checkpoints/checkpoint_200000.ckpt `
	--output generated_pass.txt `
	--batch-size 1024 `
	--num-samples 1000000
```

If training used different `--seq-length` or `--layer-dim` values, pass the same values to `sample.py`. The saved character dictionaries and checkpoint must come from the same training run. As before, sampling does not need the discriminator or optimizer variables to be present in its graph.

## Validation coverage

The five training tests supplement the nine inference tests:

- Real generator and discriminator optimizer updates change their respective weights while leaving the other model's weights unchanged. Losses and saved values must be finite.
- A newly built training graph restores every saved model and optimizer variable exactly, with no uninitialized variables remaining.
- A subprocess runs `train.py` on a small artificial dataset for iterations 0 through 101, saving every 10 iterations, then a separate subprocess runs `sample.py` on checkpoint 100 and the resulting dictionaries. The test checks that all ten checkpoints are retained and no final checkpoint 101 is added, along with 20 training preview lines, metric history, JPEG plots, and exactly five generated output lines.
- Dataset handling preserves the final character without a newline, trailing spaces, existing newline handling, filtering, and padding. Invalid arguments and an undersized dataset fail without hanging.
- Importing the training module has no argument-parsing, graph-building, or output-file side effects.

The end-to-end test uses batch size 2, sequence length 4, layer dimension 4, and one critic update per iteration. Its 101 generator updates and 102 critic updates establish that training, saving, and generation execute on CPU; they do not establish convergence or useful password quality. The artificial dataset is created locally by the test and requires no external data download.

The Windows CPU workflow runs checkpoint compatibility, pretrained sampling, and training tests as separate steps with the same pinned training environment. Execution results are recorded in the workflow runs and pull request checks.

Local validation in a fresh uv-managed Python 3.12.14 environment installed from `requirements-training.txt` passed all 14 tests. `python -m pip check` reported no broken requirements. The existing inference pins and bundled pretrained assets have no changes.
