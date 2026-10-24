# Pretrained inference migration

## Scope and provenance

The source and pretrained assets were obtained from `https://github.com/d4ichi/PassGAN` at commit `8ce9ce09a6b915a981e6244c22d34c36cd3bccb4`. The checkpoint, character dictionaries, and saved graph are unchanged. No retraining or training-data download is required.

The inference path is `sample.py` -> `models.py` -> `tflib` linear/convolution helpers. TensorFlow 1 graph execution is provided through `tflib/tf_compat.py`. The generator architecture, parameter names, shapes, residual scaling, softmax, character IDs, and padding removal are preserved. Variables are cached per graph. Conv1D only squeezes the added fourth axis, so a batch of one retains its batch dimension.

## Verified environment

| Component | Version / target |
| --- | --- |
| OS | Windows 11 x64 (build 26300) |
| Execution | CPU |
| Python | 3.12.14 |
| TensorFlow | 2.21.0 |
| NumPy | 2.5.4 |
| Keras (transitive dependency) | 3.15.1 |
| protobuf (transitive dependency) | 7.36.2 |
| uv (Python bootstrap tool) | 0.13.5 |

`.python-version` records the Python patch version. `requirements.txt` contains the exact resolved inference package versions, including transitive dependencies. It was generated from a dedicated virtual environment with only TensorFlow and NumPy requested directly. Matplotlib and the obsolete separate `tensorflow-gpu` package are not required for inference.

Python 3.12.14 is obtained on Windows through uv's Astral-managed CPython distribution. The GitHub Actions `actions/python-versions` manifest does not provide a Windows build for that patch, so CI uses pinned `setup-uv` and uv 0.13.5 instead of `setup-python`. uv is a bootstrap tool and is kept out of the inference requirements.

TensorFlow 2.21.0's [PyPI metadata](https://pypi.org/pypi/tensorflow/2.21.0/json) includes a Python 3.12 Windows x64 wheel and permits NumPy >=1.26.0. Native Windows CPU is the validated target; GPU/WSL2, Linux, and macOS need separate validation and may need their own dependency pins.

## Validation

Run the README installation and test commands from a clean virtual environment. Checks cover:

- Reading the original checkpoint and matching all 24 required generator tensors by name, shape, dtype, and exact restored values; no generator variables remain uninitialized.
- Feeding the same explicit float32 noise array (64 x 128) into the migrated generator and the original `.meta` graph. Probabilities must match within `rtol=1e-5, atol=1e-6`; selected character IDs must match exactly.
- Preserving the batch dimension for batch size one and preventing variable sharing between independent graphs.
- Exact output counts for one sample, less than a batch, partial batches, and zero samples; explicit append and default overwrite; import behavior and invalid arguments, checkpoint dimensions, and dictionaries.

Both graphs execute with the installed TensorFlow 2 runtime on CPU. This verifies the graph migration and weight restoration, not numerical identity to a running TensorFlow 1.13 installation or equivalence of random streams across versions. No password-quality or training-performance claim is made.

Local validation completed on 2026-10-24 (Asia/Tokyo):

- All 9 unittest cases passed in both the initial environment and fresh environments installed from `requirements.txt`, including an environment created with uv-managed Python 3.12.14.
- `python -m pip check` reported no broken requirements.
- A generation smoke test produced exactly 1,000 lines. The bundled checkpoint also generated one sample with batch size one.
- The pretrained assets have no Git changes. Python syntax checks and `git diff --check` passed.

GitHub Actions results are reported separately in the workflow runs and pull request checks.

## Updating dependencies later

1. Select a stable TensorFlow release with a wheel for the target Python and OS; check its NumPy constraints. Record the candidate versions before editing code.
2. Create a new empty virtual environment. Install the selected TensorFlow and NumPy versions with exact `==` specifiers.
3. Run checkpoint compatibility and CLI tests. Use fixed input noise for numeric comparisons; a matching random seed alone is insufficient.
4. Regenerate `requirements.txt` from that dedicated environment using `python -m pip freeze`, retaining the environment comments. Update `.python-version`, this environment table, and README as appropriate.
5. Install the resulting `requirements.txt` into another empty virtual environment, run `python -m pip check`, and rerun the tests. Commit the pins together with the tested source changes.

Training and notebook migration, Keras model conversion, and GPU support are subsequent stages.
