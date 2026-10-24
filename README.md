# PassGAN


This repository is an updated version of [@brannondorsey/PassGAN](https://github.com/brannondorsey/PassGAN). It contains code for the [_PassGAN: A Deep Learning Approach for Password Guessing_](https://arxiv.org/abs/1709.00440) paper.

Pretrained sampling now supports Python 3.12 and TensorFlow 2 using `tf.compat.v1`. The training code and notebooks still target the legacy TensorFlow 1.x environment and have not been migrated.

The PassGAN model is based on [_Improved Training of Wasserstein GANs_](https://arxiv.org/abs/1704.00028). It is assumed that the PassGAN authors used the [improved_wgan_training](https://github.com/igul222/improved_wgan_training) TensorFlow implementation in their work.

This repository includes:

- Command-line interfaces: `sample.py` and `train.py`
- A pretrained PassGAN model trained on the RockYou dataset
- Jupyter notebooks for debugging: `notebook-sample.ipynb` and `notebook-train.ipynb`

## Getting Started

Pretrained sampling has been verified on Windows 11 x64 (CPU) with Python 3.12.14, TensorFlow 2.21.0, and NumPy 2.5.4. CUDA is not required. The Python version is pinned in `.python-version`, and package versions are pinned in `requirements.txt`.

Run the following commands from the repository root in PowerShell to set up an isolated environment. Using a virtual environment and [uv](https://docs.astral.sh/uv/guides/install-python/) is optional. uv downloads the pinned Python version, and the virtual environment isolates this project's dependencies from those of other projects.

```powershell
py -3 -m venv .venv-bootstrap
.\.venv-bootstrap\Scripts\Activate.ps1
python -m pip install uv==0.13.5
uv venv --python 3.12.14 --managed-python --seed .venv
deactivate

.\.venv\Scripts\Activate.ps1
python --version
python -m pip install -r requirements.txt
python -m pip check
```

Check that the reported Python version matches `.python-version`. Keep the environment active for the commands below. In a new PowerShell session, activate it again with `.\.venv\Scripts\Activate.ps1`.

This setup has not been tested on other platforms or with GPU acceleration. For TensorFlow GPU support on Windows, see the [WSL2 installation guide](https://www.tensorflow.org/install/pip).

### Generating password samples

Use the pretrained model to generate 1,000,000 passwords, saving them to `generated_pass.txt`.

```powershell
python sample.py `
	--input-dir pretrained `
	--checkpoint pretrained/checkpoints/checkpoint_200000.ckpt `
	--output generated_pass.txt `
	--batch-size 1024 `
	--num-samples 1000000
```

Samples are written in UTF-8, one per line. The command writes exactly the requested number of samples, even when the final batch is incomplete. By default, **the output file is overwritten**; add `--append` to append samples. With the default behavior, setting `--num-samples` to `0` produces an empty file. Samples may repeat or be empty.

Use the checkpoint prefix without the `.index`, `.meta`, or `.data-*` suffix. Sampling requires the `.index` and `.data-*` files and matching character dictionaries. The `.meta` file is needed only for the compatibility tests. For custom models, set `--seq-length` and `--layer-dim` to the values used during training. The bundled model uses 10 and 128, respectively.

Run `python sample.py --help` to see all options.

### Verification

Run the tests using the bundled checkpoint and saved graph.

```powershell
python -m unittest discover -s tests -v
```

The tests verify all 24 generator tensors and compare outputs for fixed input noise against the saved `.meta` graph. They also cover a batch size of one, output counts, file handling, and invalid inputs. Both graphs run on TensorFlow 2.21.0. A comparison with a TensorFlow 1.13 runtime has not been performed.

The Windows CPU CI workflow uses the same Python version and pinned package versions. See [the migration notes](docs/inference-migration.md) for details about the environment and verification.

### Training your own models

> The instructions below target the original TensorFlow 1.x environment. The training code and notebooks have not been migrated to TensorFlow 2 or tested with the current inference dependencies. To reproduce the original setup, use the source code and requirements from the [original TensorFlow 1.13 revision](https://github.com/d4ichi/PassGAN/tree/8ce9ce09a6b915a981e6244c22d34c36cd3bccb4). The CUDA prerequisite and GTX 1080 training-time estimate below refer to the original environment.

```bash
# requires CUDA 8 to be pre-installed
pip3 install -r requirements.txt
```

You can download sample datasets from the releases page or generate a sample RockYou dataset yourself using the scripts in `bin`.

Training a model on a large dataset (100 MB or more) can take several hours on a GTX 1080.

```bash
# download the rockyou training data
# contains 80% of the RockYou passwords (including repeats)
# that are 10 characters or fewer
curl -L -o data/train.txt https://github.com/d4ichi/PassGAN/releases/download/data/rockyou-test.txt

# train for 200000 iterations, saving checkpoints every 5000
# uses the default hyperparameters from the paper
python train.py --output-dir output --training-data data/train.txt
```


You are encouraged to train using your own password leaks and datasets. Some great places to find those include:

- [LinkedIn leak](https://github.com/brannondorsey/PassGAN/releases/download/data/68_linkedin_found_hash_plain.txt.zip) (1.7 GB compressed; direct download; mirror from [Hashes.org](https://hashes.org/leaks.php))
- [Exploit.in torrent](https://thepiratebay.org/torrent/16016494/exploit.in) (10 GB+, 800 million accounts. Infamous!)
- [Hashes.org](https://hashes.org/leaks.php): An awesome site for collaborative password recovery. Consider donating if you have the resources ;)



## Attribution and License

This code is released under the [MIT License](https://github.com/igul222/improved_wgan_training/blob/master/LICENSE). You are free to use, modify, distribute, or sell it under those terms.

Credit for the code in this repository goes to [@igul222](https://github.com/igul222) for his work on [improved_wgan_training](https://github.com/igul222/improved_wgan_training) and [@brannondorsey](https://github.com/brannondorsey) for adapting it for PassGAN.

This is an updated version of their work.

The [PassGAN paper](https://arxiv.org/abs/1709.00440) was written by Briland Hitaj, Paolo Gasti, Giuseppe Ateniese, and Fernando Perez-Cruz.
