# PassGAN


This repository is updated version of [@brannondorsey/PassGAN](https://github.com/brannondorsey/PassGAN) for Python 3 & TensorFlow 1.13, contains code for the [_PassGAN: A Deep Learning Approach for Password Guessing_](https://arxiv.org/abs/1709.00440) paper. 

The model from PassGAN is taken from [_Improved Training of Wasserstein GANs_](https://arxiv.org/abs/1704.00028) and it is assumed that the authors of PassGAN used the [improved_wgan_training](https://github.com/igul222/improved_wgan_training) tensorflow implementation in their work. 

This repo contributes:

- A command-line interface `sample.py` `train.py`
- A pretrained PassGAN model trained on the RockYou dataset
- Jupyter notebook for debugging `notebook-sample.py` `notebook-train.py`

## Getting Started

```bash
# requires CUDA 8 to be pre-installed
pip3 install -r requirements.txt
```

### Generating password samples

Use the pretrained model to generate 1,000,000 passwords, saving them to `generated_pass.txt`.

```bash
python sample.py \
	--input-dir pretrained \
	--checkpoint pretrained/checkpoints/checkpoint_200000.ckpt \
	--output generated_pass.txt \
	--batch-size 1024 \
	--num-samples 1000000
```

### Training your own models

You can downlaod sample datasets from release page, or generate sample rockyou dataset by yourself with codes under `bin`.

Training a model on a large dataset (100MB+) can take several hours on a GTX 1080.

```bash
# download the rockyou training data
# contains 80% of the full rockyou passwords (with repeats)
# that are 10 characters or less
curl -L -o data/train.txt https://github.com/d4ichi/PassGAN/releases/download/data/rockyou-test.txt

# train for 200000 iterations, saving checkpoints every 5000
# uses the default hyperparameters from the paper
python train.py --output-dir output --training-data data/train.txt
```


You are encouraged to train using your own password leaks and datasets. Some great places to find those include:

- [LinkedIn leak](https://github.com/brannondorsey/PassGAN/releases/download/data/68_linkedin_found_hash_plain.txt.zip) (1.7GB compressed, direct download. Mirror from [Hashes.org](https://hashes.org/leaks.php))
- [Exploit.in torrent](https://thepiratebay.org/torrent/16016494/exploit.in) (10GB+, 800 million accounts. Infamous!)
- [Hashes.org](https://hashes.org/leaks.php): Awesome shared password recovery site. Consider donating if you have the resources ;)



## Attribution and License

This code is released under an [MIT License](https://github.com/igul222/improved_wgan_training/blob/master/LICENSE). You are free to use, modify, distribute, or sell it under those terms. 

The credit for the code in this repository goes to [@igul222](https://github.com/igul222) for his work on the [improved_wgan_training](https://github.com/igul222/improved_wgan_training) and [@brannondorsey](https://github.com/brannondorsey) for specializing it in the PassGAN paper.

This is updated version for Python 3 / TensorFlow 1.13 of their work.

The PassGAN [research and paper](https://arxiv.org/abs/1709.00440) was published by Briland Hitaj, Paolo Gasti, Giuseppe Ateniese, Fernando Perez-Cruz.
