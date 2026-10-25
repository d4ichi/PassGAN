# Generating a RockYou sample dataset
These scripts prepare a RockYou dataset in `data` for training and testing.

Use the Python setup described in [Getting Started](../README.md#getting-started). Activate your virtual environment if applicable. A POSIX `sh`, `wget`, and `bzip2` must be available.

Run the following commands from this `bin` directory:

```sh
sh get-dataset.sh
python make-rockyou-full.py
python prep-data.py
```

1. `get-dataset.sh` downloads and extracts `rockyou-withcount.txt.bz2` into `../data/rockyou-withcount.txt`.
2. `make-rockyou-full.py` repeats each password according to its count and writes `../data/rockyou-full.txt`.
3. `prep-data.py` shuffles the expanded lines with seed `1337`, then writes an 80% training split to `../data/train.txt` and the remainder to `../data/test.txt`.

The original count file is decoded as ISO-8859-1. `rockyou-full.txt`, `train.txt`, and `test.txt` use UTF-8, which is also used to read training data in the command-line script and training notebook.

The expansion step strips surrounding whitespace from input records and removes spaces from passwords. The preparation scripts do not filter by password length.

The Python scripts overwrite their output files at these fixed paths. Dataset files are written directly; no shell output redirection is needed.
