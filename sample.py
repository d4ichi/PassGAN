"""Generate samples from an existing PassGAN TensorFlow 1 checkpoint."""

import argparse
from pathlib import Path
import pickle
import sys
import time

import models
from tflib.tf_compat import tf


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", "-i", required=True, type=Path,
                        help="Trained model directory containing the character maps.")
    parser.add_argument("--checkpoint", "-c", required=True, type=Path,
                        help="Checkpoint prefix, without .index or .data suffixes.")
    parser.add_argument("--output", "-o", default=Path("samples.txt"), type=Path,
                        help="UTF-8 output file (default: samples.txt; overwritten).")
    parser.add_argument("--num-samples", "-n", type=nonnegative_int, default=1000000,
                        help="Exact number of samples to write (default: 1000000).")
    parser.add_argument("--batch-size", "-b", type=positive_int, default=64,
                        help="Generation batch size (default: 64).")
    parser.add_argument("--seq-length", "-l", type=positive_int, default=10,
                        help="Sequence length used during training (default: 10).")
    parser.add_argument("--layer-dim", "-d", type=positive_int, default=128,
                        help="Hidden dimension used during training (default: 128).")
    parser.add_argument("--append", action="store_true",
                        help="Append to the output file instead of overwriting it.")
    return parser.parse_args(argv)


def load_charmaps(input_dir):
    """Read the original character IDs without rebuilding or reordering them."""
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise ValueError(f"Trained model directory does not exist: {input_dir}")
    with (input_dir / "charmap.pickle").open("rb") as stream:
        charmap = pickle.load(stream, encoding="latin1")
    with (input_dir / "charmap_inv.pickle").open("rb") as stream:
        inverse = pickle.load(stream, encoding="latin1")
    if not isinstance(charmap, dict) or not isinstance(inverse, (list, tuple)):
        raise ValueError("Invalid character maps: expected a dictionary and an inverse list.")
    if not charmap or len(charmap) != len(inverse):
        raise ValueError("Character maps must be nonempty and have matching vocabulary sizes.")
    if any(not isinstance(char, str) or type(index) is not int
           for char, index in charmap.items()):
        raise ValueError("Character map entries must map strings to integer IDs.")
    if set(charmap.values()) != set(range(len(inverse))):
        raise ValueError("Character map IDs must be consecutive, starting at zero.")
    if any(inverse[index] != char for char, index in charmap.items()):
        raise ValueError("Character maps disagree about the character ID mapping.")
    return charmap, inverse


def validate_checkpoint(checkpoint, variables):
    """Require every inference variable to exist with the correct shape and dtype."""
    try:
        reader = tf.train.load_checkpoint(str(checkpoint))
    except (tf.errors.OpError, ValueError) as error:
        raise ValueError(
            f"Cannot read checkpoint {checkpoint}: provide a prefix with readable "
            ".index and .data files."
        ) from error
    shapes = reader.get_variable_to_shape_map()
    dtypes = reader.get_variable_to_dtype_map()
    problems = []
    for variable in variables:
        name = variable.op.name
        if name not in shapes:
            problems.append(f"missing {name}")
        elif list(shapes[name]) != variable.shape.as_list():
            problems.append(f"shape mismatch for {name}: checkpoint {shapes[name]}, "
                            f"model {variable.shape.as_list()}")
        elif dtypes[name] != variable.dtype.base_dtype:
            problems.append(f"dtype mismatch for {name}")
    if problems:
        raise ValueError("Checkpoint is incompatible with the model: "
                         + "; ".join(problems[:4])
                         + ". Check --seq-length, --layer-dim, and the character maps.")


def generate(args):
    charmap, inverse = load_charmaps(args.input_dir)
    graph = tf.Graph()
    with graph.as_default():
        probabilities = models.Generator(args.batch_size, args.seq_length,
                                         args.layer_dim, len(charmap))
        sample_ids = tf.argmax(probabilities, axis=2, output_type=tf.int32)
        variables = tf.global_variables()
        validate_checkpoint(args.checkpoint, variables)
        saver = tf.train.Saver(var_list=variables)
        uninitialized = tf.report_uninitialized_variables(variables)

    start = time.monotonic()
    written = 0
    with tf.Session(graph=graph) as session:
        saver.restore(session, str(args.checkpoint))
        if session.run(uninitialized).size:
            raise ValueError("Checkpoint restore left uninitialized generator variables.")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if args.append else "w"
        with args.output.open(mode, encoding="utf-8", newline="\n") as stream:
            for offset in range(0, args.num_samples, args.batch_size):
                count = min(args.batch_size, args.num_samples - offset)
                ids = session.run(sample_ids)[:count]
                for row in ids:
                    # Preserve the pretrained model's original padding convention.
                    text = "".join(inverse[index] for index in row).replace('`', "")
                    stream.write(text + "\n")
                written += len(ids)
    print(f"Wrote {written} samples to {args.output} "
          f"in {time.monotonic() - start:.2f} seconds.")
    return written


def main(argv=None):
    args = parse_args(argv)
    try:
        generate(args)
    except (OSError, ValueError, EOFError, pickle.UnpicklingError, tf.errors.OpError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
