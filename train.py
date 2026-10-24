"""Train PassGAN using the original TensorFlow graph and checkpoint format."""

import argparse
import os
import pickle
import sys
import time

import numpy as np

import models
import tflib as lib
from tflib.tf_compat import tf
import utils


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
    parser.add_argument('--training-data', '-i', default='data/train.txt',
                        dest='training_data',
                        help='Path to training data file (one password per line) (default: data/train.txt)')
    parser.add_argument('--output-dir', '-o', required=True, dest='output_dir',
                        help="Output directory. If directory doesn't exist it will be created.")
    parser.add_argument('--save-every', '-s', type=positive_int, default=5000,
                        dest='save_every',
                        help='Save model checkpoints after this many iterations (default: 5000)')
    parser.add_argument('--iters', '-n', type=nonnegative_int, default=200000,
                        dest='iters',
                        help='The number of training iterations (default: 200000)')
    parser.add_argument('--batch-size', '-b', type=positive_int, default=64,
                        dest='batch_size', help='Batch size (default: 64).')
    parser.add_argument('--seq-length', '-l', type=positive_int, default=10,
                        dest='seq_length', help='The maximum password length (default: 10)')
    parser.add_argument('--layer-dim', '-d', type=positive_int, default=128,
                        dest='layer_dim',
                        help='The hidden layer dimensionality for the generator and discriminator (default: 128)')
    parser.add_argument('--critic-iters', '-c', type=positive_int, default=10,
                        dest='critic_iters',
                        help='The number of discriminator weight updates per generator update (default: 10)')
    parser.add_argument('--lambda', '-p', type=int, default=10,
                        dest='lamb', help='The gradient penalty lambda hyperparameter (default: 10)')
    return parser.parse_args(argv)


def build_training_graph(args, vocab_size):
    """Build the original WGAN-GP losses and Adam updates in the current graph."""
    real_inputs_discrete = tf.placeholder(tf.int32, shape=[args.batch_size, args.seq_length])
    real_inputs = tf.one_hot(real_inputs_discrete, vocab_size)
    fake_inputs = models.Generator(args.batch_size, args.seq_length, args.layer_dim,
                                   vocab_size)
    disc_real = models.Discriminator(real_inputs, args.seq_length, args.layer_dim,
                                     vocab_size)
    disc_fake = models.Discriminator(fake_inputs, args.seq_length, args.layer_dim,
                                     vocab_size)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
    gen_cost = -tf.reduce_mean(disc_fake)

    # WGAN Lipschitz penalty, with the original interpolation and reduction axes.
    alpha = tf.random_uniform(shape=[args.batch_size, 1, 1], minval=0., maxval=1.)
    differences = fake_inputs - real_inputs
    interpolates = real_inputs + (alpha * differences)
    gradients = tf.gradients(
        models.Discriminator(interpolates, args.seq_length, args.layer_dim,
                             vocab_size), [interpolates]
    )[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2]))
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
    disc_cost += args.lamb * gradient_penalty

    gen_params = lib.params_with_name('Generator')
    disc_params = lib.params_with_name('Discriminator')
    gen_train_op = tf.train.AdamOptimizer(
        learning_rate=1e-4, beta1=0.5, beta2=0.9
    ).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.AdamOptimizer(
        learning_rate=1e-4, beta1=0.5, beta2=0.9
    ).minimize(disc_cost, var_list=disc_params)
    return {
        'real_inputs_discrete': real_inputs_discrete,
        'fake_inputs': fake_inputs,
        'disc_cost': disc_cost,
        'gen_cost': gen_cost,
        'gen_train_op': gen_train_op,
        'disc_train_op': disc_train_op,
    }


def train(args):
    # Plotting remains a training-only dependency.
    import tflib.plot

    lines, charmap, inv_charmap = utils.load_dataset(
        path=args.training_data, max_length=args.seq_length)
    if len(lines) < args.batch_size:
        raise ValueError(
            f'Training data must contain at least --batch-size ({args.batch_size}) '
            f'usable lines; loaded {len(lines)}.'
        )

    for directory in (args.output_dir,
                      os.path.join(args.output_dir, 'samples'),
                      os.path.join(args.output_dir, 'checkpoints')):
        os.makedirs(directory, exist_ok=True)

    # Preserve the character IDs and the original pickle file names and types.
    with open(os.path.join(args.output_dir, 'charmap.pickle'), 'wb') as stream:
        pickle.dump(charmap, stream)
    with open(os.path.join(args.output_dir, 'charmap_inv.pickle'), 'wb') as stream:
        pickle.dump(inv_charmap, stream)
    print('Number of unique characters in dataset: {}'.format(len(charmap)))

    def inf_train_gen():
        while True:
            np.random.shuffle(lines)
            for offset in range(0, len(lines) - args.batch_size + 1, args.batch_size):
                yield np.array(
                    [[charmap[char] for char in line]
                     for line in lines[offset:offset + args.batch_size]],
                    dtype='int32'
                )

    # Keep the original held-out split and n-gram diagnostics.
    true_char_ngram_lms = [utils.NgramLanguageModel(
        size + 1, lines[10 * args.batch_size:], tokenize=False) for size in range(4)]
    validation_char_ngram_lms = [utils.NgramLanguageModel(
        size + 1, lines[:10 * args.batch_size], tokenize=False) for size in range(4)]
    for size in range(4):
        print('validation set JSD for n={}: {}'.format(
            size + 1, true_char_ngram_lms[size].js_with(validation_char_ngram_lms[size])))
    true_char_ngram_lms = [utils.NgramLanguageModel(
        size + 1, lines, tokenize=False) for size in range(4)]

    lib.plot.reset()
    lib.plot.output_dir = args.output_dir
    with tf.Graph().as_default():
        operations = build_training_graph(args, len(charmap))
        with tf.Session() as session:
            print('Starting TensorFlow session...')
            print('Local current time :', time.asctime(time.localtime(time.time())))
            session.run(tf.global_variables_initializer())

            def generate_samples():
                samples = np.argmax(session.run(operations['fake_inputs']), axis=2)
                return [tuple(inv_charmap[index] for index in row) for row in samples]

            batches = inf_train_gen()
            # Iteration zero updates only the critic, as in the original code.
            for iteration in range(args.iters + 1):
                start_time = time.time()
                if iteration > 0:
                    session.run(operations['gen_train_op'])
                for _ in range(args.critic_iters):
                    disc_cost, _ = session.run(
                        [operations['disc_cost'], operations['disc_train_op']],
                        feed_dict={operations['real_inputs_discrete']: next(batches)}
                    )
                lib.plot.plot('time', time.time() - start_time)
                lib.plot.plot('train disc cost', disc_cost)

                if iteration % 100 == 0 and iteration > 0:
                    samples = []
                    for _ in range(10):
                        samples.extend(generate_samples())
                    for size in range(4):
                        language_model = utils.NgramLanguageModel(size + 1, samples,
                                                                 tokenize=False)
                        lib.plot.plot('js{}'.format(size + 1),
                                      language_model.js_with(true_char_ngram_lms[size]))
                    path = os.path.join(args.output_dir, 'samples',
                                        'samples_{}.txt'.format(iteration))
                    with open(path, 'w') as stream:
                        for row in samples:
                            # Training previews retain padding; sample.py removes it.
                            stream.write(''.join(row) + '\n')

                if iteration % args.save_every == 0 and iteration > 0:
                    # A fresh saver preserves the original retention of old checkpoints.
                    model_saver = tf.train.Saver()
                    model_saver.save(session, os.path.join(
                        args.output_dir, 'checkpoints',
                        'checkpoint_{}.ckpt'.format(iteration)))
                    print('{} / {} ({}%)'.format(
                        iteration, args.iters, iteration / args.iters * 100.0))
                if iteration == args.iters:
                    print('...Training done.')
                if iteration % 100 == 0:
                    lib.plot.flush()
                lib.plot.tick()

    print('Ending TensorFlow session.')
    print('Local current time :', time.asctime(time.localtime(time.time())))


def main(argv=None):
    args = parse_args(argv)
    try:
        train(args)
    except (OSError, ValueError, tf.errors.OpError) as error:
        print(f'error: {error}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
