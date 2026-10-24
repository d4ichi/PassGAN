"""Compare migrated inference against the repository's saved TF1 graph.

Both graphs execute in the installed TensorFlow runtime. This deliberately
isolates model graph changes; it is not a comparison against a TF1 runtime.
"""

from pathlib import Path
import unittest

import numpy as np

import models
import sample
import tflib as lib
from tflib.tf_compat import tf


ROOT = Path(__file__).resolve().parents[1]
PRETRAINED = ROOT / "pretrained"
CHECKPOINT = str(PRETRAINED / "checkpoints" / "checkpoint_200000.ckpt")
SESSION_CONFIG = tf.ConfigProto(device_count={"GPU": 0})


class CheckpointCompatibilityTests(unittest.TestCase):
    def test_restored_weights_and_fixed_noise_match_saved_graph(self):
        charmap, _ = sample.load_charmaps(PRETRAINED)
        # The bundled graph fixes its generator batch dimension to 64.
        noise = np.random.default_rng(2026).normal(size=(64, 128)).astype(np.float32)
        modern_graph = tf.Graph()
        with modern_graph.as_default():
            output = models.Generator(64, 10, 128, len(charmap), noise=noise)
            variables = tf.global_variables()
            saver = tf.train.Saver(var_list=variables)
            uninitialized = tf.report_uninitialized_variables()

        reader = tf.train.load_checkpoint(CHECKPOINT)
        legacy_graph = tf.Graph()
        with legacy_graph.as_default():
            legacy_saver = tf.train.import_meta_graph(
                CHECKPOINT + ".meta", clear_devices=True
            )
            # Trainable weights exclude the saved Adam optimizer slot variables.
            saved_generator_names = {
                variable.op.name for variable in tf.trainable_variables()
                if variable.op.name.startswith("Generator.")
            }
        current_names = {variable.op.name for variable in variables}
        self.assertEqual(current_names, saved_generator_names)
        self.assertTrue(current_names, "The test must actually restore generator weights")
        sample.validate_checkpoint(CHECKPOINT, variables)
        with tf.Session(graph=modern_graph, config=SESSION_CONFIG) as session:
            saver.restore(session, CHECKPOINT)
            self.assertEqual(session.run(uninitialized).tolist(), [])
            for variable in variables:
                with self.subTest(weight=variable.op.name):
                    np.testing.assert_array_equal(
                        session.run(variable), reader.get_tensor(variable.op.name)
                    )
            modern_output = session.run(output)

        with tf.Session(graph=legacy_graph, config=SESSION_CONFIG) as session:
            legacy_saver.restore(session, CHECKPOINT)
            legacy_output = session.run(
                legacy_graph.get_tensor_by_name("Reshape_2:0"),
                feed_dict={legacy_graph.get_tensor_by_name("random_normal:0"): noise},
            )

        self.assertEqual(modern_output.shape, (64, 10, len(charmap)))
        self.assertEqual(legacy_output.shape, modern_output.shape)
        self.assertTrue(np.isfinite(modern_output).all())
        np.testing.assert_allclose(modern_output.sum(axis=2), 1.0, atol=1e-6)
        np.testing.assert_allclose(modern_output, legacy_output, rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(
            modern_output.argmax(axis=2), legacy_output.argmax(axis=2)
        )

    def test_batch_one_and_parameter_cache_are_scoped_to_each_graph(self):
        graphs = []
        for _ in range(2):
            graph = tf.Graph()
            with graph.as_default():
                noise = np.zeros((1, 128), dtype=np.float32)
                output = models.Generator(1, 3, 8, 7, noise=noise)
                variables = list(tf.global_variables())
                models.Generator(1, 3, 8, 7, noise=noise)
                self.assertEqual(tf.global_variables(), variables)
                self.assertEqual(lib.params_with_name("Generator."), variables)
                initializer = tf.global_variables_initializer()
            with tf.Session(graph=graph, config=SESSION_CONFIG) as session:
                session.run(initializer)
                result = session.run(output)
            self.assertEqual(result.shape, (1, 3, 7))
            np.testing.assert_allclose(result.sum(axis=2), 1.0, atol=1e-6)
            self.assertTrue(all(variable.graph is graph for variable in variables))
            graphs.append((graph, variables))
        self.assertEqual(
            [variable.op.name for variable in graphs[0][1]],
            [variable.op.name for variable in graphs[1][1]],
        )
        self.assertTrue(all(
            first is not second
            for first, second in zip(graphs[0][1], graphs[1][1])
        ))


if __name__ == "__main__":
    unittest.main()
