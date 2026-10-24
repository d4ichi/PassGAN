"""TensorFlow runtime shared by the legacy checkpoint inference code.

Import this module before creating tensors. The original model uses TensorFlow
1.x graphs and sessions, which are available through TensorFlow 2.x compat.v1.
"""

import tensorflow.compat.v1 as tf


tf.disable_eager_execution()
