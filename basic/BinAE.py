import tensorflow as tf

from custom_tf_models import AE
from misc_utils.math_utils import binarize


class BinAE(AE):
    @tf.function
    def encode(self, inputs):
        encoded = self.encoder(inputs)
        bin_threshold = tf.constant(0.0)
        bin_temperature = tf.constant(50.0)
        encoded = binarize(encoded, threshold=bin_threshold, temperature=bin_temperature, add_noise=True)
        return encoded
