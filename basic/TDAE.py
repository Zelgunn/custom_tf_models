# TDAE : Time Distributed Autoencoder
import tensorflow as tf

from custom_tf_models import AE


class TDAE(AE):
    def encode(self, inputs):
        return apply_over_time(inputs, super(TDAE, self).encode)

    def decode(self, inputs):
        return apply_over_time(inputs, super(TDAE, self).decode)


def apply_over_time(inputs, function):
    batch_size, length, *input_dimensions = tf.unstack(tf.shape(inputs))
    inputs = tf.reshape(inputs, [batch_size * length, *input_dimensions])

    outputs = function(inputs)

    _, *output_dimensions = tf.unstack(tf.shape(outputs))
    outputs = tf.reshape(outputs, [batch_size, length, *output_dimensions])

    return outputs
