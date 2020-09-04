import tensorflow as tf

from custom_tf_models.energy_based import EnergyStateFunction


class FlipSequence(EnergyStateFunction):
    def __init__(self):
        super(FlipSequence, self).__init__(is_low_energy=False)

    def call(self, inputs):
        return tf.reverse(inputs, axis=(1,))
