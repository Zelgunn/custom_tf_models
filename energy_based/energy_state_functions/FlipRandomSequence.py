import tensorflow as tf

from custom_tf_models.energy_based import ApplyOnRandomInput


class FlipRandomSequence(ApplyOnRandomInput):
    def __init__(self, seed):
        super(FlipRandomSequence, self).__init__(is_low_energy=False, seed=seed)

    def apply_on_one(self, input_tensor):
        return tf.reverse(input_tensor, axis=(1,))

    def apply_on_others(self, input_tensor):
        return input_tensor
