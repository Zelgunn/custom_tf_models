import tensorflow as tf

from energy_based import ApplyOnRandomInput


class DropInput(ApplyOnRandomInput):
    def apply_on_one(self, input_tensor):
        return tf.zeros_like(input_tensor)

    def apply_on_others(self, input_tensor):
        return input_tensor
