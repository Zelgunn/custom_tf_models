import tensorflow as tf
from abc import abstractmethod

from custom_tf_models.energy_based import EnergyStateFunction


class ApplyOnRandomInput(EnergyStateFunction):
    def __init__(self,
                 is_low_energy: bool,
                 ):
        super(ApplyOnRandomInput, self).__init__(is_low_energy=is_low_energy)

    def call(self, inputs):
        outputs = []
        index = tf.random.uniform(shape=[], minval=0, maxval=len(inputs), dtype=tf.int32)
        for i in range(len(inputs)):
            x = inputs[i]
            if i == index:
                output = self.apply_on_one(x)
            else:
                output = self.apply_on_others(x)
            outputs.append(output)
        return outputs

    @abstractmethod
    def apply_on_one(self, input_tensor):
        raise NotImplementedError

    @abstractmethod
    def apply_on_others(self, input_tensor):
        raise NotImplementedError
