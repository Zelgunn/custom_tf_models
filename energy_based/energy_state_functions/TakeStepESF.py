import tensorflow as tf

from custom_tf_models.energy_based import EnergyStateFunction


class TakeStepESF(EnergyStateFunction):
    def __init__(self,
                 step_count: int,
                 axis=1,
                 ):
        super(TakeStepESF, self).__init__(is_low_energy=True,
                                          ground_truth_from_inputs=True)
        self.step_count = step_count
        self.axis = axis

    def call(self, inputs):
        multiple_inputs = isinstance(inputs, (tuple, list))

        if multiple_inputs:
            state = self.take_inputs_steps(inputs)
        else:
            state = self.take_step(inputs)

        return state

    @tf.function
    def take_inputs_steps(self, inputs):
        outputs = []
        for x in inputs:
            x = self.take_step(x)
            outputs.append(x)
        return outputs

    @tf.function
    def take_step(self, inputs: tf.Tensor):
        step_size = tf.shape(inputs)[self.axis] // self.step_count
        return inputs[:, :step_size]
