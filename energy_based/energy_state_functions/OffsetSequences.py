import tensorflow as tf

from energy_based import TakeStepESF, ApplyOnRandomInput


class OffsetSequences(TakeStepESF, ApplyOnRandomInput):
    def __init__(self, seed, step_count: int):
        super(OffsetSequences, self).__init__(step_count=step_count)
        self.is_low_energy = False
        self.seed = seed

    def call(self, inputs):
        inputs = ApplyOnRandomInput.call(self, inputs)
        return inputs

    def apply_on_one(self, input_tensor):
        return self.offset_sequence(input_tensor)

    def apply_on_others(self, input_tensor):
        return self.take_step(input_tensor)

    @tf.function
    def offset_sequence(self, sequence: tf.Tensor):
        input_shape = sequence.shape
        sequence_rank = input_shape.rank
        if self.axis < 0:
            axis = sequence_rank + self.axis
        else:
            axis = self.axis

        known_length = input_shape[axis]
        if known_length is None:
            length = tf.shape(sequence)[axis]
            known_step_size = None
        else:
            length = known_length
            known_step_size = length // self.step_count

        step_size = length // self.step_count

        min_offset = step_size // 2
        max_offset = length - step_size
        offset = tf.random.uniform(shape=[], minval=min_offset, maxval=max_offset + 1, dtype=tf.int32, seed=self.seed)

        begin = tf.pad([offset], paddings=[[axis, sequence_rank - axis - 1]], constant_values=0)
        end = tf.pad([step_size], paddings=[[axis, sequence_rank - axis - 1]], constant_values=-1)
        sequence = tf.slice(sequence, begin=begin, size=end)

        output_shape = [known_step_size if i == axis else input_shape[i] for i in range(sequence_rank)]
        sequence.set_shape(output_shape)

        return sequence
