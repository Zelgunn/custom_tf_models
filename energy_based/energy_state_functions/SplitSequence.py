import tensorflow as tf
from typing import Tuple

from custom_tf_models.energy_based import ApplyOnRandomInput


class SplitSequence(ApplyOnRandomInput):
    def __init__(self,
                 gaussian_noise_stddev: float = None,
                 output_range: Tuple[float, float] = None
                 ):
        super(SplitSequence, self).__init__(is_low_energy=False)
        self.gaussian_noise_stddev = gaussian_noise_stddev
        self.output_range = output_range

    def apply_on_one(self, input_tensor):
        length = tf.shape(input_tensor)[1]
        half_length = length // 2

        split_start = tf.random.uniform(shape=[], minval=1, maxval=half_length - 1, dtype=tf.int32)
        split_end = split_start + half_length

        min_weight = 1.0 / (tf.cast(half_length, tf.float32) - 1.0)
        weights = tf.linspace(min_weight, 1.0 - min_weight, half_length)
        weights_shape = [1, half_length] + [1] * (len(input_tensor.shape) - 2)
        weights = tf.reshape(weights, weights_shape)

        start = input_tensor[:, split_start - 1]
        end = input_tensor[:, split_end + 1]
        start = tf.expand_dims(start, axis=1)
        end = tf.expand_dims(end, axis=1)
        delta = end - start
        interpolated = start + delta * weights

        if self.gaussian_noise_stddev:
            noise = tf.random.normal(shape=weights_shape, stddev=self.gaussian_noise_stddev)
            interpolated += noise

        if self.output_range:
            interpolated = tf.clip_by_value(interpolated, self.output_min, self.output_max)

        result = tf.concat([input_tensor[:, :split_start], interpolated, input_tensor[:, split_end:]], axis=1)
        return result

    def apply_on_others(self, input_tensor):
        return input_tensor

    @property
    def output_min(self) -> float:
        if not self.output_range:
            raise ValueError

        return self.output_range[0]

    @property
    def output_max(self) -> float:
        if not self.output_range:
            raise ValueError

        return self.output_range[1]
