import tensorflow as tf
from typing import Tuple

from custom_tf_models.energy_based import ApplyOnRandomInput


class NoisyInput(ApplyOnRandomInput):
    def __init__(self,
                 seed,
                 gaussian_noise_stddev: float,
                 output_range: Tuple[float, float] = None):
        super(NoisyInput, self).__init__(is_low_energy=True,
                                         ground_truth_from_inputs=False,
                                         seed=seed)
        self.gaussian_noise_stddev = gaussian_noise_stddev
        self.output_range = output_range

    def apply_on_one(self, input_tensor):
        noise: tf.Tensor = tf.random.normal(tf.shape(input_tensor), stddev=self.gaussian_noise_stddev, seed=self.seed)
        result = input_tensor + noise
        if self.output_range:
            result = tf.clip_by_value(result, self.output_min, self.output_max)
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
