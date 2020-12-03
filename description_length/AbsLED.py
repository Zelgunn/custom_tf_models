import tensorflow as tf
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Conv1D, Reshape, TimeDistributed, Layer
from tensorflow.python.ops.init_ops import VarianceScaling
import numpy as np
from typing import Dict, Any, Tuple, Union, Callable

from custom_tf_models.basic.AE import AE
from CustomKerasLayers import TileLayer
from misc_utils.math_utils import reduce_mean_from
from misc_utils.general import expand_dims_to_rank


class DecaySchedule(object):
    def __init__(self,
                 initial_rate: float,
                 decay_steps: int,
                 decay_rate: float,
                 staircase: bool,
                 offset: float):
        self.initial_rate = initial_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase
        self.offset = offset

    def __call__(self, step) -> tf.Tensor:
        return self.call(step)

    @tf.function
    def call(self, step: tf.Tensor) -> tf.Tensor:
        decay_steps = tf.cast(self.decay_steps, tf.float32)
        step = tf.cast(step, tf.float32)
        step = step / decay_steps

        if self.staircase:
            step = tf.math.floor(step)

        decay = tf.pow(self.decay_rate, step)
        goal = self.offset + decay * self.initial_rate
        return goal

    def get_config(self):
        return {
            "initial_rate": self.initial_rate,
            "decay_steps": self.decay_steps,
            "decay_rate": self.decay_rate,
            "staircase": self.staircase,
            "offset": self.offset
        }


class AbsLED(Model):
    def __init__(self,
                 description_module: Model,
                 description_loss_lambda=1e-2,
                 unmasked_pretext_lambda=1e0,
                 energy_margin=1e0,
                 pretext_margin_schedule: Callable[[tf.Tensor], tf.Tensor] = None,
                 first_stage_model: Union[Model, Callable] = None,
                 second_stage_model: Union[Model, Callable] = None,
                 pretext_loss: Callable[[tf.Tensor, tf.Tensor], tf.Tensor] = None,
                 **kwargs):
        super(AbsLED, self).__init__(**kwargs)
        self.description_module = description_module
        self.description_loss_lambda = description_loss_lambda
        self.unmasked_pretext_lambda = unmasked_pretext_lambda
        self.energy_margin = energy_margin
        self.pretext_margin_schedule = pretext_margin_schedule

        # region Optional callables
        self.first_stage_model = first_stage_model
        self.second_stage_model = second_stage_model
        self.pretext_loss = pretext_loss
        # endregion

        # region Constants
        self._description_loss_lambda = tf.constant(description_loss_lambda, dtype=tf.float32,
                                                    name="description_loss_lambda")
        self._unmasked_pretext_lambda = tf.constant(unmasked_pretext_lambda, dtype=tf.float32,
                                                    name="unmasked_pretext_lambda")
        self._energy_margin = tf.constant(energy_margin, dtype=tf.float32, name="energy_margin")
        # endregion

    # region Inference
    @tf.function
    def compute_description_module_inputs(self, inputs: tf.Tensor):
        if self.first_stage_model is not None:
            return self.first_stage_model(inputs)
        else:
            raise NotImplementedError("You either need to provide a `first_stage_model` to this model, or implement"
                                      "this function in a subclass.")

    @tf.function
    def compute_description_mask(self, description_energy: tf.Tensor):
        # assumes all(description_energy <= 0)
        activation_probability = tf.exp(description_energy)
        # so that all(0 < activation_probability <= 1)
        activation_noise = tf.random.uniform(shape=tf.shape(activation_probability), minval=0.0, maxval=1.0)
        activated = activation_probability >= activation_noise
        binarization_noise = tf.cast(activated, tf.float32) - tf.stop_gradient(activation_probability)
        description_mask = activation_probability + binarization_noise
        return description_mask

    @tf.function
    def compute_description(self, description_module_inputs: tf.Tensor):
        description_energy = self.description_module(description_module_inputs)
        description_mask = self.compute_description_mask(description_energy)
        description = description_energy * description_mask
        return description

    @tf.function
    def compute_outputs_from_description(self, description: tf.Tensor):
        if self.second_stage_model is not None:
            return self.second_stage_model(description)
        else:
            raise NotImplementedError("You either need to provide a `second_stage_model` to this model, or implement"
                                      "this function in a subclass.")

    def call(self, inputs, training=None, mask=None):
        description_module_inputs = self.compute_description_module_inputs(inputs)
        description = self.compute_description(description_module_inputs)
        outputs = self.compute_outputs_from_description(description)
        return outputs

    # endregion

    # region Training
    @property
    def use_pretext_margin_schedule(self) -> bool:
        return self.pretext_margin_schedule is not None

    @tf.function
    def compute_description_dynamic_weight(self, pretext_loss) -> tf.Tensor:
        if not self.use_pretext_margin_schedule:
            return tf.ones(shape=[], dtype=tf.float32, name="pretext_margin_weight")

        margin = self.pretext_margin_schedule(self.train_step_index)
        margin_weight = (margin - tf.stop_gradient(pretext_loss)) / margin
        min_weight = -1.0 if self.allow_negative_description_loss_weight else 0.0
        description_dynamic_weight = tf.clip_by_value(margin_weight * self.margin_delta_factor, min_weight, 1.0)
        return description_dynamic_weight

    @tf.function
    def compute_description_loss(self, description_energy: tf.Tensor) -> tf.Tensor:
        description_energy_loss = tf.nn.relu(self._energy_margin + description_energy)
        description_energy_loss = tf.reduce_mean(description_energy_loss)
        return description_energy_loss

    @tf.function
    def compute_pretext_loss(self, target, predicted):
        if self.pretext_loss is not None:
            return self.pretext_loss(target, predicted)
        else:
            raise NotImplementedError("You either need to provide a `pretext_loss` to this model, or implement"
                                      "this function in a subclass.")

    @property
    def perform_unmasked_pretext_task(self) -> bool:
        return self.unmasked_pretext_lambda > 0.0

    def compute_loss(self, inputs, target) -> Dict[str, tf.Tensor]:
        # region Forward
        description_module_inputs = self.compute_description_module_inputs(inputs)
        description_energy = self.description_module(description_module_inputs)
        description_mask = self.compute_description_mask(description_energy)
        description = description_energy * description_mask
        outputs = self.compute_outputs_from_description(description)
        # endregion

        # region Loss
        pretext_loss = self.compute_pretext_loss(target, outputs)

        description_dynamic_weight = self.compute_description_dynamic_weight(pretext_loss)
        description_loss = self.compute_description_loss(description_energy)
        description_loss *= self._description_loss_lambda * description_dynamic_weight

        if self.perform_unmasked_pretext_task:
            unmasked_description = description_module_inputs
            outputs = self.compute_outputs_from_description(unmasked_description)
            unmasked_pretext_loss = self.compute_pretext_loss(target, outputs)
            total_pretext_weight = tf.ones([], dtype=tf.float32) + self._unmasked_pretext_lambda
            pretext_loss = (pretext_loss + unmasked_pretext_loss * self._unmasked_pretext_lambda) / total_pretext_weight
        else:
            unmasked_pretext_loss = None
        # endregion

        loss = pretext_loss + description_loss

        # region Metrics
        pretext_metrics = {"pretext/error": pretext_loss}
        if self.use_pretext_margin_schedule:
            pretext_margin = self.pretext_margin_schedule(self.train_step_index)
            pretext_margin_delta = pretext_loss - pretext_margin
            pretext_metrics["pretext/margin"] = pretext_margin
            pretext_metrics["pretext/margin_delta"] = pretext_margin_delta
        if self.perform_unmasked_pretext_task:
            pretext_metrics["pretext/unmasked_loss"] = unmasked_pretext_loss

        description_energy = tf.reduce_mean(description_energy)
        description_length = tf.reduce_mean(description_mask)
        description_metrics = {
            "description/energy": description_energy,
            "description/dynamic_weight": description_dynamic_weight,
            "description/length": description_length,
        }

        metrics = {
            "loss": loss,
            **pretext_metrics,
            **description_metrics,
        }
        # endregion
        return metrics

    def train_step(self, data):
        raise NotImplementedError

    # endregion

    # region Anomaly detection

    @tf.function
    def compute_description_energy(self, inputs: tf.Tensor):
        description_module_inputs = self.get_description_module_inputs(inputs)
        description_energy = self.description_module(description_module_inputs)
        return description_energy

    @property
    def additional_test_metrics(self):
        return [self.compute_description_energy]

    # endregion

    def get_config(self):
        raise NotImplementedError
