import tensorflow as tf
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Conv1D, Reshape, TimeDistributed, Layer
from tensorflow.python.ops.init_ops import VarianceScaling
import numpy as np
from typing import Dict, Any, Tuple

from custom_tf_models.basic.AE import AE
from CustomKerasLayers import TileLayer
from misc_utils.math_utils import reduce_mean_from
from misc_utils.general import expand_dims_to_rank


# TODO : Experiment (HParam) - MAE for training (instead of MSE)

class LEDGoal(object):
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


class InvReLU(Layer):
    def call(self, inputs, **kwargs):
        return -tf.nn.relu(inputs)

    def get_config(self):
        return {}

    def compute_output_shape(self, input_shape):
        return input_shape


# LED : Low Energy Descriptors
class LED(AE):
    def __init__(self,
                 encoder: Model,
                 decoder: Model,
                 features_per_block: int,
                 merge_dims_with_features=False,
                 description_energy_loss_lambda=1e-2,
                 use_noise=True,
                 noise_stddev=0.1,
                 reconstruct_noise=False,
                 goal_schedule: LEDGoal = None,
                 allow_negative_description_loss_weight=True,
                 goal_delta_factor=4.0,
                 unmasked_reconstruction_weight=1.0,
                 energy_margin=5.0,
                 **kwargs
                 ):
        super(LED, self).__init__(encoder=encoder,
                                  decoder=decoder,
                                  **kwargs)
        self.features_per_block = features_per_block
        self.merge_dims_with_features = merge_dims_with_features
        self.description_energy_loss_lambda = description_energy_loss_lambda

        # region Input noise parameters
        self.use_noise = use_noise
        self.noise_stddev = noise_stddev
        self.reconstruct_noise = reconstruct_noise

        self.noise_factor_distribution = "normal"
        self.noise_type = "dense"
        # endregion

        self.description_energy_model = self._make_description_model()

        # region Goal schedule
        self.goal_schedule = goal_schedule
        self.train_step_index = tf.Variable(initial_value=0, trainable=False, name="train_step_index", dtype=tf.int32)

        self.allow_negative_description_loss_weight = allow_negative_description_loss_weight
        self.goal_delta_factor = goal_delta_factor
        # endregion

        self.unmasked_reconstruction_weight = unmasked_reconstruction_weight
        self.energy_margin = energy_margin

        # region Tensorflow constants
        self._description_energy_loss_lambda = tf.constant(description_energy_loss_lambda, dtype=tf.float32,
                                                           name="description_energy_loss_lambda")
        self._unmasked_reconstruction_weight = tf.constant(unmasked_reconstruction_weight, dtype=tf.float32,
                                                           name="unmasked_reconstruction_weight")
        self._energy_margin = tf.constant(energy_margin, dtype=tf.float32, name="energy_margin")
        # endregion

    # region Make description model
    def _make_description_model(self):
        return self.make_description_model(latent_code_shape=self.encoder.output_shape[1:],
                                           features_per_block=self.features_per_block,
                                           merge_dims_with_features=self.merge_dims_with_features)

    @staticmethod
    def make_description_model(latent_code_shape,
                               features_per_block: int,
                               merge_dims_with_features: bool,
                               name="DescriptionEnergyModel"):
        features_dims = latent_code_shape if merge_dims_with_features else latent_code_shape[-1:]
        block_count = np.prod(features_dims) // features_per_block

        # region Core

        kernel_size = 13  # Current field size : (13 - 1) * 5 + 1 = 61
        kernel_initializer = VarianceScaling()
        shared_params = {
            "kernel_initializer": kernel_initializer,
            "kernel_size": kernel_size,
            "padding": "causal",
            "activation": "relu",
        }
        layers = [
            Conv1D(filters=32, **shared_params),
            Conv1D(filters=32, **shared_params),
            Conv1D(filters=32, **shared_params),
            Conv1D(filters=32, **shared_params),
            Conv1D(filters=1, activation="relu", kernel_initializer=kernel_initializer,
                   kernel_size=kernel_size, padding="causal", name="DescriptionModuleOutput", use_bias=True)
        ]

        # shared_params = {
        #     "head_count": 4,
        #     "kernel_size": kernel_size,
        #     "activation": "relu",
        #     "use_bias": True,
        #     "kernel_initializer": kernel_initializer,
        # }
        # layers = [
        #     StandAloneSelfAttention1D(head_size=32, **shared_params),
        #     StandAloneSelfAttention1D(head_size=16, **shared_params),
        #     StandAloneSelfAttention1D(head_size=8, **shared_params),
        #     StandAloneSelfAttention1D(head_size=4, **shared_params),
        #     Conv1D(filters=1, kernel_size=kernel_size, use_bias=True,
        #            kernel_initializer=kernel_initializer, padding="causal",
        #            activation=descriptors_activation, name="DescriptionModuleOutput")
        # ]
        # endregion

        # region Reshape / Tile
        if merge_dims_with_features:
            target_input_shape = (block_count, features_per_block)
        else:
            layers = [TimeDistributed(layer, name="{}Distributed".format(layer.name)) for layer in layers]
            dimensions_size = np.prod(latent_code_shape[:-1])
            target_input_shape = (dimensions_size, block_count, features_per_block)

        if features_per_block != 1:
            tiling_multiples = [1, features_per_block] if merge_dims_with_features else [1, 1, features_per_block]
            layers.append(TileLayer(tiling_multiples))

        input_reshape = Reshape(target_shape=target_input_shape, input_shape=latent_code_shape)
        output_reshape = Reshape(target_shape=latent_code_shape)
        output_activation = InvReLU()
        layers = [input_reshape, *layers, output_reshape, output_activation]
        # endregion

        return Sequential(layers=layers, name=name)

    # endregion

    # region Forward
    """
        This function returns a random mask, where the probability of activation depends on the `description_energy`
        parameter.
        params:
            `description_energy` : Assumed to be a tensor of only negative values (or zeros).
        returns:
            A tensor, with same shape and dtype as `description_energy` 
    """

    @tf.function
    def get_description_mask(self, description_energy: tf.Tensor) -> tf.Tensor:
        # assumes all(description_energy <= 0)
        activation_probability = tf.exp(description_energy)
        # so that all(0 < activation_probability <= 1)
        activation_noise = tf.random.uniform(shape=tf.shape(activation_probability), minval=0.0, maxval=1.0)
        activated = activation_probability >= activation_noise
        binarization_noise = tf.cast(activated, tf.float32) - tf.stop_gradient(activation_probability)
        description_mask = activation_probability + binarization_noise
        return description_mask

    @tf.function
    def encode(self, inputs: tf.Tensor) -> tf.Tensor:
        encoded = self.encoder(inputs)
        description_energy = self.description_energy_model(encoded)
        description_mask = self.get_description_mask(description_energy)
        encoded *= description_mask
        return encoded

    # endregion

    # region Loss
    @tf.function
    def compute_loss(self, inputs) -> Dict[str, tf.Tensor]:
        inputs, target, noise_factor = self.add_training_noise(inputs)

        # region Forward
        encoded = self.encoder(inputs)
        description_energy = self.description_energy_model(encoded)
        description_mask = self.get_description_mask(description_energy)
        if not self.perform_unmasked_reconstruction:
            mask_shape = tf.shape(description_mask)
            mask_override = tf.random.uniform(shape=mask_shape, minval=0.0, maxval=1.0, dtype=tf.float32) < 0.1
            description_mask = tf.where(mask_override, tf.ones_like(description_mask), description_mask)
        outputs = self.decoder(encoded * description_mask)
        # endregion

        # region Loss
        reconstruction_loss = self.reconstruction_loss(target, outputs)
        description_energy_loss = self.description_energy_loss(description_energy)
        description_energy_loss_weight = self.get_description_energy_loss_weight(reconstruction_loss)
        description_energy_loss *= description_energy_loss_weight
        loss = reconstruction_loss + description_energy_loss

        if self.perform_unmasked_reconstruction:
            unmasked_outputs = self.decoder(encoded)
            unmasked_reconstruction_error = self.reconstruction_loss(target, unmasked_outputs)
            loss += unmasked_reconstruction_error * self._unmasked_reconstruction_weight
        else:
            unmasked_reconstruction_error = None

        # endregion

        # region Metrics
        reconstruction_metrics = {"reconstruction/error": reconstruction_loss}

        if self.goal_schedule is not None:
            reconstruction_goal = self.goal_schedule(self.train_step_index)
            reconstruction_goal_delta = reconstruction_loss - reconstruction_goal
            reconstruction_metrics["reconstruction/goal"] = reconstruction_goal
            reconstruction_metrics["reconstruction/goal_delta"] = reconstruction_goal_delta

        if self.perform_unmasked_reconstruction:
            reconstruction_metrics["reconstruction/unmasked_error"] = unmasked_reconstruction_error

        description_energy = tf.reduce_mean(description_energy)
        description_length = tf.reduce_mean(description_mask)
        description_metrics = {
            "description/energy": description_energy,
            "description/loss_weight": description_energy_loss_weight,
            "description/length": description_length,
        }

        metrics = {
            "loss": loss,
            **reconstruction_metrics,
            **description_metrics,
        }
        # endregion

        return metrics

    def train_step(self, inputs) -> Dict[str, tf.Tensor]:
        metrics = super(LED, self).train_step(inputs)
        self.train_step_index.assign_add(1)
        self._train_counter.assign(tf.cast(self.train_step_index, tf.int64))
        return metrics

    # region Objectives weights
    @tf.function
    def get_description_energy_loss_weight(self, reconstruction_loss) -> tf.Tensor:
        if self.goal_schedule is None:
            return self._description_energy_loss_lambda

        goal = self.goal_schedule(self.train_step_index)
        goal_weight = (goal - tf.stop_gradient(reconstruction_loss)) / goal
        min_weight = -1.0 if self.allow_negative_description_loss_weight else 0.0
        goal_weight = tf.clip_by_value(goal_weight * self.goal_delta_factor, min_weight, 1.0)

        return self._description_energy_loss_lambda * goal_weight

    # endregion

    # region Training noise
    @tf.function
    def add_training_noise(self, inputs) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        batch_size = tf.shape(inputs)[0]
        if self.use_noise:
            noise_factor = tf.random.uniform([batch_size], maxval=1.0)
            noise = tf.random.normal(tf.shape(inputs), stddev=self.noise_stddev)
            noisy_inputs = inputs + noise * expand_dims_to_rank(noise_factor, inputs)

            target = noisy_inputs if self.reconstruct_noise else inputs
            inputs = noisy_inputs
        else:
            noise_factor = tf.zeros([batch_size], dtype=tf.float32)
            target = inputs

        return inputs, target, noise_factor

    # endregion

    @tf.function
    def reconstruction_loss(self, inputs: tf.Tensor, outputs: tf.Tensor) -> tf.Tensor:
        # return tf.reduce_mean(tf.square(inputs - outputs))
        return tf.reduce_mean(tf.abs(inputs - outputs))

    @tf.function
    def description_energy_loss(self, description_energy: tf.Tensor) -> tf.Tensor:
        description_energy_loss = tf.nn.relu(self._energy_margin + description_energy)
        description_energy_loss = tf.reduce_mean(description_energy_loss)
        return description_energy_loss

    # endregion

    # region Config
    def get_config(self) -> Dict[str, Any]:
        base_config = super(LED, self).get_config()
        return {
            **base_config,
            "description_energy_model": self.description_energy_model.get_config(),
            "features_per_block": self.features_per_block,
            "merge_dims_with_features": self.merge_dims_with_features,
            "description_energy_loss_lambda": self.description_energy_loss_lambda,
            "use_noise": self.use_noise,
            "noise_stddev": self.noise_stddev,
            "reconstruct_noise": self.reconstruct_noise,
            "goal": self.goal_schedule.get_config() if self.goal_schedule is not None else None,
            "allow_negative_description_loss_weight": self.allow_negative_description_loss_weight,
            "goal_delta_factor": self.goal_delta_factor,
            "unmasked_reconstruction_weight": self.unmasked_reconstruction_weight,
            "energy_margin": self.energy_margin,
        }

    # endregion

    # region Anomaly detection
    @tf.function
    def compute_description_energy(self, inputs: tf.Tensor) -> tf.Tensor:
        encoded = self.encoder(inputs)
        energy = self.description_energy_model(encoded)
        return reduce_mean_from(energy, start_axis=1)

    @tf.function
    def compute_total_energy(self, inputs: tf.Tensor):
        encoded = self.encoder(inputs)
        description_energy = self.description_energy_model(encoded)
        description_mask = self.get_description_mask(description_energy)
        encoded *= description_mask
        outputs = self.decoder(encoded)
        reconstruction_error = tf.abs(inputs - outputs)

        description_energy = reduce_mean_from(description_energy, start_axis=1)
        reconstruction_error = reduce_mean_from(reconstruction_error, start_axis=1)
        total_energy = description_energy + reconstruction_error
        return total_energy

    @tf.function
    def compute_total_energy_x3(self, inputs: tf.Tensor):
        encoded = self.encoder(inputs)
        description_energy = self.description_energy_model(encoded)
        description_mask = self.get_description_mask(description_energy)
        encoded *= description_mask
        outputs = self.decoder(encoded)
        reconstruction_error = tf.abs(inputs - outputs)

        description_energy = reduce_mean_from(description_energy, start_axis=1)
        reconstruction_error = reduce_mean_from(reconstruction_error, start_axis=1) * tf.constant(3.0)
        total_energy = description_energy + reconstruction_error
        return total_energy

    @property
    def additional_test_metrics(self):
        return [self.compute_description_energy]

    # endregion

    @property
    def perform_unmasked_reconstruction(self) -> bool:
        return self.unmasked_reconstruction_weight > 0.0
