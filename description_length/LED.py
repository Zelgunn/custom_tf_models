import tensorflow as tf
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Conv1D, Reshape, TimeDistributed, Activation
from tensorflow.python.ops.init_ops import VarianceScaling
import numpy as np
from typing import Dict, Any, Tuple

from custom_tf_models.basic.AE import AE
from CustomKerasLayers import TileLayer
from misc_utils.math_utils import binarize, reduce_mean_from
from misc_utils.general import expand_dims_to_rank


# from transformers.transformer import TransformerEncoder

# TODO : Custom class for threshold (with type : constant, exp. decay, ...)
# TODO : Experiment (HParam) - Constant threshold
# TODO : Experiment (HParam) - Small model + Denoise mode
# TODO : Experiment (Model) - Progressively lower energy - Soft
# TODO : Experiment (Model) - Progressively lower energy - Hard (would enforce a mask like : [1] * n + [0] * (N-n) )
# TODO : Experiment (Struct) - Replace conv. by square matrices in desc. model


# LED : Low Energy Descriptors
class LED(AE):
    def __init__(self,
                 encoder: Model,
                 decoder: Model,
                 features_per_block: int,
                 merge_dims_with_features=False,
                 descriptors_activation="tanh",
                 binarization_temperature=50.0,
                 add_binarization_noise_to_mask=True,
                 description_energy_loss_lambda=1e-2,
                 use_noise=True,
                 noise_stddev=0.1,
                 reconstruct_noise=False,
                 seed=None,
                 **kwargs
                 ):
        super(LED, self).__init__(encoder=encoder,
                                  decoder=decoder,
                                  **kwargs)
        self.features_per_block = features_per_block
        self.merge_dims_with_features = merge_dims_with_features
        self.descriptors_activation = descriptors_activation
        self.binarization_temperature = binarization_temperature
        self.add_binarization_noise_to_mask = add_binarization_noise_to_mask
        self.description_energy_loss_lambda = description_energy_loss_lambda
        self.use_noise = use_noise
        self.noise_stddev = noise_stddev
        self.reconstruct_noise = reconstruct_noise
        self.seed = seed

        self.description_energy_model = self._make_description_model()

        self._binarization_threshold = tf.constant(0.0, dtype=tf.float32, name="bin_threshold")
        self._binarization_temperature = tf.constant(binarization_temperature, dtype=tf.float32, name="bin_temperature")
        self._description_energy_loss_lambda = tf.constant(description_energy_loss_lambda, dtype=tf.float32,
                                                           name="description_energy_loss_lambda")

        self.noise_factor_distribution = "normal"
        self.noise_type = "dense"

        self.train_step_index = tf.Variable(initial_value=0, trainable=False, name="train_step_index", dtype=tf.int32)
        self.rec_threshold_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.1,
                                                                                     decay_steps=1000,
                                                                                     decay_rate=0.6,
                                                                                     staircase=False)
        self.rec_threshold_offset = tf.constant(0.035, dtype=tf.float32, name="rec_threshold_offset")

    # region Make description model
    def _make_description_model(self):
        return self.make_description_model(latent_code_shape=self.encoder.output_shape[1:],
                                           features_per_block=self.features_per_block,
                                           merge_dims_with_features=self.merge_dims_with_features,
                                           descriptors_activation=self.descriptors_activation,
                                           seed=self.seed)

    @staticmethod
    def make_description_model(latent_code_shape,
                               features_per_block: int,
                               merge_dims_with_features: bool,
                               descriptors_activation: str,
                               name="DescriptionEnergyModel",
                               seed=None):
        features_dims = latent_code_shape if merge_dims_with_features else latent_code_shape[-1:]
        block_count = np.prod(features_dims) // features_per_block

        if descriptors_activation not in LED.descriptor_activations_map():
            valid_activations = tuple(LED.descriptor_activations_map().keys())
            raise ValueError("`output_activation` must be in ({}). Got {}."
                             .format(valid_activations, descriptors_activation))

        kernel_size = 13  # Current field size : (13 - 1) * 5 + 1 = 61
        kernel_initializer = VarianceScaling(seed=seed, scale=1.0)
        shared_params = {
            "kernel_initializer": kernel_initializer,
            "kernel_size": kernel_size,
            "padding": "causal",
            "activation": "relu",
        }
        conv_layers = [
            Conv1D(filters=32, **shared_params),
            Conv1D(filters=16, **shared_params),
            Conv1D(filters=8, **shared_params),
            Conv1D(filters=4, **shared_params),
            # TransformerEncoder(embedding_layer=None, layers_intermediate_size=16, layers_count=2,
            #                    attention_heads_count=4, attention_key_size=4, attention_values_size=4,
            #                    dropout_rate=0.0, return_attention=False, kernel_initializer=kernel_initializer),
        ]
        last_conv_layer = Conv1D(filters=1,
                                 activation=descriptors_activation,
                                 kernel_initializer=kernel_initializer,
                                 kernel_size=kernel_size,
                                 padding="causal")
        conv_layers.append(last_conv_layer)

        if merge_dims_with_features:
            target_input_shape = (block_count, features_per_block)
            tiling_multiples = [1, features_per_block]
        else:
            conv_layers = [TimeDistributed(layer) for layer in conv_layers]
            dimensions_size = np.prod(latent_code_shape[:-1])
            target_input_shape = (dimensions_size, block_count, features_per_block)
            tiling_multiples = [1, 1, features_per_block]

        description_energy_model = Sequential(layers=[
            Reshape(target_shape=target_input_shape, input_shape=latent_code_shape),
            *conv_layers,
            TileLayer(multiples=tiling_multiples),
            Reshape(target_shape=latent_code_shape)
        ], name=name)

        return description_energy_model

    @staticmethod
    def descriptor_activations_map() -> Dict[str, float]:
        return {"tanh": 0.0, "sigmoid": 0.5, "linear": 0.0}

    # endregion

    # region Forward
    @tf.function
    def get_description_mask(self, description_energy: tf.Tensor) -> tf.Tensor:
        mask = binarize(description_energy,
                        threshold=self._binarization_threshold,
                        temperature=self._binarization_temperature,
                        add_noise=self.add_binarization_noise_to_mask)
        return mask

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
        encoded *= description_mask
        outputs = self.decoder(encoded)
        # endregion

        # region Loss
        reconstruction_loss = self.reconstruction_loss(target, outputs)
        description_energy_loss = self.description_energy_loss(description_energy)

        reconstruction_goal = self.get_reconstruction_loss_goal()
        description_energy_loss_weight = self.get_description_energy_loss_weight(reconstruction_loss)

        loss = reconstruction_loss + description_energy_loss_weight * description_energy_loss
        # endregion

        # region Metrics
        description_length = tf.reduce_mean(tf.stop_gradient(description_mask))
        reconstruction_goal_delta = reconstruction_loss - reconstruction_goal

        metrics = {
            "loss": loss,
            "reconstruction/error": reconstruction_loss,
            "reconstruction/goal": reconstruction_goal,
            "reconstruction/goal_delta": reconstruction_goal_delta,
            "description/energy": description_energy_loss,
            "description/loss_weight": description_energy_loss_weight,
            "description/length": description_length,
        }
        # endregion

        return metrics

    def train_step(self, inputs) -> Dict[str, tf.Tensor]:
        metrics = super(LED, self).train_step(inputs)
        self.train_step_index.assign_add(1)
        return metrics

    # region Objectives weights

    @tf.function
    def get_reconstruction_loss_goal(self) -> tf.Tensor:
        return self.rec_threshold_schedule(self.train_step_index) + self.rec_threshold_offset

    @tf.function
    def get_description_energy_loss_weight(self, reconstruction_loss) -> tf.Tensor:
        goal = self.get_reconstruction_loss_goal()
        reconstruction_loss = tf.stop_gradient(reconstruction_loss)
        goal_weight = (goal - reconstruction_loss) / goal
        goal_weight = tf.clip_by_value(goal_weight * 4.0, -1.0, 1.0)
        # goal_weight = 1.0 - tf.minimum((reconstruction_loss - goal) / goal, 1.0)
        return self._description_energy_loss_lambda * goal_weight

    # endregion

    # region Gradient penalty
    @tf.function
    def compute_description_gradient_penalty(self, inputs: tf.Tensor, noise_factor: tf.Tensor) -> tf.Tensor:
        with tf.GradientTape() as tape:
            encoded = self.encoder(inputs)
            description_energy = self.description_energy_model(encoded)
            description_energy_loss = self.description_energy_loss(description_energy)

        trained_variables = self.encoder.trainable_variables + self.description_energy_model.trainable_variables
        gradients = tape.gradient(description_energy_loss, trained_variables)

        gradients = [tensor for tensor in gradients if tensor is not None]
        gradient_penalty = tf.linalg.global_norm(gradients)
        gradient_penalty = tf.nn.relu(gradient_penalty - 1.0)
        # gradient_penalty = [tf.sqrt(tf.reduce_sum(tf.square(tensor))) for tensor in gradients]
        # gradient_penalty = [tf.square(tf.nn.relu(tensor - tf.constant(1.0))) for tensor in gradient_penalty]
        # gradient_penalty = tf.reduce_mean(gradient_penalty)
        return gradient_penalty

    # endregion

    # region Training noise
    @tf.function
    def add_training_noise(self, inputs) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        batch_size = tf.shape(inputs)[0]
        if self.use_noise:
            noise_factor = tf.random.uniform([batch_size], maxval=1.0, seed=self.seed)
            noise = tf.random.normal(tf.shape(inputs), stddev=self.noise_stddev, seed=self.seed)
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
        return tf.reduce_mean(tf.square(inputs - outputs))

    @tf.function
    def description_energy_loss(self, description_energy: tf.Tensor) -> tf.Tensor:
        description_energy = tf.reduce_mean(description_energy)
        return description_energy

    # endregion

    # region Config
    def get_config(self) -> Dict[str, Any]:
        base_config = super(LED, self).get_config()
        return {
            **base_config,
            "description_energy_model": self.description_energy_model.get_config(),
            "features_per_block": self.features_per_block,
            "merge_dims_with_features": self.merge_dims_with_features,
            "binarization_temperature": self.binarization_temperature,
            "add_binarization_noise_to_mask": self.add_binarization_noise_to_mask,
            "description_energy_loss_lambda": self.description_energy_loss_lambda,
            "use_noise": self.use_noise,
            "noise_stddev": self.noise_stddev,
            "reconstruct_noise": self.reconstruct_noise,
            "rec_threshold_schedule": self.rec_threshold_schedule.get_config(),
            "rec_threshold_offset": self.rec_threshold_offset.numpy(),
            "seed": self.seed,
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
        # return [self.compute_description_energy, self.compute_total_energy, self.compute_total_energy_x3]
        return [self.compute_description_energy]
    # endregion
