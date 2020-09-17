import tensorflow as tf
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Conv1D, Reshape, TimeDistributed
from tensorflow.python.keras.initializers.initializers_v2 import VarianceScaling
import numpy as np
from typing import Dict, Any

from custom_tf_models.basic.AE import AE
from custom_tf_models.utils import LearningRateType
from CustomKerasLayers import TileLayer
from misc_utils.math_utils import binarize, reduce_mean_from
from misc_utils.general import expand_dims_to_rank


# LED : Low Energy Descriptors
class LED(AE):
    def __init__(self,
                 encoder: Model,
                 decoder: Model,
                 learning_rate: LearningRateType,
                 features_per_block: int,
                 merge_dims_with_features=False,
                 descriptors_activation="tanh",
                 binarization_temperature=50.0,
                 add_binarization_noise_to_mask=True,
                 description_energy_loss_lambda=1e-2,
                 noise_stddev=0.1,
                 reconstruct_noise=False,
                 seed=None,
                 **kwargs
                 ):
        super(LED, self).__init__(encoder=encoder,
                                  decoder=decoder,
                                  learning_rate=learning_rate,
                                  **kwargs)
        self.features_per_block = features_per_block
        self.merge_dims_with_features = merge_dims_with_features
        self.descriptors_activation = descriptors_activation
        self.binarization_temperature = binarization_temperature
        self.add_binarization_noise_to_mask = add_binarization_noise_to_mask
        self.description_energy_loss_lambda = description_energy_loss_lambda
        self.noise_stddev = noise_stddev
        self.reconstruct_noise = reconstruct_noise
        self.seed = seed

        self.description_energy_model = self._make_description_model()

        self._binarization_threshold = tf.constant(0.0, dtype=tf.float32, name="bin_threshold")
        self._binarization_temperature = tf.constant(binarization_temperature, dtype=tf.float32, name="bin_temperature")
        self._description_energy_loss_lambda = tf.constant(description_energy_loss_lambda, dtype=tf.float32,
                                                           name="description_energy_loss_lambda")

        self.noise_factor_distribution = "normal"
        self.noise_type = "sparse"

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

        shared_params = {
            "kernel_initializer": VarianceScaling(seed=seed, scale=1.0),
            "kernel_size": 13,  # Current field size : (13 - 1) * 5 + 1 = 61
            "padding": "causal"
        }
        conv_layers = [
            Conv1D(filters=32, activation="relu", **shared_params),
            Conv1D(filters=16, activation="relu", **shared_params),
            Conv1D(filters=8, activation="relu", **shared_params),
            Conv1D(filters=4, activation="relu", **shared_params),
            Conv1D(filters=1, activation=descriptors_activation, **shared_params)
        ]

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
    def reconstruction_loss(self, inputs: tf.Tensor, outputs: tf.Tensor) -> tf.Tensor:
        return tf.reduce_mean(tf.square(inputs - outputs))

    @tf.function
    def description_energy_loss(self, description_energy: tf.Tensor, noise_factor: tf.Tensor) -> tf.Tensor:
        description_energy = reduce_mean_from(description_energy, start_axis=1)
        if self.reconstruct_noise:
            weights = tf.constant(1.0) - tf.square(noise_factor)
            description_energy = description_energy * weights
        description_energy = tf.reduce_mean(description_energy)
        return description_energy

    @tf.function
    def compute_loss(self, inputs) -> Dict[str, tf.Tensor]:
        # region Noisy inputs
        batch_size = tf.shape(inputs)[0]
        noise_factor = tf.random.uniform([batch_size], maxval=1.0, seed=self.seed)
        noise = tf.random.normal(tf.shape(inputs), stddev=self.noise_stddev, seed=self.seed)
        noisy_inputs = inputs + noise * expand_dims_to_rank(noise_factor, inputs)
        # endregion

        # region Forward
        encoded = self.encoder(noisy_inputs)
        description_energy = self.description_energy_model(encoded)
        description_mask = self.get_description_mask(description_energy)
        encoded *= description_mask
        outputs = self.decode(encoded)
        # endregion

        # region Loss
        reconstruction_target = noisy_inputs if self.reconstruct_noise else inputs
        reconstruction_loss = self.reconstruction_loss(reconstruction_target, outputs)
        description_energy_loss = self.description_energy_loss(description_energy, noise_factor)
        loss = reconstruction_loss + self._description_energy_loss_lambda * description_energy_loss
        # endregion

        # region Metrics
        description_length = tf.reduce_mean(tf.stop_gradient(description_mask))

        metrics = {
            "loss": loss,
            "reconstruction_loss": reconstruction_loss,
            "description_energy_loss": description_energy_loss,
            "description_length": description_length,
        }
        # endregion

        return metrics

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
            "noise_stddev": self.noise_stddev,
            "reconstruct_noise": self.reconstruct_noise,
            "seed": self.seed,
        }

    # endregion

    # region Anomaly detection
    @tf.function
    def compute_description_energy(self, inputs: tf.Tensor) -> tf.Tensor:
        encoded = self.encoder(inputs)
        energy = self.description_energy_model(encoded)
        return reduce_mean_from(energy, start_axis=1)

    @property
    def additional_test_metrics(self):
        return [self.compute_description_energy]
    # endregion
