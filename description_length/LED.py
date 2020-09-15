import tensorflow as tf
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Conv1D, Reshape, TimeDistributed
from tensorflow.python.keras.initializers import VarianceScaling
import numpy as np
from typing import Dict, Any

from custom_tf_models import AE
from CustomKerasLayers import TileLayer
from misc_utils.math_utils import binarize, reduce_mean_from


# LED : Low Energy Descriptors
class LED(AE):
    def __init__(self,
                 encoder: Model,
                 decoder: Model,
                 learning_rate,
                 features_per_block: int,
                 merge_dims_with_features=False,
                 binarization_temperature=50.0,
                 add_binarization_noise_to_mask=False,
                 description_energy_loss_lambda=1e-2,
                 seed=None,
                 **kwargs
                 ):
        super(MinimalistDescriptorV4, self).__init__(encoder=encoder,
                                                     decoder=decoder,
                                                     learning_rate=learning_rate,
                                                     **kwargs)
        self.features_per_block = features_per_block
        self.merge_dims_with_features = merge_dims_with_features
        self.description_energy_model = self.make_description_model(latent_code_shape=encoder.output_shape[1:],
                                                                    features_per_block=features_per_block,
                                                                    merge_dims_with_features=merge_dims_with_features,
                                                                    seed=seed)
        self.binarization_temperature = binarization_temperature
        self.add_binarization_noise_to_mask = add_binarization_noise_to_mask
        self.description_energy_loss_lambda = description_energy_loss_lambda
        self.seed = seed

        self._binarization_threshold = tf.constant(0.0, dtype=tf.float32, name="bin_threshold")
        self._binarization_temperature = tf.constant(binarization_temperature, dtype=tf.float32, name="bin_temperature")
        self._description_energy_loss_lambda = tf.constant(description_energy_loss_lambda, dtype=tf.float32,
                                                           name="description_energy_loss_lambda")

        self.noise_factor_distribution = "normal"
        self.noise_type = "sparse"

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
    def description_energy_loss(self, description_energy: tf.Tensor) -> tf.Tensor:
        return tf.reduce_mean(description_energy)

    @tf.function
    def compute_loss(self,
                     inputs
                     ) -> Dict[str, tf.Tensor]:
        noise_factor = tf.random.uniform([], maxval=0.1)
        noise = tf.random.normal(tf.shape(inputs), stddev=1.0, seed=self.seed) * noise_factor
        inputs += noise

        encoded = self.encoder(inputs)
        description_energy = self.description_energy_model(encoded)
        description_mask = self.get_description_mask(description_energy)
        encoded *= description_mask
        outputs = self.decode(encoded)

        reconstruction_loss = self.reconstruction_loss(inputs, outputs)
        description_energy_loss = self.description_energy_loss(description_energy)
        loss = reconstruction_loss + self._description_energy_loss_lambda * description_energy_loss

        description_length = tf.reduce_mean(tf.stop_gradient(description_mask))

        metrics = {
            "loss": loss,
            "reconstruction_loss": reconstruction_loss,
            "description_energy_loss": description_energy_loss,
            "description_length": description_length,
        }

        return metrics

    # endregion

    def get_config(self) -> Dict[str, Any]:
        base_config = super(MinimalistDescriptorV4, self).get_config()
        return {
            **base_config,
            "description_energy_model": self.description_energy_model.get_config(),
            "features_per_block": self.features_per_block,
            "merge_dims_with_features": self.merge_dims_with_features,
            "binarization_temperature": self.binarization_temperature,
            "add_binarization_noise_to_mask": self.add_binarization_noise_to_mask,
            "seed": self.seed,
        }

    @staticmethod
    def make_description_model(latent_code_shape,
                               features_per_block: int,
                               merge_dims_with_features: bool,
                               name="DescriptionEnergyModel",
                               seed=None):
        features_dims = latent_code_shape if merge_dims_with_features else latent_code_shape[-1:]
        block_count = np.prod(features_dims) // features_per_block

        shared_params = {
            "kernel_initializer": VarianceScaling(seed=seed, scale=1.0),
            "kernel_size": 13,  # Current field size : 66
            "padding": "causal"
        }
        conv_layers = [
            Conv1D(filters=32, activation="relu", **shared_params),
            Conv1D(filters=16, activation="relu", **shared_params),
            Conv1D(filters=8, activation="relu", **shared_params),
            Conv1D(filters=4, activation="relu", **shared_params),
            Conv1D(filters=1, activation="tanh", **shared_params)
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
