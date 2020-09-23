import tensorflow as tf
from tensorflow.python.keras.models import Model
# noinspection PyUnresolvedReferences
from tensorflow.python.keras.initializers import VarianceScaling
import numpy as np
from typing import Dict, Any, Optional

from custom_tf_models import LED
from custom_tf_models.basic.AEP import get_temporal_loss_weights
from misc_utils.math_utils import reduce_mean_from


# PredLED : Predicting with Low Energy Descriptors
class PreLED(LED):
    def __init__(self,
                 encoder: Model,
                 decoder: Model,
                 predictor: Model,
                 input_length: int,
                 use_temporal_reconstruction_loss=True,
                 features_per_block=1,
                 merge_dims_with_features=False,
                 descriptors_activation="tanh",
                 binarization_temperature=50.0,
                 add_binarization_noise_to_mask=True,
                 description_energy_loss_lambda=1e-2,
                 use_noise=True,
                 noise_stddev=0.1,
                 # reconstruct_noise=False,
                 seed=None,
                 **kwargs
                 ):
        super(PreLED, self).__init__(encoder=encoder,
                                     decoder=decoder,
                                     features_per_block=features_per_block,
                                     merge_dims_with_features=merge_dims_with_features,
                                     descriptors_activation=descriptors_activation,
                                     binarization_temperature=binarization_temperature,
                                     add_binarization_noise_to_mask=add_binarization_noise_to_mask,
                                     description_energy_loss_lambda=description_energy_loss_lambda,
                                     use_noise=use_noise,
                                     noise_stddev=noise_stddev,
                                     reconstruct_noise=False,
                                     seed=seed,
                                     **kwargs)
        self.predictor = predictor
        self.input_length = input_length
        self.use_temporal_reconstruction_loss = use_temporal_reconstruction_loss

    # region Forward
    @tf.function
    def encode(self, inputs):
        inputs = inputs[:, :self.input_length]
        encoded = super(PreLED, self).encode(inputs)
        return encoded

    @tf.function
    def predict_next(self, inputs):
        encoded = self.encode(inputs)
        return self.predictor(encoded)

    @tf.function
    def decode_and_predict_next(self, encoded):
        decoded = self.decode(encoded)
        predicted = self.predictor(encoded)
        outputs = tf.concat([decoded, predicted], axis=1)
        return outputs

    @tf.function
    def autoencode_and_predict_next(self, inputs):
        encoded = self.encode(inputs)
        outputs = self.decode_and_predict_next(encoded)
        return outputs

    # endregion

    # region Loss
    @tf.function
    def compute_loss(self, inputs) -> Dict[str, tf.Tensor]:
        inputs, target, _ = self.add_training_noise(inputs)
        inputs = inputs[:, :self.input_length]

        # region Forward
        encoded = self.encoder(inputs)
        description_energy = self.description_energy_model(encoded)
        description_mask = self.get_description_mask(description_energy)
        encoded *= description_mask
        outputs = self.decode_and_predict_next(encoded)
        # endregion

        # region Loss
        reconstruction_loss = self.reconstruction_loss(target, outputs)
        description_energy_loss = self.description_energy_loss(description_energy)
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

    @tf.function
    def reconstruction_loss(self, inputs: tf.Tensor, outputs: tf.Tensor) -> tf.Tensor:
        if self.use_temporal_reconstruction_loss:
            loss = tf.square(inputs - outputs)
            loss = reduce_mean_from(loss, start_axis=2)
            output_length = tf.shape(outputs)[1] - self.input_length
            weights = get_temporal_loss_weights(self.input_length, output_length)
            loss = tf.reduce_mean(loss * weights)
        else:
            loss = super(PreLED, self).reconstruction_loss(inputs, outputs)
        return loss

    # endregion

    @tf.function
    def compute_description_energy(self, inputs: tf.Tensor) -> tf.Tensor:
        inputs = inputs[:, :self.input_length]
        return super(PreLED, self).compute_description_energy(inputs)

    def get_config(self) -> Dict[str, Any]:
        base_config = super(PreLED, self).get_config()
        config = {
            **base_config,
            "predictor": self.predictor.get_config(),
            "input_length": self.input_length,
            "use_temporal_reconstruction_loss": self.use_temporal_reconstruction_loss,
        }
        return config
