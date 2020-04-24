# AEP : Autoencoder-Predictor

import tensorflow as tf
from tensorflow.python.keras import Model
from typing import Dict

from custom_tf_models import AE


class AEP(AE):
    def __init__(self,
                 encoder: Model,
                 decoder: Model,
                 predictor: Model,
                 input_length: int,
                 learning_rate=1e-3,
                 use_temporal_loss=True,
                 **kwargs):
        super(AEP, self).__init__(encoder=encoder,
                                  decoder=decoder,
                                  learning_rate=learning_rate,
                                  **kwargs)
        self.predictor = predictor
        self.input_length = input_length
        self.use_temporal_loss = use_temporal_loss

    def call(self, inputs, training=None, mask=None):
        encoded = self.encode(inputs)
        decoded = self.decode(encoded)
        predicted = self.predictor(encoded)
        outputs = tf.concat([decoded, predicted], axis=1)
        return outputs

    @tf.function
    def encode(self, inputs):
        inputs = inputs[:, :self.input_length]
        return self.encoder(inputs)

    @tf.function
    def compute_loss(self,
                     inputs
                     ) -> tf.Tensor:
        outputs = self(inputs)
        reconstruction_error = tf.reduce_mean(tf.square(inputs - outputs))
        if self.use_temporal_loss:
            output_length = tf.shape(outputs)[1] - self.input_length
            reconstruction_error *= get_temporal_loss_weights(self.input_length, output_length)
        return reconstruction_error

    def get_config(self):
        config = {
            "encoder": self.encoder.get_config(),
            "decoder": self.decoder.get_config(),
            "predictor": self.predictor.get_config(),
            "input_length": self.input_length,
            "learning_rate": self.learning_rate,
            "use_temporal_loss": self.use_temporal_loss
        }
        return config

    @property
    def models_ids(self) -> Dict[Model, str]:
        return {self.encoder: "encoder",
                self.decoder: "decoder",
                self.predictor: "predictor"}


@tf.function
def get_temporal_loss_weights(input_length, output_length, start=1.0, stop=0.1):
    reconstruction_loss_weights = tf.ones([input_length], dtype=tf.float32)
    step = (stop - start) / output_length
    prediction_loss_weights = tf.range(start=start, limit=stop, delta=step, dtype=tf.float32)
    loss_weights = tf.concat([reconstruction_loss_weights, prediction_loss_weights], axis=0)
    loss_weights *= (input_length + output_length) / tf.reduce_sum(loss_weights)
    loss_weights = tf.expand_dims(loss_weights, axis=0)
    return loss_weights
