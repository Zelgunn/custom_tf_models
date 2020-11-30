import tensorflow as tf
from tensorflow.python.keras import Model
from typing import Dict

from custom_tf_models.basic.AE import AE
from misc_utils.math_utils import reduce_mean_from


# AEP : Autoencoder-Predictor
class AEP(AE):
    def __init__(self,
                 encoder: Model,
                 decoder: Model,
                 predictor: Model,
                 input_length: int,
                 use_temporal_loss=True,
                 **kwargs):
        super(AEP, self).__init__(encoder=encoder,
                                  decoder=decoder,
                                  **kwargs)
        self.predictor = predictor
        self.input_length = input_length
        self.use_temporal_loss = use_temporal_loss

    @tf.function
    def autoencode(self, inputs):
        encoded = self.encode(inputs)
        decoded = self.decode(encoded)
        predicted = self.predictor(encoded)
        temporal_mean = self.get_temporal_mean(inputs)
        outputs = tf.concat([decoded, predicted + temporal_mean], axis=1)
        return outputs

    @tf.function
    def encode(self, inputs):
        inputs = inputs[:, :self.input_length]
        return self.encoder(inputs)

    @tf.function
    def get_temporal_mean(self, inputs):
        inputs = inputs[:, self.input_length:]
        return tf.reduce_mean(inputs, axis=1, keepdims=True)

    @tf.function
    def get_reconstruction_loss(self, inputs, outputs):
        return reduce_mean_from(tf.square(inputs - outputs), start_axis=2)

    @tf.function
    def compute_loss(self,
                     inputs
                     ) -> Dict[str, tf.Tensor]:
        outputs = self(inputs)

        encoded = self.encode(inputs)
        decoded = self.decode(encoded)
        predicted = self.predictor(encoded)

        decoder_target = inputs[:, :self.input_length]
        predictor_target = inputs[:, self.input_length:]

        decoder_loss = self.get_reconstruction_loss(decoder_target, decoded)
        predictor_loss = self.get_reconstruction_loss(predictor_target, predicted)
        loss = tf.concat([decoder_loss, predictor_loss], axis=1)

        if self.use_temporal_loss:
            output_length = tf.shape(outputs)[1] - self.input_length
            weights = get_temporal_loss_weights(self.input_length, output_length)
            loss = tf.reduce_mean(loss * weights)
        else:
            loss = tf.reduce_mean(loss)

        decoder_loss = tf.reduce_mean(decoder_loss)
        predictor_loss = tf.reduce_mean(predictor_loss)

        return {"loss": loss, "decoder_loss": decoder_loss, "predictor_loss": predictor_loss}

    def get_config(self):
        config = {
            "encoder": self.encoder.get_config(),
            "decoder": self.decoder.get_config(),
            "predictor": self.predictor.get_config(),
            "input_length": self.input_length,
            "use_temporal_loss": self.use_temporal_loss
        }
        if self.optimizer is not None:
            config["optimizer"] = self.optimizer.get_config()

        return config


@tf.function
def get_temporal_loss_weights(input_length, output_length, start=1.0, stop=0.1):
    reconstruction_loss_weights = tf.ones([input_length], dtype=tf.float32)

    input_length = tf.cast(input_length, tf.float32)
    output_length = tf.cast(output_length, tf.float32)
    step = (stop - start) / output_length

    prediction_loss_weights = tf.range(start=start, limit=stop, delta=step, dtype=tf.float32)

    loss_weights = tf.concat([reconstruction_loss_weights, prediction_loss_weights], axis=0)
    loss_weights *= (input_length + output_length) / tf.reduce_sum(loss_weights)
    loss_weights = tf.expand_dims(loss_weights, axis=0)
    return loss_weights
