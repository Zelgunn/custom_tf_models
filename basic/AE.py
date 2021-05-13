import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.callbacks import CallbackList
from tensorflow.python.keras.losses import mean_absolute_error
from typing import Dict, Callable

from misc_utils.train_utils import CustomLearningRateSchedule


# AE : (Vanilla) Autoencoder
class AE(Model):
    def __init__(self,
                 encoder: Model,
                 decoder: Model,
                 reconstruction_loss_function: Callable = mean_absolute_error,
                 **kwargs):
        super(AE, self).__init__(**kwargs)

        self.encoder = encoder
        self.decoder = decoder
        self.reconstruction_loss_function = reconstruction_loss_function

        self.train_step_index = tf.Variable(initial_value=0, trainable=False, name="train_step_index", dtype=tf.int32)

    @tf.function
    def call(self, inputs, training=None, mask=None):
        return self.autoencode(inputs)

    @tf.function
    def encode(self, inputs):
        return self.encoder(inputs)

    @tf.function
    def decode(self, inputs):
        return self.decoder(inputs)

    @tf.function
    def autoencode(self, inputs):
        return self.decode(self.encode(inputs))

    @tf.function
    def train_step(self, data) -> Dict[str, tf.Tensor]:
        with tf.GradientTape() as tape:
            metrics = self.compute_loss(data)
            loss = metrics["loss"]

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.train_step_index.assign_add(1)
        self._train_counter.assign(tf.cast(self.train_step_index, tf.int64))

        return metrics

    @tf.function
    def test_step(self, data):
        return self.compute_loss(data)

    @tf.function
    def compute_loss(self,
                     inputs
                     ) -> Dict[str, tf.Tensor]:
        outputs = self(inputs)
        loss = self.compute_reconstruction_loss(inputs, outputs)
        return {"loss": loss}

    @tf.function
    def compute_reconstruction_loss(self, inputs, outputs, axis=None) -> tf.Tensor:
        loss = self.reconstruction_loss_function(inputs, outputs)
        loss = tf.reduce_mean(loss, axis=axis)
        return loss

    def on_train_begin(self, callbacks: CallbackList, initial_epoch: int, steps_per_epoch: int):
        super(AE, self).on_train_begin(callbacks, initial_epoch, steps_per_epoch)
        if isinstance(self.learning_rate, CustomLearningRateSchedule):
            self.learning_rate.step_offset = initial_epoch * steps_per_epoch

    def compute_encoded_shape(self, input_shape):
        return self.encoder.compute_output_shape(input_shape)

    def compute_output_signature(self, input_signature):
        return input_signature

    def get_config(self):
        config = {
            "encoder": self.encoder.get_config(),
            "decoder": self.decoder.get_config(),
            "reconstruction_loss_function": self.reconstruction_loss_function.__name__
        }
        if self.optimizer is not None:
            config["optimizer"] = self.optimizer.get_config()

        return config
