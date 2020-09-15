# AE : Autoencoder
import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2
from tensorflow.python.keras.callbacks import CallbackList
from typing import Dict

from custom_tf_models.utils import LearningRateType
from misc_utils.train_utils import CustomLearningRateSchedule


class AE(Model):
    def __init__(self,
                 encoder: Model,
                 decoder: Model,
                 learning_rate: LearningRateType = None,
                 **kwargs):
        super(AE, self).__init__(**kwargs)

        self.encoder = encoder
        self.decoder = decoder
        self.learning_rate = learning_rate

        self.optimizer = None
        if self.learning_rate is not None:
            self.set_optimizer(tf.keras.optimizers.Adam(learning_rate=learning_rate))

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

        return metrics

    @tf.function
    def test_step(self, data):
        return self.compute_loss(data)

    @tf.function
    def compute_loss(self,
                     inputs
                     ) -> Dict[str, tf.Tensor]:
        outputs = self(inputs)
        loss = tf.reduce_mean(tf.square(inputs - outputs))
        return {"loss": loss}

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
            "learning_rate": self.learning_rate
        }
        return config

    @property
    def optimizers_ids(self) -> Dict[OptimizerV2, str]:
        return {
            self.optimizer: "optimizer"
        }

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
        self.encoder.optimizer = optimizer
        self.decoder.optimizer = optimizer
