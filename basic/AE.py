# AE : Autoencoder
import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2
from tensorflow.python.keras.callbacks import CallbackList
from typing import Dict

from custom_tf_models import CustomModel
from misc_utils.train_utils import CustomLearningRateSchedule


class AE(CustomModel):
    def __init__(self,
                 encoder: Model,
                 decoder: Model,
                 learning_rate=None,
                 **kwargs):
        super(AE, self).__init__(**kwargs)

        self.encoder = encoder
        self.decoder = decoder
        self.learning_rate = learning_rate

        self.optimizer = None
        if self.learning_rate is not None:
            self.set_optimizer(tf.keras.optimizers.Adam(learning_rate=learning_rate))

    def call(self, inputs, training=None, mask=None):
        return self.decode(self.encode(inputs))

    @tf.function
    def encode(self, inputs):
        return self.encoder(inputs)

    @tf.function
    def decode(self, inputs):
        return self.decoder(inputs)

    @property
    def metrics_names(self):
        return ["reconstruction"]

    @tf.function
    def train_step(self, inputs, *args, **kwargs):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(inputs)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss

    @tf.function
    def compute_loss(self,
                     inputs
                     ) -> tf.Tensor:
        outputs = self(inputs)
        return tf.reduce_mean(tf.square(inputs - outputs))

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
    def models_ids(self) -> Dict[Model, str]:
        return {
            self.encoder: "encoder",
            self.decoder: "decoder"
        }

    @property
    def optimizers_ids(self) -> Dict[OptimizerV2, str]:
        return {
            self.optimizer: "optimizer"
        }

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
        self.encoder.optimizer = optimizer
        self.decoder.optimizer = optimizer
