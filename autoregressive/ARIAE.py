import tensorflow as tf
from tensorflow.python.keras import Model
from typing import Dict

from transformers import Transformer
from custom_tf_models import IAE


# ARIAE : Autoregressive Interpolating Autoencoder
class ARIAE(Model):
    def __init__(self,
                 iae: IAE,
                 transformer: Transformer,
                 **kwargs
                 ):
        super(ARIAE, self).__init__(**kwargs)

        self.iae = iae
        self.transformer = transformer

        self.iae.trainable = False

    @tf.function
    def train_step(self, inputs) -> Dict[str, tf.Tensor]:
        with tf.GradientTape() as tape:
            metrics = self.compute_loss(inputs)
            loss = metrics["loss"]

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return metrics

    @tf.function
    def test_step(self, inputs):
        return self.compute_loss(inputs)

    @tf.function
    def compute_loss(self, inputs) -> Dict[str, tf.Tensor]:
        pass
        # inputs, inputs_shape, split_shape = self.iae.split_inputs(inputs, merge_batch_and_steps=True)
        # encoded = self.iae.encode(inputs)

    def get_config(self):
        pass
