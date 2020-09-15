# EBGAN : Energy-based Generative Adversarial Network
import tensorflow as tf
from tensorflow.python.keras import Model
from typing import Dict, Union

from custom_tf_models.basic.AE import AE


class EBGAN(Model):
    def __init__(self,
                 autoencoder: AE,
                 generator: Model,
                 margin: float,
                 generator_learning_rate=1e-3,
                 discriminator_learning_rate=1e-3,
                 seed=None,
                 ):
        super(EBGAN, self).__init__()

        self.autoencoder = autoencoder
        self.generator = generator
        self.margin = margin
        self.generator_learning_rate = generator_learning_rate
        self.discriminator_learning_rate = discriminator_learning_rate
        self.seed = seed

        self.autoencoder.set_optimizer(tf.keras.optimizers.Adam(self.generator_learning_rate))
        self.generator.optimizer = tf.keras.optimizers.Adam(self.discriminator_learning_rate)
        self._margin = tf.constant(margin, dtype=tf.float32, name="energy_margin")

    @tf.function
    def call(self, inputs, training=None, mask=None):
        return self.autoencoder(inputs)

    @tf.function
    def train_step(self, inputs) -> Dict[str, tf.Tensor]:
        with tf.GradientTape() as discriminator_tape, tf.GradientTape() as generator_tape:
            metrics = self.compute_loss(inputs)
            (
                discriminator_real_loss,
                discriminator_fake_loss,
                generator_adversarial_loss
            ) = metrics.values()

            discriminator_loss = discriminator_real_loss + discriminator_fake_loss

        disc_gradients = discriminator_tape.gradient(discriminator_loss, self.autoencoder.trainable_variables)
        generator_gradients = generator_tape.gradient(generator_adversarial_loss, self.generator.trainable_variables)

        self.autoencoder.optimizer.apply_gradients(zip(disc_gradients, self.autoencoder.trainable_variables))
        self.generator.optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))

        return metrics

    @tf.function
    def test_step(self, inputs) -> Dict[str, tf.Tensor]:
        return self.compute_loss(inputs)

    @tf.function
    def compute_loss(self,
                     inputs: tf.Tensor
                     ) -> Dict[str, tf.Tensor]:
        batch_size = tf.shape(inputs)[0]
        generated = self.generate(batch_size)

        real_discriminated = self.discriminate(inputs)
        fake_discriminated = self.discriminate(generated)

        discriminator_real_loss = real_discriminated
        discriminator_fake_loss = tf.nn.relu(self._margin - fake_discriminated)
        generator_adversarial_loss = fake_discriminated

        return {
            "discriminator_real_loss": discriminator_real_loss,
            "discriminator_fake_loss": discriminator_fake_loss,
            "generator_adversarial_loss": generator_adversarial_loss,
        }

    @tf.function
    def generate(self, batch_size: Union[int, tf.Tensor]) -> tf.Tensor:
        code_shape = self.generator.input_shape[1:]
        code_shape = [batch_size, *code_shape]
        code = tf.random.normal(shape=code_shape, mean=0.0, stddev=1.0, seed=self.seed)
        generated = self.generator(code)
        return generated

    @tf.function
    def discriminate(self, inputs: tf.Tensor) -> tf.Tensor:
        reconstructed = self.autoencoder(inputs)
        reconstruction_error = tf.square(inputs - reconstructed)
        reconstruction_error = tf.reduce_mean(reconstruction_error, axis=tuple(range(1, inputs.shape.rank)))
        return reconstruction_error

    def get_config(self):
        config = {
            "autoencoder": self.autoencoder.get_config(),
            "generator": self.generator.get_config(),
            "margin": self.margin,
            "autoencoder_learning_rate": self.discriminator_learning_rate,
            "generator_learning_rate": self.generator_learning_rate,
            "seed": self.seed,
        }
        return config
