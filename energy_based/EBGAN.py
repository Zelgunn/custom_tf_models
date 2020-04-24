# EBGAN : Energy-based Generative Adversarial Network
import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2
from typing import Tuple, Dict

from custom_tf_models import CustomModel, AE


class EBGAN(CustomModel):
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

    def call(self, inputs, training=None, mask=None):
        return self.autoencoder(inputs)

    @tf.function
    def train_step(self, inputs, *args, **kwargs):
        with tf.GradientTape() as discriminator_tape, tf.GradientTape() as generator_tape:
            losses = self.compute_loss(inputs)
            (
                discriminator_real_loss,
                discriminator_fake_loss,
                generator_adversarial_loss
            ) = losses

            discriminator_loss = discriminator_real_loss + discriminator_fake_loss

        disc_gradients = discriminator_tape.gradient(discriminator_loss, self.autoencoder.trainable_variables)
        generator_gradients = generator_tape.gradient(generator_adversarial_loss, self.generator.trainable_variables)

        self.autoencoder.optimizer.apply_gradients(zip(disc_gradients, self.autoencoder.trainable_variables))
        self.generator.optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))

        return discriminator_real_loss, discriminator_fake_loss, generator_adversarial_loss

    @tf.function
    def compute_loss(self,
                     inputs
                     ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        batch_size = tf.shape(inputs)[0]
        generated = self.generate(batch_size)

        real_discriminated = self.discriminate(inputs)
        fake_discriminated = self.discriminate(generated)

        discriminator_real_loss = real_discriminated
        discriminator_fake_loss = tf.nn.relu(self.margin - fake_discriminated)
        generator_adversarial_loss = fake_discriminated

        return discriminator_real_loss, discriminator_fake_loss, generator_adversarial_loss

    @tf.function
    def generate(self, batch_size):
        code_shape = self.generator.input_shape[1:]
        code_shape = [batch_size, *code_shape]
        code = tf.random.normal(shape=code_shape, mean=0.0, stddev=1.0, seed=self.seed)
        generated = self.generator(code)
        return generated

    @tf.function
    def discriminate(self, inputs):
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

    @property
    def models_ids(self) -> Dict[Model, str]:
        return {
            **self.autoencoder.models_ids,
            self.generator: "generator"
        }

    @property
    def optimizers_ids(self) -> Dict[OptimizerV2, str]:
        return {
            self.autoencoder.optimizer: "autoencoder_optimizer",
            self.generator.optimizer: "generator_optimizer",
        }

    @property
    def metrics_names(self):
        return ["discriminator_real", "discriminator_fake", "generator_adversarial"]
