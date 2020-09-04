import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2
from typing import Tuple, Dict, Any

from misc_utils.math_utils import lerp
from custom_tf_models import IAE
from custom_tf_models.adversarial import GANLoss, GANLossMode, gradient_penalty


class IAEGAN(IAE):
    def __init__(self,
                 encoder: Model,
                 decoder: Model,
                 discriminator: Model,
                 step_size: int,
                 autoencoder_learning_rate=1e-3,
                 discriminator_learning_rate=1e-3,
                 reconstruction_loss_weight=1e3,
                 wgan=True,
                 seed=None,
                 ):
        super(IAEGAN, self).__init__(encoder=encoder,
                                     decoder=decoder,
                                     step_size=step_size,
                                     learning_rate=autoencoder_learning_rate,
                                     seed=seed)
        self.discriminator = discriminator
        self.discriminator_learning_rate = discriminator_learning_rate
        self.discriminator.optimizer = tf.keras.optimizers.Adam(learning_rate=self.discriminator_learning_rate)
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self._reconstruction_loss_weight = tf.constant(value=reconstruction_loss_weight, dtype=tf.float32,
                                                       name="reconstruction_loss_weight")
        self.wgan = wgan

    @tf.function
    def train_step(self, inputs):
        losses, gradients = self.compute_gradients(inputs)
        autoencoder_gradients, disc_gradients = gradients

        self.optimizer.apply_gradients(zip(autoencoder_gradients, self.autoencoder_trainable_variables))
        self.discriminator.optimizer.apply_gradients(zip(disc_gradients, self.discriminator_trainable_variables))

        (
            reconstruction_loss,
            generator_adversarial_loss,
            discriminator_adversarial_loss,
            gradient_penalty_loss
        ) = losses

        return {
            "reconstruction_loss": reconstruction_loss,
            "generator_adversarial_loss": generator_adversarial_loss,
            "discriminator_adversarial_loss": discriminator_adversarial_loss,
            "gradient_penalty_loss": gradient_penalty_loss,
        }

    @tf.function
    def compute_gradients(self, inputs):
        with tf.GradientTape() as autoencoder_tape, \
                tf.GradientTape() as discriminator_tape:
            losses = self.compute_loss(inputs)
            (
                reconstruction_loss,
                generator_adversarial_loss,
                discriminator_adversarial_loss,
                gradient_penalty_loss
            ) = losses

            autoencoder_loss = reconstruction_loss + generator_adversarial_loss / self._reconstruction_loss_weight

            if self.wgan:
                gradient_penalty_weight = tf.constant(10.0)
            else:
                gradient_penalty_weight = tf.constant(1.0)
            discriminator_loss = discriminator_adversarial_loss + gradient_penalty_loss * gradient_penalty_weight
            discriminator_loss /= self._reconstruction_loss_weight

        autoencoder_gradients = autoencoder_tape.gradient(autoencoder_loss, self.autoencoder_trainable_variables)
        disc_gradients = discriminator_tape.gradient(discriminator_loss, self.discriminator_trainable_variables)

        return losses, (autoencoder_gradients, disc_gradients)

    @tf.function
    def compute_loss(self,
                     inputs
                     ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        step_count = tf.shape(inputs)[1]

        start = inputs[:, :self.step_size]
        end = inputs[:, -self.step_size:]

        start_encoded = self.encode(start)
        end_encoded = self.encode(end)

        # region Reconstruction Loss
        max_offset = step_count - self.step_size
        offset = tf.random.uniform(shape=[], minval=0, maxval=max_offset + 1, dtype=tf.int32, seed=self.seed)
        step = inputs[:, offset:offset + self.step_size]

        factor = tf.cast(offset / max_offset, tf.float32)
        latent_code = lerp(start_encoded, end_encoded, factor)
        step_interpolated = self.decoder(latent_code)
        interpolation_loss = tf.reduce_mean(tf.square(step - step_interpolated))

        # step_decoded = self.decode(self.encode(step))
        # reconstruction_loss = tf.reduce_mean(tf.square(step - step_decoded))

        reconstruction_loss = interpolation_loss
        # reconstruction_loss = (interpolation_loss + reconstruction_loss) * 0.5
        # endregion

        # region Adversarial Loss
        factor = tf.random.uniform(shape=[], minval=-0.5, maxval=1.5, dtype=tf.float32, seed=self.seed)
        latent_code = lerp(start_encoded, end_encoded, factor)
        generated = self.decode(latent_code)

        real = step
        fake = generated

        real_discriminated = self.discriminator(real)
        fake_discriminated = self.discriminator(fake)

        if self.wgan:
            real_logits = tf.reduce_mean(real_discriminated)
            fake_logits = tf.reduce_mean(fake_discriminated)

            generator_adversarial_loss = fake_logits
            discriminator_adversarial_loss = real_logits - fake_logits
        else:
            gan_loss = GANLoss(mode=GANLossMode.VANILLA)
            generator_adversarial_loss = gan_loss(fake_discriminated, is_real=True)
            discriminator_adversarial_loss = (
                    gan_loss(fake_discriminated, is_real=False) +
                    gan_loss(real_discriminated, is_real=True)
            )
        # endregion

        gradient_penalty_loss = self.gradient_penalty(real, fake)

        return reconstruction_loss, generator_adversarial_loss, discriminator_adversarial_loss, gradient_penalty_loss

    @tf.function
    def gradient_penalty(self, real, fake) -> tf.Tensor:
        return gradient_penalty(real=real, fake=fake, discriminator=self.discriminator, seed=self.seed)

    # region select_two_latent_codes (from interpolated latent code)
    @tf.function
    def select_two_latent_codes(self, latent_code):
        step_count = tf.shape(latent_code)[1]

        def select_center():
            return latent_code[:, 1:-1]

        def select_random():
            return self.select_two_random_latent_codes(latent_code)

        selected = tf.cond(step_count == 4,
                           true_fn=select_center,
                           false_fn=select_random)

        return selected

    @tf.function
    def select_two_random_latent_codes(self, latent_code):
        code_shape = tf.shape(latent_code)
        batch_size = code_shape[0]
        step_count = code_shape[1]

        batch_range = tf.range(batch_size, dtype=tf.int32)
        batch_range = tf.reshape(batch_range, [batch_size, 1, 1])
        batch_range = tf.tile(batch_range, [1, 2, 1])

        first_indices = tf.random.uniform([batch_size], minval=1, maxval=step_count - 1, dtype=tf.int32, seed=self.seed)
        second_indices = tf.random.uniform([batch_size], minval=1, maxval=step_count - 1,
                                           dtype=tf.int32, seed=self.seed)
        second_indices = self.select_next_if_same(first_indices, second_indices, step_count)

        indices = tf.stack([first_indices, second_indices], axis=-1)
        indices = tf.expand_dims(indices, axis=-1)
        batch_indices = tf.concat([batch_range, indices], axis=-1)

        return tf.gather_nd(latent_code, batch_indices)

    @tf.function
    def select_next_if_same(self, first_indices, second_indices, step_count):
        def elem_compare(indices):
            def move_if_last():
                return tf.cond(indices[0] == step_count - 2,
                               true_fn=lambda: 1,
                               false_fn=lambda: indices[0] + 1)

            return tf.cond(indices[0] == indices[1],
                           true_fn=move_if_last,
                           false_fn=lambda: indices[1])

        return tf.map_fn(fn=elem_compare, elems=tf.stack([first_indices, second_indices], axis=-1))

    # endregion

    @tf.function
    def discriminate(self, inputs):
        batch_size = tf.shape(inputs)[0]
        inputs = self.split_inputs(inputs, merge_batch_and_steps=True)
        result = self.discriminator(inputs)
        result = tf.reshape(result, [batch_size, -1])
        return result

    @property
    def autoencoder_trainable_variables(self):
        return self.encoder.trainable_variables + self.decoder.trainable_variables

    @property
    def discriminator_trainable_variables(self):
        return self.discriminator.trainable_variables

    @property
    def models_ids(self) -> Dict[Model, str]:
        return {
            **super(IAEGAN, self).models_ids,
            self.discriminator: "discriminator"
        }

    @property
    def optimizers_ids(self) -> Dict[OptimizerV2, str]:
        return {
            self.optimizer: "autoencoder_optimizer",
            self.discriminator.optimizer: "discriminator_optimizer",
        }

    @property
    def metrics_names(self):
        return ["reconstruction", "generator_adversarial", "discriminator_loss", "gradient_penalty"]

    def get_config(self) -> Dict[str, Any]:
        base_config = super(IAEGAN, self).get_config()
        config = {
            **base_config,
            "autoencoder_learning_rate": self.learning_rate,
            "discriminator_learning_rate": self.discriminator_learning_rate,
            "reconstruction_loss_weight": self.reconstruction_loss_weight,
            "wgan": self.wgan,
            "seed": self.seed,
        }
        return config
