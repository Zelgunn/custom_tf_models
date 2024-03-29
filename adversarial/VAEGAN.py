import tensorflow as tf
from tensorflow.python.keras import Model
from typing import Tuple, Dict

from custom_tf_models import VAE
from custom_tf_models.adversarial import GANLoss, GANLossMode


class VAEGAN(VAE):
    def __init__(self,
                 encoder: Model,
                 decoder: Model,
                 discriminator: Model,
                 discriminator_learning_rate=1e-4,
                 balance_discriminator_learning_rate=True,
                 reconstruction_loss_factor=100.0,
                 learned_reconstruction_loss_factor=1000.0,
                 kl_divergence_loss_factor=1.0,
                 **kwargs):
        super(VAEGAN, self).__init__(encoder=encoder,
                                     decoder=decoder,
                                     kl_divergence_loss_factor=kl_divergence_loss_factor,
                                     **kwargs)

        self.discriminator = discriminator

        self.discriminator_learning_rate = discriminator_learning_rate
        self.balance_discriminator_learning_rate = balance_discriminator_learning_rate

        self.reconstruction_loss_factor = reconstruction_loss_factor
        self.learned_reconstruction_loss_factor = learned_reconstruction_loss_factor

        self.generator_fake_loss = tf.Variable(initial_value=1.0)
        self.discriminator_fake_loss = tf.Variable(initial_value=1.0)

        self.discriminator_optimizer = tf.keras.optimizers.Adam(self._get_discriminator_learning_rate,
                                                                beta_1=0.5, beta_2=0.999)

    # region Training
    @property
    def metrics_names(self):
        return ["reconstruction", "learned_reconstruction", "kl_divergence",
                "decoder_fake", "generator_fake",
                "discriminator_real", "discriminator_fake"]

    @tf.function
    def train_step(self, inputs):
        with tf.GradientTape() as encoder_tape, \
                tf.GradientTape() as decoder_tape, \
                tf.GradientTape() as discriminator_tape:
            losses = self.compute_loss(inputs)
            (
                reconstruction_loss,
                learned_reconstruction_loss,
                kl_divergence,
                decoder_fake_loss,
                generator_fake_loss,
                discriminator_real_loss,
                discriminator_fake_loss
            ) = losses

            reconstruction_loss *= self.reconstruction_loss_factor
            learned_reconstruction_loss *= self.learned_reconstruction_loss_factor
            kl_divergence *= self.kl_divergence_loss_factor

            encoder_loss = reconstruction_loss + kl_divergence + learned_reconstruction_loss + decoder_fake_loss
            decoder_loss = reconstruction_loss + learned_reconstruction_loss + generator_fake_loss + decoder_fake_loss
            discriminator_loss = discriminator_fake_loss + discriminator_real_loss

        encoder_gradients = encoder_tape.gradient(encoder_loss, self.encoder.trainable_variables)
        decoder_gradients = decoder_tape.gradient(decoder_loss, self.decoder.trainable_variables)
        discriminator_gradients = discriminator_tape.gradient(discriminator_loss,
                                                              self.discriminator.trainable_variables)

        self.optimizer.apply_gradients(zip(encoder_gradients, self.encoder.trainable_variables))
        self.optimizer.apply_gradients(zip(decoder_gradients, self.decoder.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                         self.discriminator.trainable_variables))

        return losses

    @tf.function
    def compute_loss(self,
                     inputs
                     ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        latent_mean, latent_log_var = self.get_latent_distribution(inputs)
        latent_code = self.sample_distribution(latent_mean, latent_log_var)
        decoded = self.decoder(latent_code)

        noise = tf.random.normal(shape=latent_code.shape, mean=0.0, stddev=1.0)
        generated = self.decoder(noise)

        discriminated_inputs, inputs_high_level_features = self.discriminator(inputs)
        discriminated_decoded, decoded_high_level_features = self.discriminator(decoded)
        discriminated_generated, generated_high_level_features = self.discriminator(generated)

        reconstruction_loss = tf.reduce_mean(tf.square(decoded - inputs))

        learned_reconstruction_loss = tf.square(inputs_high_level_features - decoded_high_level_features)
        learned_reconstruction_loss = tf.reduce_mean(learned_reconstruction_loss)
        learned_reconstruction_loss = learned_reconstruction_loss

        kl_divergence = self.compute_kl_divergence(latent_mean, latent_log_var)

        gan_loss = GANLoss(mode=GANLossMode.VANILLA)
        decoder_fake_loss = gan_loss(discriminated_decoded, is_real=True)
        generator_fake_loss = gan_loss(discriminated_generated, is_real=True)
        discriminator_real_loss = gan_loss(discriminated_inputs, is_real=True)
        discriminator_fake_loss = gan_loss(discriminated_generated, is_real=False)

        if self.balance_discriminator_learning_rate:
            update_losses = [self.generator_fake_loss.assign(generator_fake_loss),
                             self.discriminator_fake_loss.assign(discriminator_fake_loss)]
            with tf.control_dependencies(update_losses):
                generator_fake_loss = tf.identity(generator_fake_loss)

        return (
            reconstruction_loss,
            learned_reconstruction_loss,
            kl_divergence,
            decoder_fake_loss,
            generator_fake_loss,
            discriminator_real_loss,
            discriminator_fake_loss
        )

    # endregion

    def _get_discriminator_learning_rate(self):
        if not self.balance_discriminator_learning_rate:
            return self.discriminator_learning_rate

        loss_difference = self.discriminator_fake_loss - self.generator_fake_loss
        ratio = tf.nn.sigmoid(loss_difference)
        return self.discriminator_learning_rate * ratio

    def get_config(self):
        config = {
            **super(VAEGAN, self).get_config(),
            "discriminator": self.discriminator.get_config(),
            "discriminator_learning_rate": self.discriminator_learning_rate,
            "balance_discriminator_learning_rate": self.balance_discriminator_learning_rate,
            "reconstruction_loss_factor": self.reconstruction_loss_factor,
            "learned_reconstruction_loss_factor": self.learned_reconstruction_loss_factor,
            "kl_divergence_loss_factor": self.kl_divergence_loss_factor,
        }
        return config

    @property
    def models_ids(self) -> Dict[Model, str]:
        return {self.encoder: "encoder",
                self.decoder: "decoder",
                self.discriminator: "discriminator"}
