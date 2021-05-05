import tensorflow as tf
from tensorflow.python.keras import Model
from typing import Dict, Tuple

from custom_tf_models.basic.AE import AE


# VAE : Variational Autoencoder
class VAE(AE):
    def __init__(self,
                 encoder: Model,
                 decoder: Model,
                 kl_divergence_loss_factor=1e-2,
                 **kwargs):
        super(VAE, self).__init__(encoder=encoder,
                                  decoder=decoder,
                                  **kwargs)
        self.kl_divergence_loss_factor = kl_divergence_loss_factor

    @tf.function
    def get_latent_distribution(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        encoder_outputs = self.encoder(inputs)
        latent_mean, latent_log_var = tf.split(encoder_outputs, num_or_size_splits=2, axis=-1)
        return latent_mean, latent_log_var

    @tf.function
    def sample_distribution(self, latent_mean: tf.Tensor, latent_log_var: tf.Tensor):
        epsilon = tf.random.normal(shape=tf.shape(latent_mean))
        # noinspection PyTypeChecker
        return latent_mean + tf.exp(0.5 * latent_log_var) * epsilon

    @tf.function
    def encode(self, inputs: tf.Tensor, deterministic=True):
        latent_mean, latent_log_var = self.get_latent_distribution(inputs)
        if deterministic:
            latent_code = latent_mean
        else:
            latent_code = self.sample_distribution(latent_mean, latent_log_var)
        return latent_code

    def autoencode(self, inputs: tf.Tensor, deterministic=True):
        return self.decode(self.encode(inputs, deterministic=deterministic))

    @tf.function
    def compute_loss(self,
                     inputs
                     ) -> Dict[str, tf.Tensor]:
        latent_mean, latent_log_var = self.get_latent_distribution(inputs)
        latent_code = self.sample_distribution(latent_mean, latent_log_var)
        decoded = self.decode(latent_code)

        reconstruction_loss = tf.reduce_mean(tf.square(inputs - decoded))
        kl_divergence = self.compute_kl_divergence(latent_mean, latent_log_var)
        # noinspection PyTypeChecker
        loss = reconstruction_loss + kl_divergence * self.kl_divergence_loss_factor

        return {"loss": loss, "reconstruction": reconstruction_loss, "kl_divergence": kl_divergence}

    @tf.function
    def compute_kl_divergence(self, latent_mean: tf.Tensor, latent_log_var: tf.Tensor) -> tf.Tensor:
        kl_divergence = tf.square(latent_mean) + tf.exp(latent_log_var) - tf.constant(1.0) - latent_log_var
        kl_divergence = 0.5 * tf.reduce_mean(kl_divergence)
        return kl_divergence

    def get_config(self):
        return {
            **super(VAE, self).get_config(),
            "kl_divergence_factor": self.kl_divergence_loss_factor,
        }
