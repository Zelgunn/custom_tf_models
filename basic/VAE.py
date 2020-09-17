import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.keras import Model
from typing import Dict

from custom_tf_models.basic.AE import AE


# VAE : Variational Autoencoder
class VAE(AE):
    def __init__(self,
                 encoder: Model,
                 decoder: Model,
                 kl_divergence_loss_factor=1e-2,
                 seed=None,
                 **kwargs):
        super(VAE, self).__init__(encoder=encoder,
                                  decoder=decoder,
                                  **kwargs)
        self.kl_divergence_loss_factor = kl_divergence_loss_factor
        self.seed = seed

    @tf.function
    def encode(self, inputs):
        return self.sample_latent_distribution(inputs)

    def get_latent_distribution(self, inputs) -> tfp.distributions.MultivariateNormalDiag:
        encoder_outputs = self.encoder(inputs)
        latent_mean, latent_variance = tf.split(encoder_outputs, num_or_size_splits=2, axis=-1)
        latent_distribution = tfp.distributions.MultivariateNormalDiag(loc=latent_mean, scale_diag=latent_variance)
        return latent_distribution

    @tf.function
    def sample_latent_distribution(self, inputs) -> tf.Tensor:
        return self.get_latent_distribution(inputs).sample()

    @tf.function
    def compute_loss(self,
                     inputs
                     ) -> Dict[str, tf.Tensor]:
        latent_distribution = self.get_latent_distribution(inputs)
        reference_distribution = get_reference_distribution(latent_distribution)
        latent_code = latent_distribution.sample()
        decoded = self.decode(latent_code)

        reconstruction_loss = tf.reduce_mean(tf.square(inputs - decoded))
        kl_divergence = tf.reduce_mean(tfp.distributions.kl_divergence(latent_distribution, reference_distribution))
        kl_divergence *= self.kl_divergence_loss_factor
        loss = reconstruction_loss + kl_divergence

        return {"loss": loss, "reconstruction": reconstruction_loss, "kl_divergence": kl_divergence}

    def get_config(self):
        return {
            **super(VAE, self).get_config(),
            "kl_divergence_factor": self.kl_divergence_loss_factor,
            "seed": self.seed,
        }


def get_reference_distribution(latent_distribution: tfp.distributions.MultivariateNormalDiag
                               ) -> tfp.distributions.MultivariateNormalDiag:
    event_size = latent_distribution.event_shape[-1]
    mean = tf.zeros(shape=[event_size])
    variance = tf.ones(shape=[event_size])
    return tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=variance)
