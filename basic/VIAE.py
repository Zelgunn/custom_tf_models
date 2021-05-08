import tensorflow as tf
from tensorflow.python.keras import Model
from typing import Dict

from custom_tf_models.basic.IAE import IAE
from misc_utils.math_utils import reduce_mean_from


# VIAE : Variational Interpolating Autoencoder
class VIAE(IAE):
    def __init__(self,
                 encoder: Model,
                 decoder: Model,
                 step_size: int,
                 use_stochastic_loss: bool = False,
                 kl_divergence_lambda=1e-2,
                 **kwargs):
        super(VIAE, self).__init__(encoder=encoder,
                                   decoder=decoder,
                                   step_size=step_size,
                                   use_stochastic_loss=use_stochastic_loss,
                                   **kwargs)

        self.kl_divergence_lambda = kl_divergence_lambda
        self._kl_divergence_lambda = tf.constant(value=kl_divergence_lambda, dtype=tf.float32,
                                                 name="kl_divergence_lambda")

        self.epsilon_optimizer = tf.keras.optimizers.Adam()

    def autoencode(self, inputs: tf.Tensor, deterministic=True):
        if inputs.shape[1] == self.step_size:
            return self.autoencode_one(inputs)
        else:
            return self.autoencode_sequence(inputs)

    @tf.function
    def autoencode_sequence(self, inputs, deterministic=True):
        inputs, inputs_shape, new_shape = self.split_inputs(inputs, merge_batch_and_steps=True)
        decoded = self.autoencode_one(inputs, deterministic=deterministic)
        decoded = tf.reshape(decoded, inputs_shape)
        return decoded

    @tf.function
    def autoencode_one(self, inputs, deterministic=True):
        encoded = self.encoder(inputs)
        latent_mean, latent_log_var = tf.split(encoded, num_or_size_splits=2, axis=-1)
        if deterministic:
            latent_code = latent_mean
        else:
            latent_code = self.sample_latent_distribution(latent_mean, latent_log_var)
        decoded = self.decoder(latent_code)
        return decoded

    @tf.function
    def sample_latent_distribution(self, latent_mean: tf.Tensor, latent_log_var: tf.Tensor) -> tf.Tensor:
        epsilon = tf.random.normal(shape=tf.shape(latent_mean))
        # noinspection PyTypeChecker
        latent_code = latent_mean + tf.exp(0.5 * latent_log_var) * epsilon
        return latent_code

    @tf.function
    def interpolate(self, inputs, deterministic=True):
        inputs_shape = tf.shape(inputs)
        encoded = self.get_interpolated_latent_code(inputs, merge_batch_and_steps=True)

        latent_mean, latent_log_var = tf.split(encoded, num_or_size_splits=2, axis=-1)
        if deterministic:
            latent_code = latent_mean
        else:
            latent_code = self.sample_latent_distribution(latent_mean, latent_log_var)
        decoded = self.decoder(latent_code)
        decoded = tf.reshape(decoded, inputs_shape)

        return decoded

    @tf.function
    def compute_kl_divergence(self, latent_mean: tf.Tensor, latent_log_var: tf.Tensor) -> tf.Tensor:
        kl_divergence = tf.square(latent_mean) + tf.exp(latent_log_var) - tf.constant(1.0) - latent_log_var
        kl_divergence = 0.5 * tf.reduce_mean(kl_divergence)
        return kl_divergence

    @tf.function
    def compute_deterministic_loss(self, inputs) -> Dict[str, tf.Tensor]:
        target = inputs

        inputs_shape = tf.shape(inputs)
        # inputs = tf.clip_by_value(inputs + tf.random.normal(inputs_shape, stddev=0.01), 0.0, 1.0)

        encoded = self.get_interpolated_latent_code(inputs, merge_batch_and_steps=True)
        latent_mean, latent_log_var = tf.split(encoded, num_or_size_splits=2, axis=-1)
        latent_code = self.sample_latent_distribution(latent_mean, latent_log_var)

        decoded = self.decoder(latent_code)
        decoded = tf.reshape(decoded, inputs_shape)

        reconstruction_error = self.compute_reconstruction_loss(target, decoded)
        interpolation_error = self.compute_interpolation_error(target, decoded)
        kl_divergence = self.compute_kl_divergence(latent_mean, latent_log_var)

        loss = reconstruction_error + kl_divergence * self._kl_divergence_lambda

        return {
            "loss": loss,
            "reconstruction_error": reconstruction_error,
            "interpolation_error": interpolation_error,
            "kl_divergence": kl_divergence,
        }

    # @tf.function
    def compute_stochastic_loss(self, inputs) -> Dict[str, tf.Tensor]:
        raise NotImplementedError

    @tf.function
    def autoencode_sampling_epsilon(self, inputs: tf.Tensor, n_steps: tf.Tensor = 64):
        batch_size, total_length, channels = inputs.shape
        step_count = total_length // self.step_size

        # region Encode mu, sigma² = e(x)
        # encoded = self.get_interpolated_latent_code(inputs, merge_batch_and_steps=True)
        encoded = self.encoder(self.split_inputs(inputs, merge_batch_and_steps=True)[0])
        encoded = tf.expand_dims(encoded, axis=0)
        latent_mean, latent_log_var = tf.split(encoded, num_or_size_splits=2, axis=-1)
        # endregion

        # region Sample p(z|x)
        code_length, code_size = latent_log_var.shape[2:]
        # noinspection PyTypeChecker
        epsilon_shape = [n_steps - 1, 1, code_length, code_size]
        epsilon = tf.random.normal(epsilon_shape)
        epsilon = tf.concat([tf.zeros([1, 1, code_length, code_size]), epsilon], axis=0)

        latent_mean = tf.clip_by_value(latent_mean, -1e7, 1e7)
        latent_log_var = tf.minimum(latent_log_var, 8e1)
        latent_code = latent_mean + tf.exp(0.5 * latent_log_var) * epsilon

        latent_code = tf.reshape(latent_code, [n_steps * batch_size * step_count, code_length, code_size])
        # endregion

        # region Decode
        decoded = self.decoder(latent_code)
        decoded = tf.reshape(decoded, [n_steps, batch_size, total_length, channels])
        # endregion

        # region Select best epsilon (lowest MAE)
        inputs = tf.expand_dims(inputs, axis=0)
        mae = reduce_mean_from(tf.abs(inputs - decoded), start_axis=2)

        best_indices = tf.argmin(mae, axis=0, output_type=tf.int32)
        batch_range = tf.range(batch_size, dtype=tf.int32)
        gather_indices = tf.stack([best_indices, batch_range], axis=-1)

        decoded = tf.gather_nd(decoded, gather_indices)
        # endregion

        return decoded

    def get_config(self):
        config = {
            "encoder": self.encoder.get_config(),
            "decoder": self.decoder.get_config(),
            "step_size": self.step_size,
            "use_stochastic_loss": self.use_stochastic_loss,
            "kl_divergence_lambda": self.kl_divergence_lambda,
            # "noise_std": 0.01
        }

        if self.optimizer is not None:
            config["optimizer"] = self.optimizer.get_config()

        return config