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

    @tf.function
    def autoencode(self, inputs: tf.Tensor, deterministic=True):
        inputs = tf.convert_to_tensor(inputs)
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
    def _sample_latent_distribution(self,
                                    latent_mean: tf.Tensor,
                                    latent_log_var: tf.Tensor,
                                    epsilon: tf.Tensor,
                                    ) -> tf.Tensor:
        # noinspection PyTypeChecker
        latent_code = latent_mean + tf.exp(0.5 * latent_log_var) * epsilon
        return latent_code

    @tf.function
    def _sample_epsilon(self, latent_mean: tf.Tensor):
        epsilon = tf.random.normal(shape=tf.shape(latent_mean))
        return epsilon

    @tf.function
    def sample_latent_distribution(self, latent_mean: tf.Tensor, latent_log_var: tf.Tensor) -> tf.Tensor:
        epsilon = self._sample_epsilon(latent_mean)
        latent_code = self._sample_latent_distribution(latent_mean, latent_log_var, epsilon)
        return latent_code

    @tf.function
    def _split_encoded(self, encoded: tf.Tensor) -> tf.Tensor:
        return tf.split(encoded, num_or_size_splits=2, axis=-1)

    @tf.function
    def interpolate(self, inputs, take_mean=True):
        inputs, inputs_shape, new_shape = self.split_inputs(inputs, merge_batch_and_steps=False)
        step_count = new_shape[1]

        # 1) encode
        encoded_first = self.encoder(inputs[:, 0])
        encoded_last = self.encoder(inputs[:, -1])

        # 2) sample (or take mean)
        latent_mean_first, latent_log_var_first = self._split_encoded(encoded_first)
        latent_mean_last, latent_log_var_last = self._split_encoded(encoded_last)
        if take_mean:
            latent_code_first = latent_mean_first
            latent_code_last = latent_mean_last
        else:
            epsilon = self._sample_epsilon(latent_mean_first)
            latent_code_first = self._sample_latent_distribution(latent_mean_first, latent_log_var_first, epsilon)
            latent_code_last = self._sample_latent_distribution(latent_mean_last, latent_log_var_last, epsilon)

        # 3) interpolate
        latent_code = self.interpolate_latent_codes(latent_code_first, latent_code_last,
                                                    merge_batch_and_steps=True, step_count=step_count)

        # 4) decode
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

        # inputs = tf.clip_by_value(inputs + tf.random.normal(inputs_shape, stddev=0.01), 0.0, 1.0)
        inputs, inputs_shape, new_shape = self.split_inputs(inputs, merge_batch_and_steps=False)
        step_count = new_shape[1]

        # 1) encode
        encoded_first = self.encoder(inputs[:, 0])
        encoded_last = self.encoder(inputs[:, -1])

        # 2) sample (or take mean)
        latent_mean_first, latent_log_var_first = self._split_encoded(encoded_first)
        latent_mean_last, latent_log_var_last = self._split_encoded(encoded_last)

        epsilon = self._sample_epsilon(latent_mean_first)
        latent_code_first = self._sample_latent_distribution(latent_mean_first, latent_log_var_first, epsilon)
        latent_code_last = self._sample_latent_distribution(latent_mean_last, latent_log_var_last, epsilon)

        # 3) interpolate
        latent_code = self.interpolate_latent_codes(latent_code_first, latent_code_last,
                                                    merge_batch_and_steps=True, step_count=step_count)
        # 4) decode
        decoded = self.decoder(latent_code)
        decoded = tf.reshape(decoded, inputs_shape)

        # 5) loss (interpolation error + kld)
        reconstruction_error = self.compute_reconstruction_loss(target, decoded)
        interpolation_error = self.compute_interpolation_error(target, decoded)
        kl_divergence_first = self.compute_kl_divergence(latent_mean_first, latent_log_var_first)
        kl_divergence_last = self.compute_kl_divergence(latent_mean_last, latent_log_var_last)
        kl_divergence = (kl_divergence_first + kl_divergence_last) * tf.constant(0.5)

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
        batch_size, total_length, *dimensions = inputs.shape
        step_count = total_length // self.step_size

        # region Encode mu, sigma² = e(x)
        # encoded = self.get_interpolated_latent_code(inputs, merge_batch_and_steps=True)
        encoded = self.encoder(self.split_inputs(inputs, merge_batch_and_steps=True)[0])
        encoded = tf.expand_dims(encoded, axis=0)
        latent_mean, latent_log_var = tf.split(encoded, num_or_size_splits=2, axis=-1)
        # endregion

        # region Sample p(z|x)
        code_shape = latent_log_var.shape[2:]
        # noinspection PyTypeChecker
        epsilon_shape = [n_steps - 1, 1, *code_shape]
        epsilon = tf.random.normal(epsilon_shape)
        epsilon = tf.concat([tf.zeros([1, 1, *code_shape]), epsilon], axis=0)

        latent_mean = tf.clip_by_value(latent_mean, -1e7, 1e7)
        latent_log_var = tf.minimum(latent_log_var, 8e1)
        latent_code = latent_mean + tf.exp(0.5 * latent_log_var) * epsilon

        latent_code = tf.reshape(latent_code, [n_steps * batch_size * step_count, *code_shape])
        # endregion

        # region Decode
        decoded = self.decoder(latent_code)
        decoded = tf.reshape(decoded, [n_steps, batch_size, total_length, *dimensions])
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

    @tf.function
    def multi_sample_error(self, inputs: tf.Tensor, metric, n_steps: tf.Tensor = 8):
        outputs = self.autoencode_sampling_epsilon(inputs, n_steps=n_steps)
        error = metric(inputs, outputs)
        return reduce_mean_from(error, start_axis=2)

    @tf.function
    def multi_sample_mae(self, inputs: tf.Tensor):
        return self.multi_sample_error(inputs, metric=tf.losses.mae)

    @tf.function
    def multi_sample_mse(self, inputs: tf.Tensor):
        return self.multi_sample_error(inputs, metric=tf.losses.mse)

    @property
    def additional_test_metrics(self):
        return [
            self.interpolation_mse,
            self.interpolation_mae,
            # self.latent_code_surprisal,
            # self.multi_sample_mae,
            # self.multi_sample_mse,
        ]

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
