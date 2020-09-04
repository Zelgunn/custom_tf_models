import tensorflow as tf
from tensorflow.python.keras import Model
from typing import Dict

from custom_tf_models import AE
from misc_utils.general import expand_dims_to_rank
from misc_utils.math_utils import lerp


class MinimalistDescriptor(AE):
    def __init__(self,
                 encoder: Model,
                 decoder: Model,
                 stop_encoder: Model,
                 max_steps: int,
                 learning_rate,
                 stop_lambda=1e-3,
                 stop_residual_gradients=True,
                 train_stride=1000,
                 noise_type="dense",
                 noise_factor_distribution="normal",
                 binarization_temperature=50.0,
                 seed=None,
                 **kwargs):
        super(MinimalistDescriptor, self).__init__(encoder=encoder,
                                                   decoder=decoder,
                                                   learning_rate=learning_rate,
                                                   **kwargs)

        self.stop_encoder = stop_encoder
        self.max_steps = max_steps
        self.stop_lambda = stop_lambda
        self.stop_residual_gradients = stop_residual_gradients
        self.noise_type = noise_type
        self.noise_factor_distribution = noise_factor_distribution
        self.binarization_temperature = binarization_temperature
        self.seed = seed

        self._binarization_threshold = tf.constant(0.5, dtype=tf.float32, name="bin_threshold")
        self._binarization_temperature = tf.constant(binarization_temperature, dtype=tf.float32, name="bin_temperature")
        self.train_stride = train_stride
        self.train_step_counter = tf.Variable(initial_value=0, trainable=False, dtype=tf.int32)

    # region Call
    @tf.function
    def call(self, inputs, training=None, mask=None):
        training = False if training is None else training

        if training:
            return self.call_history(inputs)
        else:
            return self.call_no_history(inputs)

    @tf.function
    def call_history(self, inputs):
        inputs_shape = (None, *inputs.shape[1:])
        batch_size = tf.shape(inputs)[0]
        max_steps = self.get_current_max_steps()

        def loop_cond(i, keep_going, _, __, ___):
            keep_going = tf.reduce_any(keep_going)
            below_max_steps = i < max_steps
            return keep_going and below_max_steps

        def loop_body(i, keep_going, residual, continue_array, residuals_array):
            step_residual, step_keep_going = self.main_loop_step(residual)
            step_keep_going = tf.where(keep_going, step_keep_going, tf.zeros_like(step_keep_going))
            keep_going = tf.logical_and(keep_going, step_keep_going > 0.5)

            expanded_keep_going = tf.reshape(keep_going, [batch_size] + [1] * (residual.shape.rank - 1))
            residual = tf.where(expanded_keep_going, step_residual, residual)

            continue_array = continue_array.write(i, step_keep_going)
            residuals_array = residuals_array.write(i, residual)
            i = i + 1

            if self.stop_residual_gradients:
                residual = tf.stop_gradient(residual)

            return i, keep_going, residual, continue_array, residuals_array

        initial_i = tf.constant(0, dtype=tf.int32)
        initial_keep_going = tf.ones(shape=[batch_size], dtype=tf.bool)
        initial_residual = inputs
        initial_keep_going_array = tf.TensorArray(dtype=tf.float32, element_shape=[None], size=max_steps)
        initial_residuals_array = tf.TensorArray(dtype=tf.float32, element_shape=inputs_shape, size=max_steps)
        loop_vars = [initial_i, initial_keep_going, initial_residual, initial_keep_going_array, initial_residuals_array]

        loop_outputs = tf.while_loop(cond=loop_cond, body=loop_body, loop_vars=loop_vars)
        final_i, final_keep_going, final_residual, final_keep_going_array, final_residuals_array = loop_outputs

        final_keep_going_array = final_keep_going_array.stack()
        final_residuals_array = final_residuals_array.stack()

        return final_residual, final_keep_going_array, final_residuals_array

    @tf.function
    def call_no_history(self, inputs):
        batch_size = tf.shape(inputs)[0]

        def loop_cond(i, keep_going, _):
            keep_going = tf.reduce_any(keep_going)
            below_max_steps = i < self.max_steps
            return keep_going and below_max_steps

        def loop_body(i, keep_going, residual):
            step_residual, step_keep_going = self.main_loop_step(residual)
            keep_going = tf.logical_and(keep_going, tf.greater(step_keep_going, 0.5))

            expanded_keep_going = tf.reshape(keep_going, [batch_size] + [1] * (residual.shape.rank - 1))
            residual = tf.where(expanded_keep_going, step_residual, residual)

            i = i + 1
            return i, keep_going, residual

        initial_i = tf.constant(0, dtype=tf.int32)
        initial_keep_going = tf.ones(shape=[batch_size], dtype=tf.bool)
        initial_residual = inputs
        loop_vars = [initial_i, initial_keep_going, initial_residual]

        loop_outputs = tf.while_loop(cond=loop_cond, body=loop_body, loop_vars=loop_vars)
        final_residual = loop_outputs[2]

        outputs = inputs - final_residual
        return outputs

    @tf.function
    def main_loop_step(self, residual):
        encoded = self.encoder(residual)

        keep_going = self.stop_encoder(encoded)
        keep_going = tf.squeeze(keep_going, axis=-1)
        keep_going = self.binarize(keep_going)

        latent_code_rank = encoded.shape.rank
        keep_going_multiplier = tf.reshape(keep_going, [-1] + [1] * (latent_code_rank - 1))

        # encoded = binarize(encoded, 0.0, 50.0)
        # bins_count = 100
        # bins_init = tf.zeros(shape=[bins_count], dtype=tf.int32)
        # tmp = tf.cast(keep_going * (bins_count - 1), tf.int32)
        # tmp = tf.reshape(tmp, [-1])
        # tmp_count = tf.shape(tmp)[0]
        #
        # def tmp_cond(i, _):
        #     return i < tmp_count
        #
        # def tmp_loop(i, bins):
        #     bins += tf.one_hot(tmp[i], bins_count, dtype=tf.int32)
        #     i += 1
        #     return i, bins
        #
        # _, bins_final = tf.while_loop(tmp_cond, tmp_loop, [0, bins_init])
        # bins_final /= tmp_count
        # amount = 10
        # a = tf.reduce_sum(bins_final[:amount])
        # b = tf.reduce_sum(bins_final[amount:-amount])
        # c = tf.reduce_sum(bins_final[-amount:])
        # tf.print(a, b, c)

        decoded = self.decoder(encoded * keep_going_multiplier)
        residual -= decoded

        return residual, keep_going

    @tf.function
    def get_current_max_steps(self) -> tf.Tensor:
        if self.train_stride is None:
            current_max_steps = self.max_steps
        else:
            stride = tf.constant(self.train_stride, dtype=tf.int32, name="train_stride")
            current_max_steps = tf.minimum(self.max_steps, (self.train_step_counter // stride) + 2)
        return current_max_steps

    @tf.function
    def binarize(self, x: tf.Tensor):
        return binarize(x, self._binarization_threshold, self._binarization_temperature)

    # endregion

    # region Noise
    @tf.function
    def sample_noise_factor(self) -> tf.Tensor:
        if self.noise_factor_distribution == "uniform":
            noise_factor = tf.random.uniform(shape=[], minval=0.0, maxval=1.0, dtype=tf.float32, seed=self.seed)
        else:
            noise_factor = tf.random.normal(shape=[], mean=1.0, stddev=0.25, dtype=tf.float32, seed=self.seed)
            noise_factor = tf.clip_by_value(noise_factor, 0.0, 2.0)
            noise_factor = tf.where(noise_factor <= 1.0, noise_factor, 2.0 - noise_factor)
            noise_factor = 1.0 - noise_factor
        return noise_factor

    @tf.function
    def add_noise(self, tensor: tf.Tensor, noise_factor: tf.Tensor) -> tf.Tensor:
        noise = tf.random.normal(shape=tf.shape(tensor), mean=0.0, stddev=1.0, seed=self.seed)
        # noise = tf.random.uniform(shape=tf.shape(tensor), minval=-1.0, maxval=1.0, seed=self.seed)
        if self.noise_type == "dense":
            tensor_with_noise = lerp(tensor, noise, noise_factor)
        else:
            noise_mask = tf.random.uniform(shape=tf.shape(tensor), minval=0.0, maxval=1.0, seed=self.seed)
            tensor_with_noise = tf.where(noise_mask < noise_factor, noise, tensor)
        return tensor_with_noise

    # endregion

    # region Compute Loss
    @tf.function
    def compute_reconstruction_loss(self,
                                    # inputs: tf.Tensor,
                                    residuals_array: tf.Tensor,
                                    keep_going_array: tf.Tensor
                                    ) -> tf.Tensor:
        steps_count = tf.shape(residuals_array)[0]
        # inputs = tf.expand_dims(inputs, axis=0)
        # outputs = inputs - residuals_array

        # inputs_perceptual_code = self.get_perceptual_code(inputs)
        # inputs_perceptual_code = tf.tile(inputs_perceptual_code, multiples=[self.max_steps, 1, 1, 1, 1, 1])
        # outputs_perceptual_code = self.get_perceptual_code(outputs)
        # perceptual_loss = tf.abs(inputs_perceptual_code - outputs_perceptual_code)

        reconstruction_loss = tf.square(residuals_array)

        keep_going = tf.math.greater(keep_going_array, 0.5)
        mask = tf.pad(keep_going[:-1], ((1, 0), (0, 0)), constant_values=True)
        mask = expand_dims_to_rank(mask, target=reconstruction_loss)
        weights = tf.cast(steps_count, tf.float32) / tf.reduce_sum(tf.cast(mask, tf.float32), axis=0, keepdims=True)

        reconstruction_loss = tf.where(mask, reconstruction_loss, tf.zeros_like(reconstruction_loss))
        reconstruction_loss = tf.reduce_mean(reconstruction_loss * weights)

        return reconstruction_loss

    @tf.function
    def compute_stop_loss(self,
                          keep_going_array: tf.Tensor,
                          noise_factor: tf.Tensor
                          ) -> tf.Tensor:
        first_token = keep_going_array[0]
        next_tokens = keep_going_array[1:self.get_current_max_steps()]

        first_token_loss = tf.square(tf.reduce_mean(tf.constant(1.0) - first_token))
        next_tokens_loss = tf.square(tf.reduce_mean(next_tokens[:])) * (tf.constant(1.0) - noise_factor)

        stop_loss = next_tokens_loss
        stop_loss += first_token_loss
        return stop_loss

    @tf.function
    def compute_loss(self,
                     inputs: tf.Tensor
                     ) -> Dict[str, tf.Tensor]:
        noise_factor = self.sample_noise_factor()
        noised_inputs = self.add_noise(inputs, noise_factor)

        final_residuals, keep_going_array, residuals_array = self(noised_inputs, training=True)
        if self.stop_residual_gradients:
            reconstruction_loss = self.compute_reconstruction_loss(residuals_array, keep_going_array)
        else:
            reconstruction_loss = tf.reduce_mean(tf.square(final_residuals))

        stop_loss = self.compute_stop_loss(keep_going_array, noise_factor)

        stop_lambda = tf.constant(self.stop_lambda, name="stop_lambda")
        loss = reconstruction_loss + stop_lambda * stop_loss

        steps_count = tf.reduce_sum(tf.cast(tf.greater(keep_going_array, 0.5), tf.float32), axis=0)

        return {
            "loss": loss,
            "reconstruction_loss": reconstruction_loss,
            "stop_loss": stop_loss,
            "steps_count": steps_count,
        }

    # endregion

    # region Training
    @tf.function
    def train_step(self, data):
        with tf.GradientTape() as tape:
            metrics = self.compute_loss(data)
            loss = metrics["loss"]

        gradients = tape.gradient(loss, self.trainable_variables)
        gradients = [tf.clip_by_value(grad, -1e-2, 1e-2) for grad in gradients]
        # gradients = [tf.clip_by_norm(grad, 1.0) for grad in gradients]
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        grad_norms = [tf.norm(grad) for grad in gradients]
        metrics["grad_norms"] = tf.reduce_max(grad_norms)

        self.train_step_counter.assign_add(1)
        return metrics

    @tf.function
    def test_step(self, data):
        return self.compute_loss(data)

    @property
    def autoencoder_trainable_variables(self):
        return self.encoder.trainable_variables + self.decoder.trainable_variables

    # endregion

    # region Anomaly detection
    @tf.function
    def compute_description_length(self, inputs: tf.Tensor) -> tf.Tensor:
        _, keep_going_array, _, = self.call_history(inputs)
        keep_going_array = tf.reduce_sum(keep_going_array, axis=0)
        return keep_going_array

    @tf.function
    def compute_description_length_2(self, inputs: tf.Tensor) -> tf.Tensor:
        _, keep_going_array, _ = self.call_history(inputs)
        mask = keep_going_array > 0.5
        length = tf.reduce_sum(tf.cast(mask, tf.int32), axis=0)

        batch_size = tf.shape(keep_going_array)[1]
        last_indices = tf.stack([length - 1, tf.range(batch_size)], axis=1)
        last = tf.gather_nd(keep_going_array, last_indices)

        result = tf.cast(length, tf.float32) + last
        return result

    # endregion

    # region Config
    def get_config(self) -> Dict:
        base_config = super(MinimalistDescriptor, self).get_config()
        config = {
            **base_config,
            "stop_encoder": self.stop_encoder.get_config(),
            "max_steps": self.max_steps,
            "stop_lambda": self.stop_lambda,
            "stop_residual_gradients": self.stop_residual_gradients,
            "train_stride": self.train_stride,
            "noise_type": self.noise_type,
            "noise_factor_distribution": self.noise_factor_distribution,
            "binarization_temperature": self.binarization_temperature,
            "seed": self.seed,
        }
        return config

    @property
    def models_ids(self) -> Dict[Model, str]:
        base_models_ids = super(MinimalistDescriptor, self).models_ids
        models_ids = {
            **base_models_ids,
            self.stop_encoder: "stop_encoder",
        }
        return models_ids

    @property
    def additional_test_metrics(self):
        return [self.compute_description_length, self.compute_description_length_2]

    # endregion


@tf.function
def binarize(x: tf.Tensor,
             threshold: tf.Tensor,
             temperature: tf.Tensor
             ) -> tf.Tensor:
    return 1.0 / (1.0 + tf.exp(-temperature * (x - threshold)))


def main():
    pass


if __name__ == "__main__":
    main()
