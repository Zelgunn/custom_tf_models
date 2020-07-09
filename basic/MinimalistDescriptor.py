import tensorflow as tf
from tensorflow.python.keras import Model
from typing import Tuple, Dict

from custom_tf_models import AE


class MinimalistDescriptor(AE):
    def __init__(self,
                 encoder: Model,
                 decoder: Model,
                 stop_encoder: Model,
                 max_steps: int,
                 learning_rate,
                 stop_lambda=1e-3,
                 stop_residual_gradients=False,
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
        self.seed = seed

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

        def loop_cond(i, keep_going, _, __, ___):
            keep_going = tf.reduce_any(keep_going)
            below_max_steps = i < self.max_steps
            return keep_going and below_max_steps

        def loop_body(i, keep_going, residual, continue_array, residuals_array):
            step_residual, step_keep_going = self.main_loop_step(inputs, residual)
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
        initial_keep_going_array = tf.TensorArray(dtype=tf.float32, element_shape=[None], size=self.max_steps)
        initial_residuals_array = tf.TensorArray(dtype=tf.float32, element_shape=inputs_shape, size=self.max_steps)
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
            step_residual, step_keep_going = self.main_loop_step(inputs, residual)
            keep_going = tf.logical_and(keep_going, step_keep_going > 0.5)

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
    def autoencode(self, inputs):
        rank = inputs.shape.rank
        inputs = tf.tile(inputs, [1] * (rank - 1) + [2])
        return super(MinimalistDescriptor, self).autoencode(inputs)

    @tf.function
    def main_loop_step(self, inputs, residual):
        step_inputs = tf.concat([inputs, residual], axis=-1)
        encoded = self.encoder(step_inputs)

        keep_going = self.stop_encoder(encoded)
        keep_going = tf.squeeze(keep_going, axis=-1)

        latent_code_rank = encoded.shape.rank
        keep_going_multiplier = tf.reshape(keep_going, [-1] + [1] * (latent_code_rank - 1))

        decoded = self.decoder(encoded * keep_going_multiplier)
        residual -= decoded

        return residual, keep_going

    @tf.function
    def main_loop_cond(self, step, keep_going):
        keep_going = tf.reduce_any(keep_going)
        keep_going = tf.logical_and(keep_going, step < self.max_steps)
        return keep_going

    # region Compute Loss
    @tf.function
    def compute_reconstruction_loss(self, final_residual, residuals_array, keep_going_array):
        mask = keep_going_array > 0.5
        mask = tf.pad(mask[:-1], ((1, 0), (0, 0)), constant_values=True)
        rank_offset = residuals_array.shape.rank - mask.shape.rank
        mask = tf.reshape(mask, tf.concat([tf.shape(mask), [1] * rank_offset], axis=0))
        residuals_array = tf.where(mask, residuals_array, tf.zeros_like(residuals_array))
        # if self.stop_residual_gradients:
        #     raise NotImplementedError()
        # else:
        #     reconstruction_loss = tf.reduce_mean(tf.square(final_residual))
        reconstruction_loss = tf.reduce_mean(tf.square(residuals_array))
        return reconstruction_loss

    @tf.function
    def compute_stop_loss(self, keep_going_array, noise_factor):
        first_token = keep_going_array[0]
        next_tokens = keep_going_array[1:]

        first_token_loss = tf.reduce_mean(tf.losses.binary_crossentropy(tf.ones_like(first_token), first_token))
        next_tokens_loss = tf.reduce_mean(next_tokens)

        stop_loss = first_token_loss + next_tokens_loss * noise_factor
        return stop_loss

    @tf.function
    def sample_noise_factor(self):
        x = tf.random.normal(shape=[], mean=1.0, stddev=0.25, dtype=tf.float32, seed=self.seed)
        x = tf.clip_by_value(x, 0.0, 2.0)
        x = tf.where(x <= 1.0, x, 2.0 - x)
        return x

    @tf.function
    def compute_loss(self,
                     inputs
                     ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        noise = tf.random.normal(shape=tf.shape(inputs))
        noise_factor = self.sample_noise_factor()
        inputs = noise_factor * inputs + (1.0 - noise_factor) * noise

        final_residual, keep_going_array, residuals_array = self(inputs, training=True)
        reconstruction_loss = self.compute_reconstruction_loss(final_residual, residuals_array, keep_going_array)
        stop_loss = self.compute_stop_loss(keep_going_array, noise_factor)

        loss = reconstruction_loss + self.stop_lambda * stop_loss
        weighted_reconstruction = tf.stop_gradient(reconstruction_loss) * noise_factor
        return loss, reconstruction_loss, stop_loss, weighted_reconstruction

    # endregion

    @tf.function
    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            loss, reconstruction_loss, stop_loss, weighted_reconstruction = self.compute_loss(inputs)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss, reconstruction_loss, stop_loss, weighted_reconstruction

    @property
    def metrics_names(self):
        return ["loss", "reconstruction", "stop", "weighted_reconstruction"]

    def get_config(self) -> Dict:
        base_config = super(MinimalistDescriptor, self).get_config()
        config = {
            **base_config,
            "stop_encoder": self.stop_encoder.get_config(),
            "max_steps": self.max_steps,
            "stop_lambda": self.stop_lambda,
            "stop_residual_gradients": self.stop_residual_gradients,
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


def main():
    pass


if __name__ == "__main__":
    main()
