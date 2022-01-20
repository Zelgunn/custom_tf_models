import tensorflow as tf
from tensorflow.python.keras import Model
from typing import Dict, Union, Callable

from misc_utils.math_utils import lerp
from custom_tf_models.basic.AE import AE
from custom_tf_models.utils import split_steps, get_known_shape
from misc_utils.train_utils import CustomLearningRateSchedule


# IAE : Interpolating Autoencoder
class IAE(AE):
    def __init__(self,
                 encoder: Model,
                 decoder: Model,
                 step_size: int,
                 use_stochastic_loss: bool = True,
                 **kwargs):
        super(IAE, self).__init__(encoder=encoder,
                                  decoder=decoder,
                                  **kwargs)
        self.step_size = step_size
        self.use_stochastic_loss = use_stochastic_loss

    @tf.function
    def autoencode(self, inputs):
        inputs, inputs_shape, new_shape = self.split_inputs(inputs, merge_batch_and_steps=True)
        decoded = self.decoder(self.encoder(inputs))
        decoded = tf.reshape(decoded, inputs_shape)
        return decoded

    @tf.function
    def encode(self, inputs):
        inputs, _, _ = self.split_inputs(inputs, merge_batch_and_steps=True)
        return self.encoder(inputs)

    @tf.function
    def compute_loss(self,
                     inputs
                     ) -> Dict[str, tf.Tensor]:
        if self.use_stochastic_loss:
            return self.compute_stochastic_loss(inputs)
        else:
            return self.compute_deterministic_loss(inputs)

    @tf.function
    def compute_stochastic_loss(self, inputs) -> Dict[str, tf.Tensor]:
        # region Forward
        start = inputs[:, :self.step_size]
        end = inputs[:, -self.step_size:]

        step_count = tf.shape(inputs)[1]
        max_offset = step_count - self.step_size
        offset = tf.random.uniform(shape=[], minval=0, maxval=max_offset + 1, dtype=tf.int32)
        target = inputs[:, offset:offset + self.step_size]

        factor = tf.cast(offset / max_offset, tf.float32)
        start_encoded = self.encoder(start)
        end_encoded = self.encoder(end)
        latent_code = lerp(start_encoded, end_encoded, factor)

        decoded = self.decoder(latent_code)
        # endregion

        loss = self.compute_reconstruction_loss(target, decoded)
        interpolation_error = loss

        return {"loss": loss, "interpolation_error": interpolation_error}

    @tf.function
    def compute_deterministic_loss(self, inputs) -> Dict[str, tf.Tensor]:
        decoded = self.interpolate(inputs)
        loss = self.compute_reconstruction_loss(inputs, decoded)
        interpolation_error = self.compute_interpolation_error(inputs, decoded)

        return {"loss": loss, "interpolation_error": interpolation_error}

    def compute_interpolation_error(self, inputs, outputs) -> tf.Tensor:
        inputs = inputs[:, self.step_size:-self.step_size]
        outputs = outputs[:, self.step_size:-self.step_size]
        return self.compute_reconstruction_loss(inputs, outputs)

    @tf.function
    def interpolate(self, inputs):
        inputs_shape = tf.shape(inputs)
        encoded = self.get_interpolated_latent_code(inputs, merge_batch_and_steps=True)
        decoded = self.decoder(encoded)
        decoded = tf.reshape(decoded, inputs_shape)
        return decoded

    @tf.function
    def interpolation_mse(self, inputs):
        return self.interpolation_error(inputs, tf.losses.mse)

    @tf.function
    def interpolation_mae(self, inputs):
        return self.interpolation_error(inputs, tf.losses.mae)

    @tf.function
    def interpolation_error(self, inputs, metric):
        interpolated = self.interpolate(inputs)

        inputs = inputs[:, self.step_size: - self.step_size]
        interpolated = interpolated[:, self.step_size: - self.step_size]

        error = metric(inputs, interpolated)
        error = tf.reduce_mean(error, axis=list(range(2, error.shape.rank)))
        return error

    @tf.function
    def interpolation_relative_mse(self, inputs):
        return self.interpolation_relative_error(inputs, tf.losses.mse)

    @tf.function
    def interpolation_relative_mae(self, inputs):
        return self.interpolation_relative_error(inputs, tf.losses.mae)

    @tf.function
    def interpolation_relative_error(self, inputs, metric):
        base_error = metric(inputs, self(inputs))
        base_error = tf.reduce_mean(base_error, axis=list(range(2, base_error.shape.rank)))

        interpolation_error = metric(inputs, self.interpolate(inputs))
        interpolation_error = tf.reduce_mean(interpolation_error, axis=list(range(2, interpolation_error.shape.rank)))

        relative_error = tf.abs(base_error - interpolation_error)
        return relative_error

    @tf.function
    def latent_code_surprisal(self, inputs):
        interpolated_latent_code = self.get_interpolated_latent_code(inputs, merge_batch_and_steps=False)
        interpolated_latent_code = interpolated_latent_code[:, 1: -1]

        inputs = inputs[:, self.step_size: - self.step_size]
        inputs, _, __ = self.split_inputs(inputs, merge_batch_and_steps=True)
        default_latent_code = self.encoder(inputs)
        default_latent_code = tf.reshape(default_latent_code, tf.shape(interpolated_latent_code))

        cosine_distance = tf.losses.cosine_similarity(default_latent_code, interpolated_latent_code,
                                                      axis=list(range(2, default_latent_code.shape.rank)))
        return cosine_distance

    @tf.function
    def get_interpolated_latent_code(self, inputs, merge_batch_and_steps, extra_steps=0):
        inputs, _, new_shape = self.split_inputs(inputs, merge_batch_and_steps=False)
        batch_size, step_count, *_ = new_shape

        encoded_first = self.encoder(inputs[:, 0])
        encoded_last = self.encoder(inputs[:, -1])

        encoded = self.interpolate_latent_codes(encoded_first, encoded_last,
                                                merge_batch_and_steps, step_count, extra_steps)
        return encoded

    @tf.function
    def interpolate_latent_codes(self,
                                 encoded_first: tf.Tensor,
                                 encoded_last: tf.Tensor,
                                 merge_batch_and_steps: bool,
                                 step_count: int,
                                 extra_steps: int = 0
                                 ):
        code_shape = get_known_shape(encoded_first)
        batch_size = code_shape[0]
        code_rank = encoded_first.shape.rank
        total_step_count = step_count + extra_steps * 2

        encoded_shape_dimensions = tf.unstack(tf.shape(encoded_first)[1:])
        tile_multiples = [1, total_step_count] + [1] * (code_rank - 1)
        encoded_first = tf.tile(tf.expand_dims(encoded_first, axis=1), tile_multiples)
        encoded_last = tf.tile(tf.expand_dims(encoded_last, axis=1), tile_multiples)

        weights = tf.range(start=-extra_steps, limit=step_count + extra_steps, dtype=tf.int32)
        weights = tf.cast(weights / (step_count - 1), tf.float32)
        weights = tf.reshape(weights, tile_multiples)

        encoded = encoded_first * (1.0 - weights) + encoded_last * weights
        if merge_batch_and_steps:
            encoded = tf.reshape(encoded, [batch_size * total_step_count, *encoded_shape_dimensions])
        return encoded

    @tf.function
    def step_mse(self, inputs, ground_truth):
        error = tf.square(inputs - ground_truth)
        error, _, _ = self.split_inputs(error, merge_batch_and_steps=False)
        reduction_axis = list(range(2, error.shape.rank))
        error = tf.reduce_mean(error, axis=reduction_axis)
        return error

    @tf.function
    def split_inputs(self, inputs, merge_batch_and_steps):
        return split_steps(inputs, self.step_size, merge_batch_and_steps)

    @tf.function
    def gradient_norm(self, inputs) -> tf.Tensor:
        with tf.GradientTape() as tape:
            interpolated = self.interpolate(inputs)

            inputs = inputs[:, self.step_size: - self.step_size]
            interpolated = interpolated[:, self.step_size: - self.step_size]

            error = tf.reduce_mean(tf.square(interpolated - inputs))

        gradients = tape.gradient(error, self.trainable_variables)
        gradients = [gradient for gradient in gradients if gradient is not None]
        norms = [tf.norm(gradient) for gradient in gradients]
        norm = tf.reduce_sum(norms)
        norm = tf.reshape(norm, [1, 1])

        return norm

    def get_config(self):
        config = {
            **super(IAE, self).get_config(),
            "step_count": self.step_size,
            "use_stochastic_loss": self.use_stochastic_loss,
        }
        return config

    @property
    def additional_test_metrics(self):
        return [
            self.interpolation_mse,
            self.interpolation_mae,
            # self.latent_code_surprisal,
        ]


# WIP
class IAESchedule(CustomLearningRateSchedule):
    def __init__(self,
                 learning_rate: Union[Callable, float] = 1e-3,
                 update_rate=0.01,
                 max_reduction_factor=32.0,
                 recover_rate=1.1,
                 epsilon=1e-7,
                 **kwargs):
        super(IAESchedule, self).__init__(learning_rate=learning_rate,
                                          **kwargs)

        self.update_rate = tf.constant(update_rate)
        self.max_reduction_factor = tf.constant(max_reduction_factor)
        self.recover_rate = tf.constant(recover_rate)
        self.epsilon = tf.constant(epsilon)

        self.current_loss = tf.Variable(initial_value=0.0, trainable=False, name="current_loss", dtype=tf.float32)
        self.previous_loss = tf.Variable(initial_value=0.0, trainable=False, name="previous_loss", dtype=tf.float32)
        self.step_rate = tf.Variable(initial_value=0.0, trainable=False, name="step_rate", dtype=tf.float32)
        self.current_rate = tf.Variable(initial_value=1.0, trainable=False, name="current_rate", dtype=tf.float32)

    def update(self, loss):
        delta = tf.abs(loss - self.current_loss)
        step_rate = self.step_rate * (tf.ones([]) - self.update_rate) + delta * self.update_rate
        self.step_rate.assign(step_rate)

        self.previous_loss.assign(self.current_loss)
        self.current_loss.assign(loss)

    def call(self, step):
        delta = self.current_loss - self.previous_loss
        rate = delta / (self.step_rate + self.epsilon)
        rate = 1.0 / tf.clip_by_value(rate, 1.0, self.max_reduction_factor)
        rate = tf.minimum(rate, tf.minimum(self.current_rate * self.recover_rate, 1.0))
        self.current_rate.assign(rate)
        base_learning_rate = self.get_learning_rate(step)
        return base_learning_rate * rate

    def get_config(self):
        base_config = super(IAESchedule, self).get_config()
        config = {
            "update_rate": self.update_rate.numpy(),
            "max_reduction_factor": self.max_reduction_factor.numpy(),
            "recover_rate": self.recover_rate.numpy(),
            "epsilon": self.epsilon.numpy(),
        }
        return {**base_config, **config}
