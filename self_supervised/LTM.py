import tensorflow as tf
from tensorflow.python.keras import Model
from typing import Dict, Union

from misc_utils.general import get_known_shape
from misc_utils.math_utils import reduce_mean_from
from custom_tf_models.utils import split_steps


# LTM : Linearized Temporal Mechanism
class LTM(Model):
    def __init__(self,
                 encoder: Model,
                 step_size: int,
                 interpolation_lambda: float = 1.0,
                 pull_lambda: float = 1.0,
                 normalize_interpolated: bool = True,
                 **kwargs):
        super(LTM, self).__init__(**kwargs)

        self.encoder = encoder
        self.step_size = step_size
        self.interpolation_lambda = interpolation_lambda
        self.pull_lambda = pull_lambda
        self.normalize_interpolated = normalize_interpolated

        self.train_step_index = tf.Variable(initial_value=0, trainable=False, name="train_step_index", dtype=tf.int32)

    @tf.function
    def call(self, inputs, training=None, mask=None):
        inputs, _, new_shape = split_steps(inputs, self.step_size, merge_batch_and_steps=True)
        batch_size, step_count, *_ = new_shape

        latent_code = self.encoder(inputs, training=training)
        latent_code_shape = get_known_shape(latent_code)[1:]

        latent_code = tf.reshape(latent_code, [batch_size, step_count, *latent_code_shape])
        return latent_code

    # region Train / Test

    @tf.function
    def train_step(self, data) -> Dict[str, tf.Tensor]:
        with tf.GradientTape() as tape:
            metrics = self.compute_loss(data)
            loss = metrics["loss"]

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.train_step_index.assign_add(1)
        self._train_counter.assign(tf.cast(self.train_step_index, tf.int64))

        return metrics

    @tf.function
    def test_step(self, data):
        return self.compute_loss(data)

    # endregion

    # region Loss
    @tf.function
    def compute_loss(self, inputs) -> Dict[str, tf.Tensor]:
        latent_codes = self(inputs, training=True, mask=None)
        step_count = get_known_shape(latent_codes)[1]

        first_latent_code = latent_codes[:, 0]
        last_latent_code = latent_codes[:, -1]
        target_latent_codes = self.temporal_interpolation(first_latent_code, last_latent_code, step_count)
        if self.normalize_interpolated:
            normalization_axis = [i for i in range(2, len(target_latent_codes.shape))]
            target_latent_codes = tf.linalg.l2_normalize(target_latent_codes, axis=normalization_axis)

        interpolation_loss = self.compute_interpolation_loss(latent_codes, target_latent_codes)
        pull_loss = self.compute_pull_loss(first_latent_code, last_latent_code)
        interpolation_lambda = tf.convert_to_tensor(self.interpolation_lambda)
        pull_lambda = tf.convert_to_tensor(self.pull_lambda)
        loss = (interpolation_lambda * interpolation_loss +
                pull_lambda * pull_loss)

        return {
            "loss": loss,
            "interpolation": interpolation_loss,
            "pull": pull_loss,
        }

    @tf.function
    def compute_interpolation_loss(self, latent_codes: tf.Tensor, target_latent_codes: tf.Tensor) -> tf.Tensor:
        latent_codes = latent_codes[:, 1:-1]
        target_latent_codes = target_latent_codes[:, 1:-1]
        interpolation_loss = tf.reduce_mean(tf.square(latent_codes - target_latent_codes))
        return interpolation_loss

    @tf.function
    def compute_pull_loss(self, first_latent_code: tf.Tensor, last_latent_code: tf.Tensor) -> tf.Tensor:
        delta = first_latent_code - last_latent_code
        batch_size = get_known_shape(first_latent_code)[0]
        delta = tf.reshape(delta, shape=[batch_size, -1])
        delta_norm = tf.norm(delta, ord=2, axis=-1)
        return tf.reduce_mean(tf.square(delta_norm - 1.0))

    # endregion

    # region Utils
    @tf.function
    def temporal_interpolation(self, first_latent_code: tf.Tensor,
                               last_latent_code: tf.Tensor,
                               step_count: Union[tf.Tensor, int]
                               ) -> tf.Tensor:
        first_latent_code = self.tile_latent_code(first_latent_code, step_count, stop_gradient=True)
        last_latent_code = self.tile_latent_code(last_latent_code, step_count, stop_gradient=True)

        weights_shape = [1, step_count] + [1] * (len(first_latent_code.shape) - 2)
        weights = tf.range(start=0, limit=step_count, dtype=tf.int32)
        weights = tf.cast(weights / (step_count - 1), tf.float32)
        weights = tf.reshape(weights, weights_shape)

        latent_codes = first_latent_code * (1.0 - weights) + last_latent_code * weights
        return latent_codes

    @tf.function
    def tile_latent_code(self,
                         latent_code: tf.Tensor,
                         step_count: int,
                         stop_gradient: bool):

        if stop_gradient:
            latent_code = tf.stop_gradient(latent_code)

        tile_multiples = [1, step_count] + [1] * (len(latent_code.shape) - 1)
        latent_code = tf.expand_dims(latent_code, axis=1)
        latent_code = tf.tile(latent_code, tile_multiples)

        return latent_code

    # endregion

    # region Anomaly Detection
    @tf.function
    def compute_interpolation_error(self, inputs) -> tf.Tensor:
        latent_codes = self(inputs, training=False, mask=None)

        step_count = get_known_shape(latent_codes)[1]
        first_latent_code = latent_codes[:, 0]
        last_latent_code = latent_codes[:, -1]

        target_latent_codes = self.temporal_interpolation(first_latent_code, last_latent_code, step_count)

        latent_codes = latent_codes[:, 1:-1]
        target_latent_codes = target_latent_codes[:, 1:-1]
        error = tf.square(latent_codes - target_latent_codes)

        return reduce_mean_from(error, start_axis=1)

    @tf.function
    def compute_norm_error(self, inputs) -> tf.Tensor:
        first_step = inputs[:, :self.step_size]
        last_step = inputs[:, -self.step_size:]

        first_latent_code = self.encoder(first_step, training=False)
        last_latent_code = self.encoder(last_step, training=False)

        delta = first_latent_code - last_latent_code
        batch_size = get_known_shape(first_latent_code)[0]
        delta = tf.reshape(delta, shape=[batch_size, -1])
        delta_norm = tf.norm(delta, ord=2, axis=-1)

        error = tf.square(delta_norm - 1.0)

        return error

    # endregion

    def get_config(self):
        config = {
            "encoder": self.encoder.get_config(),
            "step_size": self.step_size,
            "pull_lambda": self.pull_lambda,
            "interpolation_lambda": self.interpolation_lambda,
            "normalize_interpolated": self.normalize_interpolated,
        }

        if self.optimizer is not None:
            config["optimizer"] = self.optimizer.get_config()

        return config
