import tensorflow as tf
from tensorflow.python.keras import Model
from enum import IntEnum
from typing import Union, Callable

from misc_utils.math_utils import lerp, reduce_sum_from


class GANLossMode(IntEnum):
    VANILLA = 0,
    LS_GAN = 1,
    W_GAN_GP = 2


class GANLoss(object):
    def __init__(self, mode: GANLossMode):
        self.mode = mode

    def __call__(self, predicted: tf.Tensor, is_real: bool):
        if self.mode == GANLossMode.VANILLA:
            labels = self.get_labels(predicted=predicted, is_real=is_real)
            loss = tf.losses.binary_crossentropy(y_pred=predicted, y_true=labels, from_logits=True)

        elif self.mode == GANLossMode.LS_GAN:
            labels = self.get_labels(predicted=predicted, is_real=is_real)
            loss = tf.losses.mean_squared_error(y_pred=predicted, y_true=labels)

        elif self.mode == GANLossMode.W_GAN_GP:
            loss = self.w_gan_loss(predicted=predicted, is_real=is_real)

        else:
            raise NotImplementedError("Unknown GAN loss mode {}.".format(self.mode))

        return tf.reduce_mean(loss)

    @staticmethod
    def get_labels(predicted: tf.Tensor, is_real: bool):
        if is_real:
            return tf.ones_like(predicted)
        else:
            return tf.zeros_like(predicted)

    @staticmethod
    def w_gan_loss(predicted: tf.Tensor, is_real: bool):
        if is_real:
            return predicted
        else:
            return -predicted


@tf.function
def compute_gradient_penalty(real: tf.Tensor, fake: tf.Tensor, discriminator: Union[Model, Callable]) -> tf.Tensor:
    fake = tf.stop_gradient(fake)
    batch_size = tf.shape(real)[0]
    factors_shape = [batch_size] + [1] * (real.shape.rank - 1)
    factors = tf.random.uniform(shape=factors_shape, minval=0.0, maxval=1.0, dtype=tf.float32)
    x_hat = lerp(real, fake, factors)

    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(x_hat)
        discriminated = discriminator(x_hat)

    gradients = tape.gradient(discriminated, x_hat)
    penalty = tf.sqrt(reduce_sum_from(tf.square(gradients)))
    penalty = tf.reduce_mean(tf.square(penalty - 1.0))

    return penalty
