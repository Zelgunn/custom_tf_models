import tensorflow as tf
from typing import Callable

from misc_utils.general import get_known_shape
from misc_utils.math_utils import diff


# @tf.function
def split_steps(inputs, step_size, merge_batch_and_steps):
    """ Splits inputs into N steps of size `step_size` and merge the dimension holding the number of steps into
    the batch dimension if `merge_batch_and_steps` is true.

    `N = total_length // step_size`

    `step_size` must be a valid divider of `total_length`.

    :param inputs: A 2D+ tensor with shape [batch_size, total_length, *dimensions].
    :param step_size: The size of each step. A single integer.
    :param merge_batch_and_steps: If true, output_shape is `[batch_size * N, step_size, *dimensions],
        else it is `[batch_size, N, step_size, *dimensions].

    :return: A tuple containing:
        1) A tensor with either the same rank or rank + 1 (see `merge_batch_and_steps`) with same type
            and total dimension as inputs.
        2) The original shape
        3) The resulting shape (as if it was not merged).
    """
    inputs_shape = get_known_shape(inputs)
    batch_size, total_length, *dimensions = inputs_shape
    step_count = total_length // step_size

    unmerged_shape = [batch_size, step_count, step_size, *dimensions]
    if merge_batch_and_steps:
        new_shape = [batch_size * step_count, step_size, *dimensions]
    else:
        new_shape = unmerged_shape

    inputs = tf.reshape(inputs, new_shape)

    return inputs, inputs_shape, unmerged_shape


# @tf.function
def gradient_difference_loss(y_true, y_pred, axis=(-2, -3), alpha=1):
    if not isinstance(axis, (tuple, list)):
        axis = [axis]

    grad_losses = []

    for current_axis in axis:
        true_grad = diff(y_true, axis=current_axis)
        pred_grad = diff(y_pred, axis=current_axis)
        grad_delta = tf.abs(tf.abs(true_grad) - tf.abs(pred_grad))
        grad_loss = tf.pow(grad_delta, alpha)
        grad_loss = tf.reduce_mean(grad_loss, axis=axis)
        grad_losses.append(grad_loss)

    total_grad_loss = tf.reduce_sum(grad_losses, axis=0)
    return total_grad_loss


class LambdaModel(tf.keras.Model):
    def __init__(self, function: Callable,
                 use_training_keyword=False,
                 use_mask_keyword=False,
                 **kwargs):
        super(LambdaModel, self).__init__(**kwargs)
        self.function = function
        self.use_training_keyword = use_training_keyword
        self.use_mask_keyword = use_mask_keyword

    def call(self, inputs, training=None, mask=None):
        if self.use_training_keyword:
            if self.use_mask_keyword:
                return self.function(inputs, training=training, mask=mask)
            else:
                return self.function(inputs, training=training)
        else:
            if self.use_mask_keyword:
                return self.function(inputs, mask=mask)
            else:
                return self.function(inputs)
