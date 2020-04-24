import tensorflow as tf
from typing import Callable, Tuple

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


def reduce_mean_from(inputs: tf.Tensor, start_axis=1, keepdims=False) -> tf.Tensor:
    return reduce_from(inputs, start_axis, tf.reduce_mean, keepdims=keepdims)


def reduce_sum_from(inputs: tf.Tensor, start_axis=1, keepdims=False) -> tf.Tensor:
    return reduce_from(inputs, start_axis, tf.reduce_sum, keepdims=keepdims)


def reduce_prod_from(inputs: tf.Tensor, start_axis=1, keepdims=False) -> tf.Tensor:
    return reduce_from(inputs, start_axis, tf.reduce_prod, keepdims=keepdims)


def reduce_std_from(inputs: tf.Tensor, start_axis=1, keepdims=False) -> tf.Tensor:
    return reduce_from(inputs, start_axis, tf.math.reduce_std, keepdims=keepdims)


def reduce_adjusted_std_from(inputs: tf.Tensor, start_axis=1, keepdims=False) -> tf.Tensor:
    return reduce_from(inputs, start_axis, reduce_adjusted_stddev, keepdims=keepdims)


def reduce_from(inputs: tf.Tensor, start_axis: int, fn: Callable, **kwargs):
    if start_axis < 0:
        start_axis = inputs.shape.rank + start_axis
    reduction_axis = tuple(range(start_axis, inputs.shape.rank))
    return fn(inputs, axis=reduction_axis, **kwargs)


def reduce_adjusted_stddev(inputs: tf.Tensor, axis: int, keepdims=False) -> tf.Tensor:
    inputs_shape = tf.shape(inputs)
    sample_dims = tf.gather(inputs_shape, axis)
    sample_size = tf.math.reduce_prod(input_tensor=sample_dims)
    sample_stddev = tf.math.reduce_std(input_tensor=inputs, axis=axis, keepdims=keepdims)
    min_stddev = tf.math.rsqrt(tf.cast(sample_size, inputs.dtype))
    adjusted_stddev = tf.maximum(sample_stddev, min_stddev)
    return adjusted_stddev


def get_mean_and_stddev(inputs: tf.Tensor, start_axis=1) -> Tuple[tf.Tensor, tf.Tensor]:
    sample_means = reduce_mean_from(inputs=inputs, start_axis=start_axis, keepdims=True)
    sample_stddev = reduce_adjusted_std_from(inputs=inputs, start_axis=start_axis, keepdims=True)
    return sample_means, sample_stddev


def standardize_from(inputs: tf.Tensor, start_axis=1) -> tf.Tensor:
    sample_means, sample_stddev = get_mean_and_stddev(inputs=inputs, start_axis=start_axis)
    outputs = (inputs - sample_means) / sample_stddev
    return outputs
