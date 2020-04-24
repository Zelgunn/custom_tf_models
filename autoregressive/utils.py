import tensorflow as tf
from tensorflow.python.keras import backend


def compute_latent_code_regression_loss(true_latent_code,
                                        pred_latent_code_distribution
                                        ):
    z_predicted_distribution_shape = tf.shape(pred_latent_code_distribution)
    batch_size = z_predicted_distribution_shape[0]
    n_bins = z_predicted_distribution_shape[-1]

    z_predicted_distribution = tf.nn.softmax(pred_latent_code_distribution, axis=1)
    z_predicted_distribution = tf.reshape(z_predicted_distribution, [batch_size, -1, n_bins])
    epsilon = backend.epsilon()
    z_predicted_distribution = tf.clip_by_value(z_predicted_distribution, epsilon, 1.0 - epsilon)
    z_predicted_distribution = tf.math.log(z_predicted_distribution)

    z = tf.reshape(true_latent_code, [batch_size, -1, 1])
    n_bins = tf.cast(n_bins, tf.float32)
    z = tf.clip_by_value(z * n_bins, 0.0, n_bins - 1.0)
    z = tf.cast(z, tf.int32)

    selected_bins = tf.gather(z_predicted_distribution, indices=z, batch_dims=-1)

    z_loss = - tf.reduce_mean(selected_bins)

    return z_loss
