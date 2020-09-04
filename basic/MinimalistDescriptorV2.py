import tensorflow as tf
from typing import Dict, Tuple

from custom_tf_models.basic.MinimalistDescriptor import MinimalistDescriptor


class MinimalistDescriptorV2(MinimalistDescriptor):
    @tf.function
    def compute_contrast_loss(self,
                              base_keep_going_array: tf.Tensor,
                              noised_keep_going_array: tf.Tensor,
                              noise_factor: tf.Tensor,
                              ) -> tf.Tensor:
        length_a = tf.reduce_sum(base_keep_going_array[1:], axis=0)
        length_b = tf.reduce_sum(noised_keep_going_array[1:], axis=0)
        contrast = tf.nn.relu(length_a - length_b * noise_factor)
        return tf.reduce_mean(contrast)

    @tf.function
    def compute_intermediate_loss(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        final_residuals, keep_going_array, residuals_array = self(inputs, training=True)
        if self.stop_residual_gradients:
            reconstruction_loss = self.compute_reconstruction_loss(residuals_array, keep_going_array)
        else:
            reconstruction_loss = tf.reduce_mean(tf.square(final_residuals))
        return reconstruction_loss, keep_going_array

    @tf.function
    def compute_loss(self,
                     inputs: tf.Tensor
                     ) -> Dict[str, tf.Tensor]:
        noise_factor = self.sample_noise_factor()
        noised_inputs = self.add_noise(inputs, noise_factor)

        base_reconstruction_loss, base_keep_going_array = self.compute_intermediate_loss(inputs)
        noised_reconstruction_loss, noised_keep_going_array = self.compute_intermediate_loss(noised_inputs)
        # reconstruction_loss = base_reconstruction_loss + noised_reconstruction_loss
        reconstruction_loss = base_reconstruction_loss
        stop_loss = self.compute_stop_loss(base_keep_going_array, noise_factor=tf.constant(0.0))
        contrast_loss = self.compute_contrast_loss(base_keep_going_array, noised_keep_going_array, noise_factor)

        stop_lambda = tf.constant(self.stop_lambda, name="stop_lambda")
        loss = reconstruction_loss + stop_lambda * (contrast_loss + stop_loss)

        steps_count = tf.reduce_sum(tf.cast(tf.greater(base_keep_going_array, 0.5), tf.float32), axis=0)
        noised_steps_count = tf.reduce_sum(tf.cast(tf.greater(noised_keep_going_array, 0.5), tf.float32), axis=0)

        return {
            "loss": loss,
            "reconstruction_loss": reconstruction_loss,
            "noised_reconstruction_error": noised_reconstruction_loss,
            "stop_loss": stop_loss,
            "contrast_loss": contrast_loss,
            "steps_count": steps_count,
            "noise_additional_steps_count": noised_steps_count - steps_count,
        }
