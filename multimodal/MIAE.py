# MIAE : Multi-modal Interpolating Autoencoder
import tensorflow as tf
from typing import List, Tuple

from custom_tf_models.basic.IAE import IAE
from custom_tf_models.multimodal import MMAE
from custom_tf_models.utils import split_steps


class MIAE(MMAE):
    @tf.function
    def compute_loss(self,
                     inputs
                     ) -> Tuple[tf.Tensor]:
        split_inputs, _, unmerged_shapes = self.split_inputs(inputs, merge_batch_and_steps=False)

        step_count = unmerged_shapes[0][1]
        offset = tf.random.uniform(shape=[], minval=0, maxval=step_count, dtype=tf.int32)

        original_latent_codes = []
        interpolated_latent_codes = []
        for i in range(self.modality_count):
            iae = self.iaes[i]
            modality = split_inputs[i]

            if offset == 0:
                latent_code = iae.encode(modality[:, 0])
                original_latent_code = latent_code
                interpolated_latent_code = latent_code

            elif offset == (step_count - 1):
                latent_code = iae.encode(modality[:, -1])
                original_latent_code = latent_code
                interpolated_latent_code = latent_code

            else:
                original_latent_code = iae.encode(modality[:, offset])
                factor = tf.cast(offset / (step_count - 1), tf.float32)
                start_encoded = iae.encode(modality[:, 0])
                end_encoded = iae.encode(modality[:, -1])
                interpolated_latent_code = factor * end_encoded + (1.0 - factor) * start_encoded

            original_latent_codes.append(original_latent_code)
            interpolated_latent_codes.append(interpolated_latent_code)

        refined_latent_codes = self.fusion_model([interpolated_latent_codes, original_latent_codes])

        losses = []
        for i in range(self.modality_count):
            output = self.iaes[i].decode(refined_latent_codes[i])
            target = split_inputs[i][:, offset]
            output = tf.reshape(output, tf.shape(target))
            modality_loss: tf.Tensor = tf.reduce_mean(tf.square(target - output))
            losses.append(modality_loss)

        return tuple(losses)

    @tf.function
    def interpolate(self, inputs):
        split_inputs, inputs_shape, _ = self.split_inputs(inputs, merge_batch_and_steps=True)
        original_latent_codes = self.encode_step(split_inputs)
        interpolated_latent_codes = self.get_interpolated_latent_code(inputs, merge_batch_and_steps=True)
        refined_latent_codes = self.fusion_model([interpolated_latent_codes, original_latent_codes])
        outputs = self.decode_step(refined_latent_codes)
        outputs = [tf.reshape(decoded, input_shape) for decoded, input_shape in zip(outputs, inputs_shape)]
        return outputs

    @tf.function
    def encode_step(self, inputs):
        latent_codes = []
        for i in range(self.modality_count):
            latent_code = self.iaes[i].encode(inputs[i])
            latent_codes.append(latent_code)
        return latent_codes

    @tf.function
    def decode_step(self, inputs):
        outputs = []
        for i in range(self.modality_count):
            decoded = self.iaes[i].decode(inputs[i])
            outputs.append(decoded)
        return outputs

    def get_interpolated_latent_code(self, inputs, merge_batch_and_steps):
        latent_codes = []
        for i in range(self.modality_count):
            latent_code = self.iaes[i].get_interpolated_latent_code(inputs[i], merge_batch_and_steps)
            latent_codes.append(latent_code)
        return latent_codes

    @tf.function
    def modalities_mse(self, inputs, ground_truths):
        errors = []
        for i in range(self.modality_count):
            error = tf.square(inputs[i] - ground_truths[i])
            errors.append(error)

        errors, _, _ = self.split_inputs(errors, merge_batch_and_steps=False)

        total_error = []
        factors = [1.0, 8.0]
        for i in range(self.modality_count):
            error = errors[i]
            reduction_axis = list(range(2, error.shape.rank))
            error = tf.reduce_mean(error, axis=reduction_axis) * factors[i]
            total_error.append(error)

        total_error = tf.reduce_sum(total_error, axis=0)
        return total_error

    def split_inputs(self, inputs, merge_batch_and_steps):
        split_inputs = []
        inputs_shapes = []
        new_shapes = []

        for i in range(self.modality_count):
            split_input, inputs_shape, new_shape = split_steps(inputs[i], self.step_sizes[i], merge_batch_and_steps)
            split_inputs.append(split_input)
            inputs_shapes.append(inputs_shape)
            new_shapes.append(new_shape)

        return split_inputs, inputs_shapes, new_shapes

    @property
    def iaes(self) -> List[IAE]:
        return self.autoencoders

    @property
    def step_sizes(self) -> List[int]:
        return [iae.step_size for iae in self.iaes]
