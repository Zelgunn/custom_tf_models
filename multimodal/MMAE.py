# MMAE : Multi-modal Autoencoder
import tensorflow as tf
from tensorflow.python.keras import Model
from typing import List, Dict

from custom_tf_models.basic.AE import AE


class MMAE(Model):
    def __init__(self,
                 autoencoders: List[AE],
                 fusion_model: Model,
                 concatenate_latent_codes=False,
                 **kwargs):
        super(MMAE, self).__init__(**kwargs)

        self.autoencoders = autoencoders
        self.fusion_model = fusion_model
        self.concatenate_latent_codes = concatenate_latent_codes

        self.optimizer = None

    def call(self, inputs, training=None, mask=None):
        latent_codes = []
        for i in range(self.modality_count):
            latent_code = self.autoencoders[i].encode(inputs[i])
            latent_codes.append(latent_code)

        if self.concatenate_latent_codes:
            latent_code_sizes = [code.shape[-1] for code in latent_codes]
            latent_codes = tf.concat(latent_codes, axis=-1)
            refined_latent_codes = self.fusion_model(latent_codes)
            refined_latent_codes = tf.split(refined_latent_codes, num_or_size_splits=latent_code_sizes, axis=-1)
        else:
            refined_latent_codes = self.fusion_model(latent_codes)

        outputs = []
        for i in range(self.modality_count):
            output = self.autoencoders[i].decode(refined_latent_codes[i])
            output = tf.reshape(output, tf.shape(inputs[i]))
            outputs.append(output)

        return outputs

    @tf.function
    def train_step(self, inputs) -> Dict[str, tf.Tensor]:
        with tf.GradientTape() as tape:
            metrics = self.compute_loss(inputs)
            loss = metrics["loss"]

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return metrics

    @tf.function
    def test_step(self, inputs) -> Dict[str, tf.Tensor]:
        return self.compute_loss(inputs)

    # region Loss
    @tf.function
    def compute_loss(self, inputs) -> Dict[str, tf.Tensor]:
        if self.modality_count == 2:
            input_1, input_2 = inputs
            return self.compute_loss_for_two(input_1, input_2)
        else:
            return self.compute_loss_unoptimized(inputs)

    @tf.function
    def compute_loss_for_two(self, input_1, input_2) -> Dict[str, tf.Tensor]:
        output_1, output_2 = self([input_1, input_2])
        loss_1 = self.compute_modality_loss(input_1, output_1)
        loss_2 = self.compute_modality_loss(input_2, output_2)
        total_loss = loss_1 + loss_2
        return {
            "loss": total_loss,
            "modality_1_loss": loss_1,
            "modality_2_loss": loss_2,
        }

    def compute_loss_unoptimized(self, inputs) -> Dict[str, tf.Tensor]:
        outputs = self(inputs)

        losses = []
        for i in range(self.modality_count):
            modality_loss: tf.Tensor = self.compute_modality_loss(inputs, outputs)
            losses.append(modality_loss)

        total_loss = tf.reduce_sum(losses)
        return {
            "loss": total_loss,
            **{"modality_{}_loss".format(i + 1): losses[i] for i in range(self.modality_count)}
        }

    @tf.function
    def compute_modality_loss(self, inputs, outputs) -> tf.Tensor:
        return tf.reduce_mean(tf.square(inputs - outputs))

    # endregion

    # region Properties
    @property
    def modality_count(self) -> int:
        return len(self.autoencoders)

    # endregion

    # region Config
    def get_config(self):
        config = {ae.name: ae.get_config() for ae in self.autoencoders}
        config["concatenate_latent_codes"] = self.concatenate_latent_codes
        return config
    # endregion
