# MMAE : Multi-modal Autoencoder
import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2
from typing import List, Dict, Tuple

from custom_tf_models import CustomModel, AE


class MMAE(CustomModel):
    def __init__(self,
                 autoencoders: List[AE],
                 fusion_model: Model,
                 concatenate_latent_codes=False,
                 learning_rate=1e-3,
                 **kwargs):
        super(MMAE, self).__init__(**kwargs)

        self.autoencoders = autoencoders
        self.fusion_model = fusion_model
        self.concatenate_latent_codes = concatenate_latent_codes

        self.optimizer = None
        self.set_optimizer(tf.keras.optimizers.Adam(learning_rate=learning_rate))

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
    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            losses = self.compute_loss(inputs)
            total_loss = losses[0]

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return losses

    def compute_loss(self, inputs, *args, **kwargs) -> Tuple:
        if self.modality_count == 2:
            input_1, input_2 = inputs
            return self.compute_loss_for_two(input_1, input_2)
        else:
            return self.compute_loss_unoptimized(inputs)

    @tf.function
    def compute_loss_for_two(self, input_1, input_2) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        output_1, output_2 = self([input_1, input_2])
        loss_1 = self.compute_modality_loss(input_1, output_1)
        loss_2 = self.compute_modality_loss(input_2, output_2)
        total_loss = loss_1 + loss_2
        return total_loss, loss_1, loss_2

    def compute_loss_unoptimized(self, inputs) -> Tuple[tf.Tensor, ...]:
        outputs = self(inputs)

        losses = []
        for i in range(self.modality_count):
            modality_loss: tf.Tensor = self.compute_modality_loss(inputs, outputs)
            losses.append(modality_loss)

        total_loss = tf.reduce_sum(losses)
        return (total_loss, *losses)

    @tf.function
    def compute_modality_loss(self, inputs, outputs):
        return tf.reduce_mean(tf.square(inputs - outputs))

    @property
    def modality_count(self):
        return len(self.autoencoders)

    @property
    def metrics_names(self):
        return ["reconstruction"] + [ae.name for ae in self.autoencoders]

    def get_config(self):
        config = {ae.name: ae.get_config() for ae in self.autoencoders}
        return config

    @property
    def models_ids(self) -> Dict[Model, str]:
        ids = {ae: ae.name for ae in self.autoencoders}
        ids[self.fusion_model] = self.fusion_model.name
        return ids

    @property
    def optimizers_ids(self) -> Dict[OptimizerV2, str]:
        return {
            self.optimizer: "optimizer",
        }

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
        for ae in self.autoencoders:
            ae.set_optimizer(optimizer)
