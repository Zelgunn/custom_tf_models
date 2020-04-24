import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2
from typing import Dict, Tuple

from custom_tf_models import CustomModel, AE


class EBL3(CustomModel):
    def __init__(self,
                 audio_autoencoder: AE,
                 video_autoencoder: AE,
                 fusion_autoencoder: AE,
                 optimizer: OptimizerV2,
                 energy_margin: float,
                 **kwargs
                 ):
        super(EBL3, self).__init__(**kwargs)

        self.audio_autoencoder = audio_autoencoder
        self.video_autoencoder = video_autoencoder
        self.fusion_autoencoder = fusion_autoencoder
        self.optimizer = optimizer
        self.energy_margin = energy_margin

    @tf.function
    def train_step(self, inputs, *args, **kwargs) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        with tf.GradientTape() as tape:
            losses = self.compute_loss(inputs)
            loss = losses[0]

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return losses

    @tf.function
    def compute_loss(self, inputs, *args, **kwargs) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        audio, video = inputs

        right_audio = audio
        wrong_audio = tf.reverse(audio, axis=(0,))

        low_energy = self.compute_energy((right_audio, video))
        high_energy = self.compute_energy((wrong_audio, video))
        high_energy = tf.nn.relu(self.energy_margin - high_energy)

        weight_decay = self.weights_decay_loss(l1=1e-7)

        loss = low_energy + high_energy + weight_decay
        return loss, low_energy, high_energy, weight_decay

    def compute_energy(self, inputs):
        outputs = self.autoencode(inputs)

        audio_inputs, video_inputs = inputs
        audio_outputs, video_outputs = outputs

        audio_energy = tf.reduce_mean(tf.square(audio_inputs - audio_outputs))
        video_energy = tf.reduce_mean(tf.square(video_inputs - video_outputs))

        return audio_energy + video_energy

    def autoencode(self, inputs):
        audio, video = inputs

        audio_latent_code = self.audio_autoencoder.encode(audio)
        video_latent_code = self.video_autoencoder.encode(video)

        latent_codes = audio_latent_code, video_latent_code
        latent_codes = self.fusion_autoencoder(latent_codes)
        audio_latent_code, video_latent_code = latent_codes

        audio = self.audio_autoencoder.decode(audio_latent_code)
        video = self.video_autoencoder.decode(video_latent_code)

        return audio, video

    @property
    def metrics_names(self):
        return ["loss", "low_energy", "high_energy", "weight_decay"]

    @property
    def models_ids(self) -> Dict[Model, str]:
        return {
            self.audio_autoencoder: "audio_autoencoder",
            self.video_autoencoder: "video_autoencoder",
            self.fusion_autoencoder: "fusion_autoencoder",
        }

    @property
    def trainable_variables(self):
        return (
                self.audio_autoencoder.trainable_variables +
                self.video_autoencoder.trainable_variables +
                self.fusion_autoencoder.trainable_variables
        )

    @property
    def optimizers_ids(self) -> Dict[OptimizerV2, str]:
        return {
            self.optimizer: "optimizer",
        }

    def get_config(self):
        return {
            "audio_autoencoder": self.audio_autoencoder.get_config(),
            "video_autoencoder": self.video_autoencoder.get_config(),
            "fusion_autoencoder": self.fusion_autoencoder.get_config(),
            "energy_margin": self.energy_margin,
        }
