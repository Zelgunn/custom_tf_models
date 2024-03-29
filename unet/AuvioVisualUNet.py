import tensorflow as tf
from tensorflow.python.keras import Model
from typing import Dict

from custom_tf_models.unet.UNet import UNet


class AudioVideoUNet(Model):
    def __init__(self,
                 image_unet: UNet,
                 audio_unet: UNet,
                 time_unet: UNet,
                 **kwargs
                 ):
        super(AudioVideoUNet, self).__init__(**kwargs)
        self.image_unet = image_unet
        self.audio_unet = audio_unet
        self.time_unet = time_unet

    def call(self, inputs, training=None, mask=None):
        audio, video = inputs

        video_input_shape = tf.shape(video)
        batch_size, video_length, video_height, video_width, video_channels = tf.unstack(video_input_shape)
        video = tf.reshape(video, [batch_size * video_length, video_height, video_width, video_channels])

        video_encoded = self.image_unet.encode(video)
        audio_encoded = self.audio_unet.encode(audio)

        video_latent_code = video_encoded[-1]
        video_latent_code_size = video_latent_code[-1]
        video_latent_code = tf.reshape(video_latent_code, [batch_size, video_length, video_latent_code_size])

        audio_latent_code = audio_encoded[-1]
        audio_latent_code_size = audio_latent_code[-1]
        time_latent_code = tf.concat([audio_latent_code, video_latent_code], axis=-1)

        time_latent_code = self.time_unet(time_latent_code)
        splits = [audio_latent_code_size, video_latent_code_size]
        audio_latent_code, video_latent_code = tf.split(time_latent_code, num_or_size_splits=splits, axis=-1)
        video_latent_code = tf.reshape(video_latent_code, [batch_size * video_length, video_latent_code_size])

        video_encoded[-1] = video_latent_code
        audio_encoded[-1] = audio_latent_code

        video_decoded = self.image_unet.decode(video_encoded)
        video_decoded = tf.reshape(video_decoded, video_input_shape)
        audio_decoded = self.audio_unet.decode(audio_encoded)

        return audio_decoded, video_decoded

    @tf.function
    def train_step(self, inputs) -> Dict[str, tf.Tensor]:
        pass

    @tf.function
    def test_step(self, inputs) -> Dict[str, tf.Tensor]:
        return self.compute_loss(inputs)

    def compute_loss(self, inputs) -> Dict[str, tf.Tensor]:
        pass

    def get_config(self):
        return {
            "ImageUNet": self.image_unet.get_config(),
            "AudioUNet": self.audio_unet.get_config(),
            "FusionUnet": self.time_unet.get_config(),
        }
