import tensorflow as tf
from tensorflow.python.keras import Model
from typing import Dict, Tuple

from custom_tf_models import AE
from misc_utils.general import expand_dims_to_rank


class RDL(AE):
    def __init__(self,
                 encoder: Model,
                 decoder: Model,
                 use_noise=True,
                 noise_stddev=0.1,
                 reconstruct_noise=False,
                 **kwargs):
        super(RDL, self).__init__(encoder=encoder,
                                  decoder=decoder,
                                  **kwargs)

        self.use_noise = use_noise
        self.noise_stddev = noise_stddev
        self.reconstruct_noise = reconstruct_noise

    def compute_loss(self,
                     inputs
                     ) -> Dict[str, tf.Tensor]:
        inputs, target, noise_factor = self.add_training_noise(inputs)

        encoded = self.encoder(inputs)
        random_description_mask = self.get_random_description_mask(encoded)
        outputs = self.decoder(encoded * random_description_mask)

        loss = self.reconstruction_loss(target, outputs)
        metrics = {"loss": loss}
        return metrics

    @tf.function
    def get_random_description_mask(self, encoded: tf.Tensor) -> tf.Tensor:
        encoded_shape = tf.shape(encoded)
        batch_size = encoded_shape[0]
        code_size = encoded_shape[-1]

        random_description_length = tf.random.uniform(shape=[batch_size], minval=0, maxval=code_size, dtype=tf.int32)
        random_description_length = tf.expand_dims(random_description_length, axis=1)
        thresholds = tf.range(start=0, limit=code_size, dtype=tf.int32)
        thresholds = tf.expand_dims(thresholds, axis=0)

        random_description_mask = random_description_length >= thresholds
        mask_shape = [batch_size] + [1] * (encoded.shape.rank - 2) + [code_size]
        random_description_mask = tf.reshape(random_description_mask, mask_shape)
        random_description_mask = tf.cast(random_description_mask, tf.float32)

        return random_description_mask

    @tf.function
    def add_training_noise(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        batch_size = tf.shape(inputs)[0]
        if self.use_noise:
            noise_factor = tf.random.uniform([batch_size], maxval=1.0)
            noise = tf.random.normal(tf.shape(inputs), stddev=self.noise_stddev)
            noisy_inputs = inputs + noise * expand_dims_to_rank(noise_factor, inputs)

            target = noisy_inputs if self.reconstruct_noise else inputs
            inputs = noisy_inputs
        else:
            noise_factor = tf.zeros([batch_size], dtype=tf.float32)
            target = inputs

        return inputs, target, noise_factor

    @tf.function
    def reconstruction_loss(self, inputs: tf.Tensor, outputs: tf.Tensor) -> tf.Tensor:
        return tf.reduce_mean(tf.square(inputs - outputs))
