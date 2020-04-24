# SparseAE : Sparse Autoencoder
import tensorflow as tf
from tensorflow.python.keras import Model

from custom_tf_models import AE


class SparseAE(AE):
    def __init__(self,
                 encoder: Model,
                 decoder: Model,
                 learning_rate=1e-3,
                 activity_regularization_factor=1e-5,
                 **kwargs):
        super(SparseAE, self).__init__(encoder=encoder,
                                       decoder=decoder,
                                       learning_rate=learning_rate,
                                       **kwargs)
        self.activity_regularization_factor = activity_regularization_factor

    @tf.function
    def compute_loss(self,
                     inputs
                     ) -> tf.Tensor:
        encoded = self.encode(inputs)
        decoded = self.decode(encoded)

        batch_size = tf.shape(inputs)[0]
        activity_regularization = tf.reduce_sum(encoded) / batch_size * self.activity_regularization_factor

        reconstruction_error = tf.reduce_mean(tf.square(inputs - decoded))

        return reconstruction_error + activity_regularization

    def get_config(self):
        config = {
            "encoder": self.encoder.get_config(),
            "decoder": self.decoder.get_config(),
            "learning_rate": self.learning_rate,
            "activity_regularization_factor": self.activity_regularization_factor
        }
        return config
