import tensorflow as tf
from tensorflow.python.keras import Model
from typing import Dict

from custom_tf_models.basic.AE import AE


# SparseAE : Sparse Autoencoder
class SparseAE(AE):
    def __init__(self,
                 encoder: Model,
                 decoder: Model,
                 activity_regularization_factor=1e-5,
                 **kwargs):
        super(SparseAE, self).__init__(encoder=encoder,
                                       decoder=decoder,
                                       **kwargs)
        self.activity_regularization_factor = activity_regularization_factor

    @tf.function
    def compute_loss(self,
                     inputs
                     ) -> Dict[str, tf.Tensor]:
        encoded = self.encode(inputs)
        decoded = self.decode(encoded)

        batch_size = tf.shape(inputs)[0]
        activity_regularization = tf.reduce_sum(encoded) / batch_size * self.activity_regularization_factor
        reconstruction_error = tf.reduce_mean(tf.square(inputs - decoded))
        loss = reconstruction_error + activity_regularization

        return {"loss": loss, "reconstruction": reconstruction_error, "activity": activity_regularization}

    def get_config(self):
        config = {
            **super(SparseAE, self).get_config(),
            "activity_regularization_factor": self.activity_regularization_factor
        }
        return config
