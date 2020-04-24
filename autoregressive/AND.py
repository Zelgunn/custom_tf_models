import tensorflow as tf
from tensorflow_core.python.keras import Model
from typing import Dict

from custom_tf_models import AE
from utils import split_steps
from CustomKerasLayers.models.ConvAM import compute_autoregression_loss


class AND(AE):
    """
    Self-Attention Temporal Auto-Encoder
    """

    def __init__(self,
                 encoder: Model,
                 decoder: Model,
                 am: Model,
                 step_size: int,
                 do_autoregression_on_latent_code=True,
                 learning_rate=1e-3,
                 **kwargs):
        super(AND, self).__init__(encoder=encoder,
                                  decoder=decoder,
                                  learning_rate=learning_rate,
                                  **kwargs)

        self.autoregressive_model = am
        self.step_size = step_size
        self.do_autoregression_on_latent_code = do_autoregression_on_latent_code

    @tf.function
    def train_step(self, inputs, *args, **kwargs):
        with tf.GradientTape() as tape:
            reconstruction_loss, autoregression_loss = self.compute_loss(inputs)
            loss = reconstruction_loss + autoregression_loss * 0.1

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        reconstruction_loss /= tf.cast(tf.reduce_prod(tf.shape(inputs)[1:]), tf.float32)
        return reconstruction_loss, autoregression_loss

    @tf.function
    def compute_loss(self, inputs, *args, **kwargs):
        inputs, _, unmerged_shape = self.split_inputs(inputs, merge_batch_and_steps=True)
        true_latent_code = self.encode(inputs)

        reconstructed = self.decode(true_latent_code)
        reduction_axis = list(range(1, inputs.shape.rank))
        reconstruction_loss = tf.square(inputs - reconstructed)
        reconstruction_loss = tf.reduce_sum(reconstruction_loss, axis=reduction_axis)
        reconstruction_loss = tf.reduce_mean(reconstruction_loss)

        true_latent_code = self.reshape_latent_code_for_autoregression(true_latent_code, step_count=unmerged_shape[1])
        pred_latent_code_distribution = self.autoregressive_model(true_latent_code)
        latent_code_regression_loss = compute_autoregression_loss(true_latent_code, pred_latent_code_distribution)
        return reconstruction_loss, latent_code_regression_loss

    def split_inputs(self, inputs, merge_batch_and_steps):
        return split_steps(inputs, self.step_size, merge_batch_and_steps)

    @tf.function
    def reshape_latent_code_for_autoregression(self, latent_code, step_count):
        base_shape = tf.shape(latent_code)
        batch_merged, *dimensions, code_size = tf.unstack(base_shape)
        batch_size = batch_merged // step_count

        if self.do_autoregression_on_latent_code:
            latent_code = tf.reshape(latent_code, [batch_size, step_count, -1, code_size])
            latent_code = tf.transpose(latent_code, [0, 2, 1, 3])
            latent_code = tf.reshape(latent_code, [-1, step_count, code_size])
        else:
            latent_code = tf.reshape(latent_code, [batch_size, step_count, -1])
            latent_code = tf.transpose(latent_code, [0, 2, 1])
            latent_code = tf.reshape(latent_code, [-1, step_count])
        return latent_code

    @property
    def models_ids(self) -> Dict[Model, str]:
        ae_models_ids = super(AND, self).models_ids
        return {
            **ae_models_ids,
            self.autoregressive_model: "autoregressive_model"
        }

    @property
    def metrics_names(self):
        return ["reconstruction", "latent_code_likelihood"]
