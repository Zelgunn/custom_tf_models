import tensorflow as tf
from tensorflow.python.keras import Model
from typing import Dict, Any

from custom_tf_models import AE
from misc_utils.math_utils import lerp


class MinimalistDescriptorV4(AE):
    def __init__(self,
                 encoder: Model,
                 decoder: Model,
                 learning_rate,
                 features_per_block: int = 1,
                 patience: int = 10,
                 trained_blocks_count: int = 1,
                 **kwargs
                 ):
        super(MinimalistDescriptorV4, self).__init__(encoder=encoder,
                                                     decoder=decoder,
                                                     learning_rate=learning_rate,
                                                     **kwargs)
        self.features_per_block = features_per_block
        self.patience = patience
        self.trained_blocks_count = trained_blocks_count

        self.current_block_index = tf.Variable(0, trainable=False, dtype=tf.int32, name="CurrentBlockIndex")
        self.patience_counter = tf.Variable(0, trainable=False, dtype=tf.int32, name="PatienceCounter")
        self.lowest_loss = tf.Variable(-1.0, trainable=False, dtype=tf.float32, name="LowestLoss")

    @tf.function
    def autoencode(self, inputs):
        encoded = self.encode(inputs)
        encoded = self.only_keep_current_block_gradients(encoded)
        decoded = self.decode(encoded)
        return decoded

    @tf.function
    def train_step(self, data) -> Dict[str, tf.Tensor]:
        metrics = super(MinimalistDescriptorV4, self).train_step(data)
        self.update_patience(metrics["loss"])
        metrics["current_block_index"] = self.current_block_index.read_value()
        metrics["patience_counter"] = self.patience_counter.read_value()
        metrics["lowest_loss"] = self.lowest_loss.read_value()
        return metrics

    @tf.function
    def only_keep_current_block_gradients(self, encoded: tf.Tensor) -> tf.Tensor:
        features_count = tf.shape(encoded)[-1]
        block_count = features_count // self.features_per_block
        max_block_index = block_count + 1 - self.trained_blocks_count

        start_block_index = tf.math.mod(self.current_block_index, max_block_index)
        end_block_index = start_block_index + self.trained_blocks_count

        start = start_block_index * self.features_per_block
        end = end_block_index * self.features_per_block

        encoded_no_grad = tf.stop_gradient(encoded)
        previous_blocks = encoded_no_grad[..., :start]
        current_blocks = encoded[..., start:end]
        next_blocks = encoded_no_grad[..., end:]

        mask_ones = tf.ones(shape=[1] * (encoded.shape.rank - 1) + [end], dtype=tf.float32)
        mask_zeros = tf.zeros(shape=[1] * (encoded.shape.rank - 1) + [features_count - end], dtype=tf.float32)
        mask = tf.concat([mask_ones, mask_zeros], axis=-1)

        encoded = tf.concat([previous_blocks, current_blocks, next_blocks], axis=-1)
        encoded *= mask
        return encoded

    @tf.function
    def update_patience(self, loss: tf.Tensor):
        lowest_loss_initialized = tf.greater(self.lowest_loss, 0.0)
        lowest_loss_initialized = tf.cast(lowest_loss_initialized, tf.float32)
        lowest_loss = lerp(loss, self.lowest_loss, lowest_loss_initialized)

        loss_not_decreased = tf.greater(loss, lowest_loss)
        loss_not_decreased = tf.cast(loss_not_decreased, tf.int32)

        new_patience_counter = (self.patience_counter + loss_not_decreased) * loss_not_decreased
        patience_exceeded = new_patience_counter >= self.patience
        patience_exceeded = tf.cast(patience_exceeded, tf.int32)
        new_patience_counter *= (1 - patience_exceeded)
        lowest_loss = lerp(lowest_loss, -1.0, tf.cast(patience_exceeded, tf.float32))

        self.current_block_index.assign_add(patience_exceeded)
        self.patience_counter.assign(new_patience_counter)
        self.lowest_loss.assign(tf.minimum(lowest_loss, loss))

    def get_config(self) -> Dict[str, Any]:
        base_config = super(MinimalistDescriptorV4, self).get_config()
        return {
            **base_config,
            "features_per_block": self.features_per_block,
            "patience": self.patience,
            "trained_blocks_count": self.trained_blocks_count,
        }
