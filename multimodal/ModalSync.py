import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.losses import binary_crossentropy
from typing import List, Dict

from misc_utils.math_utils import reduce_mean_from


class ModalSync(Model):
    def __init__(self,
                 encoders: List[Model],
                 energy_model: Model,
                 energy_margin: float = None,
                 **kwargs
                 ):
        super(ModalSync, self).__init__(**kwargs)
        self.encoders = encoders
        self.energy_model = energy_model
        self.energy_margin = energy_margin

        if energy_margin is None:
            _energy_margin = None
        else:
            _energy_margin = tf.constant(energy_margin, dtype=tf.float32, name="energy_margin")
        self._energy_margin = _energy_margin
        self.train_step_index = tf.Variable(initial_value=0, trainable=False, name="train_step_index", dtype=tf.int32)

    @property
    def modality_count(self) -> int:
        return len(self.encoders)

    def get_modality_input_length(self, modality_index: int) -> int:
        return self.encoders[modality_index].input_shape[1]

    @tf.function
    def encode(self, inputs: List[tf.Tensor], training=None, mask=None) -> tf.Tensor:
        encoded = []

        for i in range(self.modality_count):
            ith_code = self.encoders[i](inputs[i], training=training, mask=mask)
            encoded.append(ith_code)

        encoded = tf.concat(encoded, axis=-1)
        return encoded

    @tf.function
    def call(self, inputs: List[tf.Tensor], training=None, mask=None):
        encoded = self.encode(inputs)
        energy = self.energy_model(encoded, training=training, mask=mask)
        return energy

    @tf.function
    def train_step(self, inputs: List[tf.Tensor]) -> Dict[str, tf.Tensor]:
        with tf.GradientTape() as tape:
            metrics = self.compute_loss(inputs)
            loss = metrics["loss"]

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.train_step_index.assign_add(1)
        self._train_counter.assign(tf.cast(self.train_step_index, tf.int64))

        return metrics

    @tf.function
    def compute_loss(self, inputs: List[tf.Tensor]) -> Dict[str, tf.Tensor]:
        synced_inputs = self.get_synced_inputs(inputs)
        unsynced_inputs = self.get_unsynced_inputs(inputs)

        synced_energy = self(synced_inputs, training=True)
        unsynced_energy = self(unsynced_inputs, training=True)

        synced_energy = reduce_mean_from(synced_energy)
        unsynced_energy = reduce_mean_from(unsynced_energy)

        if self.energy_margin is None:
            synced_loss = binary_crossentropy(tf.zeros_like(synced_energy), synced_energy, from_logits=True)
            unsynced_loss = binary_crossentropy(tf.ones_like(unsynced_energy), unsynced_energy, from_logits=True)
            loss = tf.reduce_mean(synced_loss + unsynced_loss)
        else:
            loss = tf.nn.relu(self._energy_margin + synced_energy) + tf.nn.relu(self._energy_margin - unsynced_energy)
            loss = tf.reduce_mean(loss) - self._energy_margin * 2.0

        synced_accuracy = tf.reduce_mean(tf.cast(synced_energy <= 0.0, tf.float32))
        unsynced_accuracy = tf.reduce_mean(tf.cast(unsynced_energy > 0.0, tf.float32))
        accuracy = (synced_accuracy + unsynced_accuracy) * 0.5

        return {
            "loss": loss,
            "synced_energy": tf.reduce_mean(synced_energy),
            "unsynced_energy": tf.reduce_mean(unsynced_energy),
            "accuracy": accuracy
        }

    @tf.function
    def get_synced_inputs(self, inputs: List[tf.Tensor]) -> List[tf.Tensor]:
        synced_inputs = []
        for i in range(self.modality_count):
            modality = inputs[i]
            modality_input_length = self.get_modality_input_length(i)
            modality_slice = modality[:, :modality_input_length]
            synced_inputs.append(modality_slice)
        return synced_inputs

    @tf.function
    def get_unsynced_inputs(self, inputs: List[tf.Tensor]) -> List[tf.Tensor]:
        unsynced_index = tf.random.uniform(shape=(), minval=0, maxval=self.modality_count, dtype=tf.int32)
        unsynced_inputs = []

        for i in range(self.modality_count):
            modality = inputs[i]
            modality_length = tf.shape(modality)[1]
            modality_input_length = self.get_modality_input_length(i)
            is_unsynced = tf.cast(i == unsynced_index, tf.int32)
            start = is_unsynced * (modality_length - modality_input_length)
            end = start + modality_input_length
            modality_slice = modality[:, start:end]
            unsynced_inputs.append(modality_slice)

        return unsynced_inputs

    def get_config(self):
        return {
            "encoders": [encoder.get_config() for encoder in self.encoders],
            "energy_model": self.energy_model,
        }

    @property
    def additional_test_metrics(self):
        return [self.call]
