# EBAE : Energy-based Autoencoder
import tensorflow as tf
from tensorflow.python.keras.models import Model
from typing import List, Tuple

from custom_tf_models.energy_based import EBM, EnergyStateFunction
from custom_tf_models import AE


class EnergyModel(Model):
    def __init__(self, autoencoder: AE, **kwargs):
        super(EnergyModel, self).__init__(**kwargs)
        self.autoencoder = autoencoder

    def call(self, inputs, training=None, mask=None):
        outputs = self.autoencoder(inputs)
        reduction_axis = tuple(range(1, inputs.shape.rank))
        return tf.reduce_mean(tf.square(inputs - outputs), axis=reduction_axis)

    def get_config(self):
        return {
            "autoencoder": self.autoencoder.get_config()
        }


class EBAE(EBM):
    def __init__(self,
                 encoder: Model,
                 decoder: Model,
                 energy_state_functions: List[EnergyStateFunction],
                 energy_margin: float = None,
                 learning_rate=1e-3,
                 weights_decay=2e-6,
                 seed=None,
                 **kwargs
                 ):
        energy_model = EnergyModel(autoencoder=AE(encoder=encoder, decoder=decoder))
        super(EBAE, self).__init__(energy_model=energy_model,
                                   energy_state_functions=energy_state_functions,
                                   optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                                   energy_margin=energy_margin,
                                   weights_decay=weights_decay,
                                   seed=seed,
                                   **kwargs)

    def compute_loss_for_energy(self, inputs, low_energy: bool) -> Tuple[tf.Tensor, tf.Tensor]:
        energy_states = self.get_energy_states(inputs, low_energy=low_energy)
        losses = []
        energies = []
        for state in energy_states:
            energy = self(state)
            if low_energy:
                loss = energy
            else:
                loss = tf.nn.relu(self.energy_margin - energy)

            energy -= self.energy_margin
            # TMP
            if not low_energy:
                rev_state = tf.reverse(state, axis=(1,))
                delta = tf.abs(state - rev_state)
                delta = tf.reduce_mean(delta, axis=[1, 2, 3, 4])
                mask = tf.cast(delta > 1e-2, tf.float32)
                loss *= mask
                energy *= mask

            losses.append(loss)
            energies.append(energy)

        loss = tf.reduce_mean(losses)
        energy = tf.reduce_mean(tf.stack(energies, axis=0), axis=0)
        return loss, energy

    @property
    def autoencoder(self):
        return self.energy_model.autoencoder
