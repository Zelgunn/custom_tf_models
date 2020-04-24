# EBAE : Energy-based Autoencoder
import tensorflow as tf
from typing import Union, List

from custom_tf_models import CustomModel
from energy_based import EBM, EnergyStateFunction


class EBAE(EBM):
    def __init__(self,
                 autoencoder: CustomModel,
                 energy_state_functions: List[EnergyStateFunction],
                 energy_margin: float = None,
                 **kwargs
                 ):
        super(EBAE, self).__init__(energy_model=autoencoder,
                                   energy_state_functions=energy_state_functions,
                                   optimizer=autoencoder.optimizer,
                                   energy_margin=energy_margin,
                                   energy_model_uses_ground_truth=True,
                                   **kwargs)
        self.autoencoder = self.energy_model

    def call(self, inputs, training=None, mask=None) -> Union[tf.Tensor, List[tf.Tensor]]:
        ground_truth, inputs = inputs
        outputs = self.autoencoder(inputs)
        if isinstance(ground_truth, (tuple, list)):
            energies = [self.compute_energy(x, y) for x, y in zip(ground_truth, outputs)]
            return energies
        else:
            energy = self.compute_energy(ground_truth, outputs)
            return energy

    @tf.function
    def compute_energy(self, inputs, outputs):
        reduction_axis = tuple(range(1, inputs.shape.rank))
        return tf.reduce_mean(tf.square(inputs - outputs), axis=reduction_axis)

    def compute_loss_for_energy(self, inputs, low_energy: bool):
        energy_states = self.get_energy_states(inputs, low_energy=low_energy)

        losses = []
        for state in energy_states:
            energy = self(state, sum_energies=True)
            if not low_energy:
                energy = tf.nn.relu(self.energy_margin - energy)
            losses.append(energy)

        loss = tf.reduce_mean(tf.stack(losses, axis=0), axis=0)
        return loss

    @tf.function
    def forward(self, inputs):
        return self((inputs, inputs))

    def get_config(self):
        config = super(EBAE, self).get_config()
        config.pop("energy_model_uses_ground_truth")
        return config
