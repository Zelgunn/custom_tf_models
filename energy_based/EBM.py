# EBM : Energy-based Model
import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2
from typing import Dict, Tuple, Union, List

# from custom_tf_models import CustomModel
from custom_tf_models.energy_based import EnergyStateFunction, InputsTensor


class EBM(Model):
    def __init__(self,
                 energy_model: Model,
                 energy_state_functions: List[EnergyStateFunction],
                 energy_margin: float = None,
                 weights_decay=2e-6,
                 seed=None,
                 **kwargs
                 ):
        super(EBM, self).__init__(**kwargs)
        self.energy_model = energy_model
        self.low_energy_state_functions = [esf for esf in energy_state_functions if esf.is_low_energy]
        self.high_energy_state_functions = [esf for esf in energy_state_functions if not esf.is_low_energy]
        self.energy_margin = energy_margin
        self.weights_decay = weights_decay
        self.seed = seed

    def call(self, inputs, training=None, mask=None) -> Union[tf.Tensor, List[tf.Tensor]]:
        energies = self.energy_model(inputs)
        if isinstance(energies, (tuple, list)):
            energies = tf.reduce_sum(tf.stack(energies, axis=0), axis=0)
        return energies

    @tf.function
    def test_step(self, data):
        loss, accuracy = self.compute_loss(data)
        return {"loss": loss, "accuracy": accuracy}

    @tf.function
    def train_step(self, data) -> Dict[str, tf.Tensor]:
        with tf.GradientTape() as tape:
            loss, accuracy = self.compute_loss(data)

        gradients = tape.gradient(loss, self.trainable_variables)
        # clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)

        # grad_norms = []
        # for grad in gradients:
        #     grad_norms.append(tf.reduce_mean(tf.abs(grad)))
        # tf.print("Grad norm : ", tf.reduce_mean(grad_norms))

        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {"loss": loss, "accuracy": accuracy}

    @tf.function
    def compute_loss(self, inputs) -> Tuple[tf.Tensor, tf.Tensor]:
        low_energy_loss, low_energy_value = self.compute_loss_for_energy(inputs, low_energy=True)
        high_energy_loss, high_energy_value = self.compute_loss_for_energy(inputs, low_energy=False)
        loss = low_energy_loss + high_energy_loss

        if (self.weights_decay is not None) and (self.weights_decay > 0.0):
            loss += self.weights_decay_loss(l1=self.weights_decay)

        low_energy_accuracy = tf.reduce_mean(tf.cast(tf.less_equal(low_energy_value, 0), tf.float32))
        high_energy_accuracy = tf.reduce_mean(tf.cast(tf.greater(high_energy_value, 0), tf.float32))
        accuracy = (low_energy_accuracy + high_energy_accuracy) * 0.5
        return loss, accuracy

    def weights_decay_loss(self, l1=0.0, l2=0.0, variables=None):
        loss = 0
        variables = self.trainable_variables if variables is None else variables
        for variable in variables:
            loss += tf.keras.regularizers.L1L2(l1=l1, l2=l2)(variable)
        return loss

    def compute_loss_for_energy(self, inputs, low_energy: bool) -> Tuple[tf.Tensor, tf.Tensor]:
        energy_states = self.get_energy_states(inputs, low_energy=low_energy)

        losses = []
        energies = []
        for state in energy_states:
            energy = self(state)

            loss = energy
            if self.energy_margin is None:
                if not low_energy:
                    loss = -loss
            else:
                if not low_energy:
                    loss = tf.nn.relu(self.energy_margin - loss)
                else:
                    loss = tf.nn.relu(self.energy_margin + loss)

            losses.append(loss)
            energies.append(energy)

        loss = tf.reduce_mean(losses)
        energy = tf.reduce_mean(tf.stack(energies, axis=0), axis=0)
        return loss, energy

    def get_energy_states(self, inputs, low_energy: bool):
        if low_energy:
            energy_states = self._get_low_energy_states(inputs)
        else:
            energy_states = self._get_high_energy_states(inputs)

        return energy_states

    @tf.function
    def _get_low_energy_states(self, inputs) -> List[Tuple[InputsTensor, InputsTensor]]:
        states = [func(inputs) for func in self.low_energy_state_functions]
        return states

    @tf.function
    def _get_high_energy_states(self, inputs) -> List[Tuple[InputsTensor, InputsTensor]]:
        states = [func(inputs) for func in self.high_energy_state_functions]
        return states

    @property
    def optimizers_ids(self) -> Dict[OptimizerV2, str]:
        return {
            self.optimizer: "optimizer",
        }

    @property
    def metrics_names(self):
        return ["loss", "accuracy"]

    def get_config(self):
        try:
            config = super(EBM, self).get_config()
        except NotImplementedError:
            config = {
                "energy_model": self.energy_model.get_config(),
                "low_energy_functions": [str(func) for func in self.low_energy_state_functions],
                "high_energy_functions": [str(func) for func in self.high_energy_state_functions],
                "optimizer": self.optimizer.get_config(),
                "energy_margin": self.energy_margin,
                "weights_decay": self.weights_decay,
                "seed": self.seed,
            }
        return config
