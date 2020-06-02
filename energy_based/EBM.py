# EBAE : Energy-based Model
import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2
from typing import Dict, Tuple, Union, List

from custom_tf_models import CustomModel
from custom_tf_models.energy_based import EnergyStateFunction, InputsTensor


class EBM(CustomModel):
    def __init__(self,
                 energy_model: Model,
                 energy_state_functions: List[EnergyStateFunction],
                 optimizer: tf.keras.optimizers.Optimizer,
                 energy_margin: float = None,
                 energy_model_uses_ground_truth=False,
                 seed=None,
                 **kwargs
                 ):
        super(EBM, self).__init__(**kwargs)
        self.energy_model = energy_model
        self.low_energy_state_functions = [esf for esf in energy_state_functions if esf.is_low_energy]
        self.high_energy_state_functions = [esf for esf in energy_state_functions if not esf.is_low_energy]
        self.energy_margin = energy_margin
        self.energy_model_uses_ground_truth = energy_model_uses_ground_truth
        self.seed = seed

        self.optimizer = optimizer

    def __call__(self, *args, **kwargs):
        sum_energies = kwargs.pop("sum_energies") if "sum_energies" in kwargs else True
        energies = super(EBM, self).__call__(*args, **kwargs)

        if sum_energies and isinstance(energies, (tuple, list)):
            energies = tf.reduce_sum(tf.stack(energies, axis=0), axis=0)

        return energies

    def call(self, inputs, training=None, mask=None) -> Union[tf.Tensor, List[tf.Tensor]]:
        energies = self.energy_model(inputs)
        return energies

    @tf.function
    def train_step(self, inputs, *args, **kwargs) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        with tf.GradientTape() as tape:
            losses = self.compute_loss(inputs, *args, **kwargs)
            total_loss = losses[0]

        gradients = tape.gradient(total_loss, self.trainable_variables)
        # clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)

        grad_norms = []
        for grad in gradients:
            grad_norms.append(tf.reduce_mean(tf.abs(grad)))
        tf.print("Grad norm : ", tf.reduce_mean(grad_norms))

        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return losses

    @tf.function
    def _compute_loss(self, inputs):
        low_energy_loss = self.compute_loss_for_energy(inputs, low_energy=True)
        high_energy_loss = self.compute_loss_for_energy(inputs, low_energy=False)
        weight_decay_loss = self.weights_decay_loss(l1=2e-6)

        total_loss = low_energy_loss + high_energy_loss
        total_loss += weight_decay_loss
        return total_loss, low_energy_loss, high_energy_loss

    @tf.function
    def compute_loss(self, inputs, *args, **kwargs):
        low_state = self.get_energy_states(inputs, True)[0]
        high_state = self.get_energy_states(inputs, False)[0]

        low_audio, low_video = low_state
        high_audio, high_video = high_state

        batch_size = tf.shape(low_audio)[0]
        labels = tf.random.uniform(shape=[batch_size, 1], maxval=1.0, seed=self.seed) > 0.5

        audio = tf.where(tf.reshape(labels, [batch_size, 1, 1]),
                         high_audio, low_audio)
        video = tf.where(tf.reshape(labels, [batch_size, 1, 1, 1, 1]),
                         high_video, low_video)

        inputs = [audio, video]
        logits = self.energy_model(inputs)
        labels = tf.cast(labels, tf.float32)

        logits_dim = tf.shape(logits)[-1]
        labels = tf.tile(labels, multiples=[1, logits_dim])

        error = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        accuracy = tf.keras.metrics.binary_accuracy(labels, tf.sigmoid(logits))
        accuracy = tf.reduce_mean(accuracy)

        weight_decay = self.weights_decay_loss(l2=1e-5)
        tf.print("Weight decay : ", weight_decay)

        loss = error + weight_decay
        # loss = error

        return loss, error, accuracy

    def compute_loss_for_energy(self, inputs, low_energy: bool) -> tf.Tensor:
        energy_states = self.get_energy_states(inputs, low_energy=low_energy)

        losses = []
        for state in energy_states:
            energy = self(state, sum_energies=True)

            # loss = energy
            # if self.energy_margin is None:
            #     if not low_energy:
            #         loss = -loss
            # else:
            #     # margin_noise = tf.abs(tf.random.normal(tf.shape(loss), mean=1.0, stddev=0.05, seed=self.seed))
            #     margin_noise = 1.0
            #     if not low_energy:
            #         loss = tf.nn.relu(self.energy_margin * margin_noise - loss)
            #     else:
            #         loss = tf.nn.relu(self.energy_margin * margin_noise + loss)

            if low_energy:
                labels = tf.zeros_like(energy, energy.dtype)
            else:
                labels = tf.ones_like(energy, energy.dtype)

            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=energy)

            # loss += tf.nn.relu(self.energy_margin - tf.abs(energy))
            # loss = tf.square(loss)

            losses.append(loss)

        loss = tf.reduce_mean(tf.stack(losses, axis=0), axis=0)
        return loss

    def get_energy_states(self, inputs, low_energy: bool):
        if low_energy:
            energy_states = self._get_low_energy_states(inputs)
        else:
            energy_states = self._get_high_energy_states(inputs)

        if not self.energy_model_uses_ground_truth:
            energy_states = [state[0] for state in energy_states]

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
    def models_ids(self) -> Dict[Model, str]:
        return {self.energy_model: self.energy_model.name}

    @property
    def optimizers_ids(self) -> Dict[OptimizerV2, str]:
        return {
            self.optimizer: "optimizer",
        }

    @property
    def metrics_names(self):
        return ["total_loss", "error", "accuracy"]

    def get_config(self):
        return {
            "energy_model": self.energy_model.get_config(),
            "low_energy_functions": [str(func) for func in self.low_energy_state_functions],
            "high_energy_functions": [str(func) for func in self.high_energy_state_functions],
            "optimizer": self.optimizer.get_config(),
            "energy_margin": self.energy_margin,
            "energy_model_uses_ground_truth": self.energy_model_uses_ground_truth,
            "seed": self.seed,
        }
