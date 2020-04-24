import tensorflow as tf
from tensorflow.python.keras import Model
from typing import List, Tuple

from energy_based import EBM, EnergyStateFunction


class JEM(EBM):
    def __init__(self,
                 energy_model: Model,
                 energy_state_functions: List[EnergyStateFunction],
                 optimizer: tf.keras.optimizers.Optimizer,
                 energy_margin: float = None,
                 buffer_size: int = 400,
                 buffer_reinitialization_frequency: float = 0.05,
                 sgld_steps: int = 20,
                 sgld_step_size: float = 1.0,
                 sgld_noise: float = 0.01,
                 inputs_range: Tuple[float, float] = (-1.0, 1.0),
                 seed=None,
                 **kwargs
                 ):
        super(JEM, self).__init__(energy_model=energy_model,
                                  energy_state_functions=energy_state_functions,
                                  optimizer=optimizer,
                                  energy_margin=energy_margin,
                                  energy_model_uses_ground_truth=False,
                                  seed=seed,
                                  **kwargs)
        self.buffer_size = buffer_size
        self.buffer_reinitialization_frequency = buffer_reinitialization_frequency
        self.sgld_steps = sgld_steps
        self.sgld_step_size = sgld_step_size
        self.sgld_noise = sgld_noise
        self.inputs_range = inputs_range

        initial_replay_buffer = self.get_empty_replay_buffer_value(buffer_size)
        self.replay_buffer = [tf.Variable(initial_value=initial_replay_buffer_part, trainable=False)
                              for initial_replay_buffer_part in initial_replay_buffer]

    def compute_loss(self, inputs, *args, **kwargs):
        low_energy_pred = self.compute_energies_for_level(inputs, low_energy=True)
        high_energy_pred = self.compute_energies_for_level(inputs, low_energy=False)

        low_energy_classification_loss = get_binary_classification_loss(low_energy_pred, False)
        high_energy_classification_loss = get_binary_classification_loss(high_energy_pred, True)
        classification_loss = low_energy_classification_loss + high_energy_classification_loss

        batch_size = tf.shape(low_energy_pred)[1]
        generated_samples = self.sample_with_sgld(batch_size)
        generated_energy = self(generated_samples, sum_energies=True)

        generation_loss = (binary_log_sum_exp(low_energy_pred) + binary_log_sum_exp(high_energy_pred)) * 0.5
        generation_loss -= binary_log_sum_exp(generated_energy)

        total_loss = classification_loss + generation_loss
        return total_loss, classification_loss, generation_loss

    def compute_energies_for_level(self, inputs, low_energy: bool):
        energy_states = self.get_energy_states(inputs, low_energy=low_energy)

        energies = []
        for state in energy_states:
            energy = self(state, sum_energies=True)
            energies.append(energy)

        energies = tf.stack(energies, axis=0)
        return energies

    def get_empty_replay_buffer_value(self, size) -> List[tf.Tensor]:
        values = []
        for i in range(self.inputs_count):
            value = self.get_empty_replay_buffer_part_value(i, size)
            values.append(value)
        return values

    def get_empty_replay_buffer_part_value(self, part_index: int, size) -> tf.Tensor:
        min_input_value, max_input_value = self.inputs_range
        input_shape = self.energy_model.input_shape[part_index][1:]
        part_shape = [size, *input_shape]
        value = tf.random.uniform(minval=min_input_value, maxval=max_input_value,
                                  shape=part_shape, dtype=tf.float32, seed=self.seed)
        return value

    def sample_with_sgld(self, samples_count: tf.Tensor):
        indices = tf.random.uniform(shape=[samples_count], minval=0, maxval=self.buffer_size,
                                    dtype=tf.int32, seed=self.seed)
        samples = self.sample_replay_buffer(indices)

        def loop_cond(i, _):
            return i < self.sgld_steps

        def loop_body(i, _samples):
            _samples = self.run_sgld_step(samples)
            i = i + 1
            return i, _samples

        loop_vars = [tf.constant(0, dtype=tf.int32), samples]
        _, samples = tf.while_loop(loop_cond, loop_body, loop_vars)

        return samples

    def run_sgld_step(self, samples: List[tf.Tensor]):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(samples)
            samples_energy = self(samples, sum_energies=True)
            loss = binary_log_sum_exp(samples_energy)

        gradients = tape.gradient(loss, samples)

        outputs_samples = []
        for i in range(self.inputs_count):
            sample = samples[i]
            noise = tf.random.normal(shape=tf.shape(sample), stddev=self.sgld_noise, seed=self.seed)
            sample += self.sgld_step_size * gradients[i] + noise
            sample = tf.stop_gradient(sample)
            outputs_samples.append(sample)
        return outputs_samples

    def sample_replay_buffer(self, indices: tf.Tensor) -> List[tf.Tensor]:
        samples_count = tf.shape(indices)[0]

        base_reinitialization_mask = tf.random.uniform([samples_count], minval=0.0, maxval=1.0, seed=self.seed)
        base_reinitialization_mask = base_reinitialization_mask < self.buffer_reinitialization_frequency

        samples = []
        for i in range(self.inputs_count):
            empty_replay_buffer = self.get_empty_replay_buffer_part_value(i, samples_count)
            selected_from_buffer = tf.gather(self.replay_buffer[i], indices, axis=0)
            mask_shape = [samples_count] + [1] * (selected_from_buffer.shape.rank - 1)
            reinitialization_mask = tf.reshape(base_reinitialization_mask, mask_shape)
            part_samples = tf.where(condition=reinitialization_mask,
                                    x=empty_replay_buffer,
                                    y=selected_from_buffer)
            samples.append(part_samples)

        return samples

    def update_replay_buffer(self, new_samples: List[tf.Tensor], indices: tf.Tensor):
        for i in range(self.inputs):
            indexed_slices = tf.IndexedSlices(values=new_samples[i], indices=indices)
            self.replay_buffer[i].scatter_update(indexed_slices)

    @property
    def inputs_count(self) -> int:
        return len(self.energy_model.inputs)

    @property
    def metrics_names(self):
        return ["total_loss", "classification", "generation"]


def get_binary_classification_loss(logits: tf.Tensor, labels: bool) -> tf.Tensor:
    if labels:
        ground_truth = tf.ones_like(logits)
    else:
        ground_truth = tf.zeros_like(logits)

    loss = tf.nn.sigmoid_cross_entropy_with_logits(ground_truth, logits)
    loss = tf.reduce_mean(loss)
    return loss


def binary_log_sum_exp(tensor: tf.Tensor) -> tf.Tensor:
    tensor = tf.exp(tensor) + tf.exp(-tensor)
    return tf.math.log(tensor)
