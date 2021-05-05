import tensorflow as tf
from tensorflow_probability.python.stats import percentile
from tensorflow.python.keras.models import Model
from typing import Dict

from custom_tf_models.basic.AE import AE
from misc_utils.math_utils import reduce_mean_from


# CnC : Clear and Concise
class CnC(AE):
    def __init__(self,
                 encoder: Model,
                 relevance_estimator: Model,
                 decoder: Model,
                 relevance_loss_weight=1e-2,
                 skip_loss_weight=1e0,
                 energy_margin=1.0,
                 theta=0.9,
                 **kwargs
                 ):
        super(CnC, self).__init__(encoder=encoder,
                                  decoder=decoder,
                                  **kwargs)
        self.relevance_estimator = relevance_estimator
        self.relevance_loss_weight = relevance_loss_weight
        self.skip_loss_weight = skip_loss_weight
        self.energy_margin = energy_margin
        self.theta = theta

        self._relevance_loss_weight = tf.constant(relevance_loss_weight, dtype=tf.float32, name="relevance_loss_weight")
        self._skip_loss_weight = tf.constant(skip_loss_weight, dtype=tf.float32, name="skip_loss_weight")
        self._energy_margin = tf.constant(energy_margin, dtype=tf.float32, name="energy_margin")

        self.train_step_index = tf.Variable(initial_value=0, trainable=False, name="train_step_index", dtype=tf.int32)
        self.tmp_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1.0, decay_steps=1000,
                                                                           decay_rate=0.5)

    # region Inference
    @tf.function
    def get_relevance_energy(self, inputs: tf.Tensor) -> tf.Tensor:
        return self.relevance_estimator(inputs)

    @tf.function
    def get_relevance_total_energy(self, inputs: tf.Tensor) -> tf.Tensor:
        relevance_energy = self.get_relevance_energy(inputs)
        return tf.reduce_sum(relevance_energy, axis=-1)

    @tf.function
    def relevance_energy_to_prob(self, energy: tf.Tensor, code_size: tf.Tensor) -> tf.Tensor:
        # noinspection PyTypeChecker
        offset: tf.Tensor = (code_size - 1.0) / 2.0
        thresholds = tf.range(start=0, limit=code_size, dtype=tf.float32) - offset

        code_rank = energy.shape.rank - 1
        thresholds_shape = [*[1] * code_rank, code_size]
        thresholds = tf.reshape(thresholds, thresholds_shape)

        energy = tf.reduce_sum(energy, axis=-1, keepdims=True)
        return tf.nn.sigmoid(energy - thresholds)

    """
            This function returns a random mask, where the probability of activation depends on the `description_energy`
            parameter.
            params:
                `description_energy` : Assumed to be a tensor of only negative values (or zeros).
            returns:
                A tensor, with same shape and dtype as `description_energy` 
    """

    @tf.function
    def sample_relevance_mask(self, relevance_prob: tf.Tensor) -> tf.Tensor:
        epsilon = tf.random.uniform(shape=tf.shape(relevance_prob), minval=0.0, maxval=1.0)
        kept = relevance_prob >= epsilon
        keep_noise = tf.cast(kept, tf.float32) - tf.stop_gradient(relevance_prob)
        relevance_mask = relevance_prob + keep_noise
        return relevance_mask

    @tf.function
    def encode(self, inputs: tf.Tensor):
        encoded: tf.Tensor = self.encoder(inputs)
        code_size = encoded.shape[-1]

        relevance_energy = self.get_relevance_energy(inputs)
        relevance_prob = self.relevance_energy_to_prob(relevance_energy, code_size)
        relevance_mask = self.sample_relevance_mask(relevance_prob)

        encoded *= relevance_mask
        return encoded

    # endregion

    def train_step(self, inputs) -> Dict[str, tf.Tensor]:
        metrics = super(CnC, self).train_step(inputs)
        self.train_step_index.assign_add(1)
        self._train_counter.assign(tf.cast(self.train_step_index, tf.int64))
        return metrics

    @tf.function
    def compute_loss(self,
                     inputs
                     ) -> Dict[str, tf.Tensor]:
        encoded = self.encoder(inputs)
        code_size = encoded.shape[-1]

        skip_decoded = self.decoder(encoded)
        skip_error = reduce_mean_from(tf.abs(inputs - skip_decoded), start_axis=1)
        skip_loss = tf.reduce_mean(skip_error)

        relevance_energy = self.get_relevance_energy(inputs)
        relevance_prob = self.relevance_energy_to_prob(relevance_energy, code_size)
        relevance_mask = self.sample_relevance_mask(relevance_prob)
        # relevance_mask = tf.stop_gradient(relevance_mask)
        decoded = self.decoder(encoded * relevance_mask)

        reconstruction_error = reduce_mean_from(tf.abs(inputs - decoded), start_axis=1)
        reconstruction_loss = tf.reduce_mean(reconstruction_error)
        # reconstruction_loss = tf.reduce_mean(reconstruction_error)
        relevance_loss = self.compute_relevance_loss(relevance_energy, reconstruction_error)

        # step = tf.cast(self.train_step_index, dtype=tf.float32)
        # relevance_loss_weight = self._relevance_loss_weight * ((1.0 - tf.exp(-step / 10000.0)) * 0.9 + 0.1)
        relevance_loss_weight = self._relevance_loss_weight
        # tmp_weight = self.tmp_schedule(self.train_step_index)
        loss = (reconstruction_loss + skip_loss * self._skip_loss_weight) + \
               relevance_loss * relevance_loss_weight

        mean_relevance_prob = tf.reduce_mean(relevance_prob)
        mean_code_length = tf.reduce_mean(relevance_mask)
        # mean_reconstruction_error = tf.reduce_mean(reconstruction_error)
        mean_skip_error = tf.reduce_mean(skip_error)

        metrics = {
            "loss": loss,
            # "reconstruction_error": mean_reconstruction_error,
            "reconstruction_loss": reconstruction_loss,
            "skip_error": mean_skip_error,
            "skip_loss": skip_loss,
            "relevance_loss": relevance_loss,
            "relevance_prob": mean_relevance_prob,
            "code_length": mean_code_length,
            "total_energy": tf.reduce_mean(self.get_relevance_total_energy(inputs)),
            # "relevance_loss_weight": relevance_loss_weight,
        }

        return metrics

    @tf.function
    def compute_relevance_loss(self, relevance_energy: tf.Tensor, reconstruction_error: tf.Tensor) -> tf.Tensor:
        # target = tf.reduce_mean(reconstruction_error)
        target = percentile(reconstruction_error, q=self.theta * 100)
        below_target = reconstruction_error <= target
        below_target = tf.cast(below_target, dtype=tf.float32)

        pull_norm = 1.0 / (self.theta * 2.0)
        pull_up_weight = (self.theta / (1.0 - self.theta)) * pull_norm
        pull_down_weight = pull_norm

        # if below error -> aim for shorter code (positive loss weight)
        # if above error -> aim for better reconstruction (negative loss weight)
        pull_weights: tf.Tensor = below_target * pull_down_weight + (1.0 - below_target) * pull_up_weight
        pull_direction: tf.Tensor = below_target * 2.0 - 1.0

        batch_size = tf.shape(pull_direction)[0]
        delta_rank = relevance_energy.shape.rank - 1
        pull_direction_shape = [batch_size, *[1] * delta_rank]
        pull_direction = tf.reshape(pull_direction, pull_direction_shape)

        relevance_loss = tf.nn.relu(self._energy_margin + relevance_energy * pull_direction) - self._energy_margin
        relevance_loss = reduce_mean_from(relevance_loss, start_axis=1)
        relevance_loss = tf.reduce_mean(relevance_loss * pull_weights)
        # relevance_loss = tf.reduce_mean(relevance_prob)
        return relevance_loss

    @tf.function
    def mean_relevance_energy(self, inputs: tf.Tensor) -> tf.Tensor:
        return reduce_mean_from(self.get_relevance_total_energy(inputs), start_axis=1)

    def get_config(self):
        base_config = super(CnC, self).get_config()
        config = {
            **base_config,
            "relevance_estimator": self.relevance_estimator,
            "relevance_loss_weight": self.relevance_loss_weight,
            "skip_loss_weight": self.skip_loss_weight,
            "energy_margin": self.energy_margin,
            "theta": self.theta,
        }
        return config
