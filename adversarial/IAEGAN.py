import tensorflow as tf
from tensorflow.python.keras import Model
from typing import Tuple, Dict, Any

from custom_tf_models import VIAE
from custom_tf_models.adversarial import compute_gradient_penalty


class IAEGAN(VIAE):
    def __init__(self,
                 encoder: Model,
                 decoder: Model,
                 discriminator: Model,
                 step_size: int,
                 extra_steps: int,
                 use_stochastic_loss: bool = True,
                 reconstruction_lambda=1e0,
                 kl_divergence_lambda=1e-2,
                 adversarial_lambda: float = 1e-2,
                 gradient_penalty_lambda: float = 1e1,
                 ):
        super(IAEGAN, self).__init__(encoder=encoder,
                                     decoder=decoder,
                                     step_size=step_size,
                                     use_stochastic_loss=use_stochastic_loss,
                                     kl_divergence_lambda=kl_divergence_lambda)
        self.discriminator = discriminator
        self.extra_steps = extra_steps

        self.reconstruction_lambda = reconstruction_lambda
        self._reconstruction_lambda = tf.constant(value=reconstruction_lambda, dtype=tf.float32,
                                                  name="reconstruction_lambda")

        self.adversarial_lambda = adversarial_lambda
        self._adversarial_lambda = tf.constant(value=adversarial_lambda, dtype=tf.float32,
                                               name="adversarial_lambda")

        self.gradient_penalty_lambda = gradient_penalty_lambda
        self._gradient_penalty_lambda = tf.constant(value=gradient_penalty_lambda, dtype=tf.float32,
                                                    name="gradient_penalty_lambda")

    @tf.function
    def discriminate(self, inputs):
        return self.discriminator(inputs)

    # region Losses
    @tf.function
    def gradient_penalty(self, real, fake) -> tf.Tensor:
        return compute_gradient_penalty(real=real, fake=fake, discriminator=self.discriminator)

    @tf.function
    def compute_iae_metrics(self, inputs) -> Dict[str, tf.Tensor]:
        # region Forward
        inputs_shape = tf.shape(inputs)

        encoded = self.get_interpolated_latent_code(inputs, merge_batch_and_steps=True, extra_steps=self.extra_steps)
        latent_mean, latent_log_var = tf.split(encoded, num_or_size_splits=2, axis=-1)
        latent_code = self.sample_latent_distribution(latent_mean, latent_log_var)
        decoded = self.decoder(latent_code)

        batch_size, inputs_total_length, *dimensions = tf.unstack(inputs_shape)
        outputs_total_length = inputs_total_length + self.extra_length * 2
        outputs_shape = tf.stack([batch_size, outputs_total_length, *dimensions], axis=0)
        outputs = tf.reshape(decoded, outputs_shape)
        # endregion

        # region Reconstruction loss
        if self.extra_steps > 0:
            outputs_for_reconstruction = outputs[:, self.extra_length:-self.extra_length]
        else:
            outputs_for_reconstruction = outputs
        reconstruction_loss = self.compute_reconstruction_loss(inputs, outputs_for_reconstruction)
        # endregion

        kl_divergence = self.compute_kl_divergence(latent_mean, latent_log_var)

        # region Adversarial loss
        if self.extra_steps > 0:
            synth_before = outputs[:, :self.extra_length]
            synth_after = outputs[:, -self.extra_length:]
            synth_between = outputs_for_reconstruction[:, self.step_size:-self.step_size]
            inputs_start = inputs[:, :self.step_size]
            inputs_end = inputs[:, -self.step_size:]
            outputs_for_adversarial = [synth_before, inputs_start, synth_between, inputs_end, synth_after]
            outputs_for_adversarial = tf.concat(outputs_for_adversarial, axis=1)

            outputs_for_adversarial_left = outputs_for_adversarial[:, :-self.extra_length * 2]
            outputs_for_adversarial_right = outputs_for_adversarial[:, self.extra_length * 2:]
            outputs_for_adversarial_center = outputs_for_adversarial[:, self.extra_length: - self.extra_length]

            discriminated_left = self.discriminator(outputs_for_adversarial_left)
            discriminated_right = self.discriminator(outputs_for_adversarial_right)
            discriminated_center = self.discriminator(outputs_for_adversarial_center)
            adversarial_loss = tf.reduce_mean([discriminated_left, discriminated_right, discriminated_center])
        else:
            discriminated = self.discriminator(outputs)
            adversarial_loss = tf.reduce_mean(discriminated)
        # endregion

        loss = (reconstruction_loss * self._reconstruction_lambda +
                kl_divergence * self._kl_divergence_lambda +
                adversarial_loss * self._adversarial_lambda)

        return {
            "iae/loss": loss,
            "iae/reconstruction": reconstruction_loss,
            "iae/kl_divergence": kl_divergence,
            "iae/adversarial": adversarial_loss,
        }

    @tf.function
    def compute_discriminator_metrics(self, inputs) -> Dict[str, tf.Tensor]:
        # region Generate data
        inputs_shape = tf.shape(inputs)

        encoded = self.get_interpolated_latent_code(inputs, merge_batch_and_steps=True, extra_steps=self.extra_steps)
        latent_mean, latent_log_var = tf.split(encoded, num_or_size_splits=2, axis=-1)
        latent_code = self.sample_latent_distribution(latent_mean, latent_log_var)
        decoded = self.decoder(latent_code)

        batch_size, inputs_total_length, *dimensions = tf.unstack(inputs_shape)
        outputs_total_length = inputs_total_length + self.extra_length * 2
        outputs_shape = tf.stack([batch_size, outputs_total_length, *dimensions], axis=0)
        synth = tf.reshape(decoded, outputs_shape)
        # endregion

        # region Discriminate
        if self.extra_steps > 0:
            synth_before = synth[:, :self.extra_length]
            synth_after = synth[:, -self.extra_length:]
            synth_between = synth[:, self.extra_length + self.step_size:-self.extra_length - self.step_size]
            inputs_start = inputs[:, :self.step_size]
            inputs_end = inputs[:, -self.step_size:]

            full_synth = tf.concat([synth_before, inputs_start, synth_between, inputs_end, synth_after], axis=1)
            offset = tf.random.uniform(shape=[], maxval=self.extra_length * 2, dtype=tf.int32)
            synth = full_synth[:, offset:inputs_total_length + offset]

        synth_discriminated = self.discriminator(synth)
        real_discriminated = self.discriminator(inputs)

        synth_energy = tf.reduce_mean(synth_discriminated)
        real_energy = tf.reduce_mean(real_discriminated)
        # endregion

        gradient_penalty = compute_gradient_penalty(inputs, synth, self.discriminator)
        energy_delta = real_energy - synth_energy

        loss = (energy_delta * 0.1 +
                gradient_penalty * self._gradient_penalty_lambda)
        return {
            "discriminator/loss": loss,
            "discriminator/synth_energy": synth_energy,
            "discriminator/real_energy": real_energy,
            "discriminator/gradient_penalty": gradient_penalty,
            "discriminator/energy_delta": energy_delta,
        }

    # endregion

    # region Train / Test step
    @tf.function
    def train_step(self, data) -> Dict[str, tf.Tensor]:
        with tf.GradientTape(watch_accessed_variables=False) as iae_tape:
            iae_tape.watch(self.iae_trainable_variables)
            iae_metrics = self.compute_iae_metrics(data)
            iae_loss = iae_metrics["iae/loss"]

        iae_gradient = iae_tape.gradient(iae_loss, self.iae_trainable_variables)
        self.optimizer.apply_gradients(zip(iae_gradient, self.iae_trainable_variables))

        with tf.GradientTape(watch_accessed_variables=False) as discriminator_tape:
            discriminator_tape.watch(self.discriminator_trainable_variables)
            discriminator_metrics = self.compute_discriminator_metrics(data)
            discriminator_loss = discriminator_metrics["discriminator/loss"]

        discriminator_gradient = discriminator_tape.gradient(discriminator_loss, self.discriminator_trainable_variables)
        self.optimizer.apply_gradients(zip(discriminator_gradient, self.discriminator.trainable_variables))

        return {**iae_metrics, **discriminator_metrics}

    @tf.function
    def test_step(self, data) -> Dict[str, tf.Tensor]:
        iae_metrics = self.compute_iae_metrics(data)
        discriminator_metrics = self.compute_discriminator_metrics(data)
        return {**iae_metrics, **discriminator_metrics}

    # endregion

    # region Trainable variables
    @property
    def iae_trainable_variables(self):
        return self.encoder.trainable_variables + self.decoder.trainable_variables

    @property
    def discriminator_trainable_variables(self):
        return self.discriminator.trainable_variables

    # endregion

    @property
    def extra_length(self) -> int:
        return self.extra_steps * self.step_size

    @property
    def additional_test_metrics(self):
        return [
            *super(IAEGAN, self).additional_test_metrics,
            self.discriminate
        ]

    def get_config(self) -> Dict[str, Any]:
        base_config = super(IAEGAN, self).get_config()
        config = {
            **base_config,
            "discriminator": self.discriminator.get_config(),
            "extra_steps": self.extra_steps,
            "reconstruction_lambda": self.reconstruction_lambda,
            "kl_divergence_lambda": self.kl_divergence_lambda,
            "adversarial_lambda": self.adversarial_lambda,
            "gradient_penalty_lambda": self.gradient_penalty_lambda,
        }
        return config
