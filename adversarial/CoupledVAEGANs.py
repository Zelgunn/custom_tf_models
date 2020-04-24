import tensorflow as tf
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2
from tensorflow.python.keras import Model
from typing import Dict, Tuple

from custom_tf_models import CustomModel, AE
from adversarial import gradient_penalty


class CoupledVAEGANs(CustomModel):
    def __init__(self,
                 generator_1: AE,
                 generator_2: AE,
                 discriminator_1: Model,
                 discriminator_2: Model,
                 generators_optimizer: OptimizerV2,
                 discriminators_optimizer: OptimizerV2,
                 base_reconstruction_loss_weight=1e+1,
                 base_divergence_loss_weight=1e-2,
                 cycle_reconstruction_loss_weight=1e+1,
                 cycle_divergence_loss_weight=1e-2,
                 adversarial_loss_weight=1e+0,
                 gradient_penalty_loss_weight=1e+1,
                 domain_1_name="1",
                 domain_2_name="2",
                 seed=None,
                 **kwargs
                 ):
        super(CoupledVAEGANs, self).__init__(**kwargs)

        self.generator_1 = generator_1
        self.generator_2 = generator_2
        self.discriminator_1 = discriminator_1
        self.discriminator_2 = discriminator_2

        self.generators_optimizer = generators_optimizer
        self.discriminators_optimizer = discriminators_optimizer

        self.base_reconstruction_loss_weight = base_reconstruction_loss_weight
        self.base_divergence_loss_weight = base_divergence_loss_weight
        self.cycle_reconstruction_loss_weight = cycle_reconstruction_loss_weight
        self.cycle_divergence_loss_weight = cycle_divergence_loss_weight
        self.adversarial_loss_weight = adversarial_loss_weight
        self.gradient_penalty_loss_weight = gradient_penalty_loss_weight

        self.domain_1_name = domain_1_name
        self.domain_2_name = domain_2_name

        self.seed = seed

    @tf.function
    def train_step(self, inputs, *args, **kwargs):
        x_1, x_2 = inputs

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.generators_trainable_variables)
            generators_losses = self.compute_generators_losses(x_1, x_2)
            generators_loss = tf.reduce_sum(generators_losses)

        gradients = tape.gradient(generators_loss, self.generators_trainable_variables)
        self.generators_optimizer.apply_gradients(zip(gradients, self.generators_trainable_variables))

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.discriminators_trainable_variables)
            discriminators_losses = self.compute_discriminators_losses(x_1, x_2)
            discriminators_loss = tf.reduce_sum(discriminators_losses)

        gradients = tape.gradient(discriminators_loss, self.discriminators_trainable_variables)
        self.discriminators_optimizer.apply_gradients(zip(gradients, self.discriminators_trainable_variables))

        return (*generators_losses, *discriminators_losses)

    @tf.function
    def compute_loss(self, inputs, *args, **kwargs):
        x_1, x_2 = inputs

        generators_losses = self.compute_generators_losses(x_1, x_2)
        discriminators_losses = self.compute_discriminators_losses(x_1, x_2)
        return (*generators_losses, *discriminators_losses)

    def compute_generators_losses(self, x_1, x_2):
        # region Forward
        z_1 = self.encode_1(x_1)
        z_2 = self.encode_2(x_2)

        x_1_1 = self.decode_1(z_1)
        x_1_2 = self.decode_2(z_1)
        x_2_1 = self.decode_1(z_2)
        x_2_2 = self.decode_2(z_2)

        z_1_2 = self.encode_2(x_1_2)
        z_2_1 = self.encode_1(x_2_1)

        x_1_2_1 = self.decode_1(z_1_2)
        x_2_1_2 = self.decode_2(z_2_1)
        # endregion

        # region Reconstruction Loss
        x_1_reconstruction_loss = self.reconstruction_loss(x_1, x_1_1)
        x_2_reconstruction_loss = self.reconstruction_loss(x_2, x_2_2)
        x_1_2_1_reconstruction_loss = self.reconstruction_loss(x_1, x_1_2_1)
        x_2_1_2_reconstruction_loss = self.reconstruction_loss(x_2, x_2_1_2)

        base_reconstruction_loss = x_1_reconstruction_loss + x_2_reconstruction_loss
        cycle_reconstruction_loss = x_1_2_1_reconstruction_loss + x_2_1_2_reconstruction_loss
        reconstruction_loss = (
                base_reconstruction_loss * self.base_reconstruction_loss_weight +
                cycle_reconstruction_loss * self.cycle_reconstruction_loss_weight
        )

        # endregion

        # region Divergence loss
        z_1_divergence = self.divergence_loss(z_1)
        z_2_divergence = self.divergence_loss(z_2)
        z_1_2_divergence = self.divergence_loss(z_1_2)
        z_2_1_divergence = self.divergence_loss(z_2_1)
        base_divergence_loss = z_1_divergence + z_2_divergence
        cycle_divergence_loss = z_1_2_divergence + z_2_1_divergence
        divergence_loss = (
                base_divergence_loss * self.base_divergence_loss_weight +
                cycle_divergence_loss * self.cycle_divergence_loss_weight
        )
        # endregion

        # region Adversarial loss
        x_1_2_discriminated = tf.reduce_mean(self.discriminator_2(x_1_2))
        x_2_1_discriminated = tf.reduce_mean(self.discriminator_1(x_2_1))
        adversarial_loss = (x_1_2_discriminated + x_2_1_discriminated) * self.adversarial_loss_weight
        # endregion

        return reconstruction_loss, divergence_loss, adversarial_loss

    def compute_discriminators_losses(self, x_1, x_2):
        # region Forward
        z_1 = self.encode_1(x_1)
        z_2 = self.encode_2(x_2)

        x_2_1 = self.decode_1(z_2)
        x_1_2 = self.decode_2(z_1)

        x_2_1 = tf.stop_gradient(x_2_1)
        x_1_2 = tf.stop_gradient(x_1_2)
        # endregion

        # region Discrimination loss
        x_1_discriminated = self.discriminator_1(x_1)
        x_2_discriminated = self.discriminator_2(x_2)
        x_2_1_discriminated = self.discriminator_1(x_2_1)
        x_1_2_discriminated = self.discriminator_2(x_1_2)

        discriminator_1_loss = self.discriminator_loss(x_1_discriminated, x_2_1_discriminated)
        discriminator_2_loss = self.discriminator_loss(x_2_discriminated, x_1_2_discriminated)
        discriminators_loss = (discriminator_1_loss + discriminator_2_loss) * self.adversarial_loss_weight
        # endregion

        # region Gradient penalty
        gradient_penalty_loss_1 = gradient_penalty(real=x_1, fake=x_2_1, discriminator=self.discriminator_1,
                                                   seed=self.seed)
        gradient_penalty_loss_2 = gradient_penalty(real=x_2, fake=x_1_2, discriminator=self.discriminator_2,
                                                   seed=self.seed)
        gradient_penalty_loss = (gradient_penalty_loss_1 + gradient_penalty_loss_2) * self.gradient_penalty_loss_weight
        # endregion

        return discriminators_loss, gradient_penalty_loss

    # region Loss helpers
    @staticmethod
    def reconstruction_loss(x_true, x_pred):
        return tf.reduce_mean(tf.abs(x_true - x_pred))

    @staticmethod
    def divergence_loss(z):
        return tf.reduce_mean(tf.square(z))

    @staticmethod
    def discriminator_loss(real_discriminated, fake_discriminated):
        return tf.reduce_mean(real_discriminated) - tf.reduce_mean(fake_discriminated)

    # endregion

    # region Encode / Decode
    def autoencode(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
        x_1, x_2 = inputs

        x_1_1 = self.autoencode_1_1(x_1)
        x_2_2 = self.autoencode_2_2(x_2)

        return x_1_1, x_2_2

    def autoencode_1_1(self, inputs: tf.Tensor) -> tf.Tensor:
        return self._autoencode(inputs=inputs, from_domain_1=True, to_domain_1=True)

    def autoencode_1_2(self, inputs: tf.Tensor) -> tf.Tensor:
        return self._autoencode(inputs=inputs, from_domain_1=True, to_domain_1=False)

    def autoencode_2_1(self, inputs: tf.Tensor) -> tf.Tensor:
        return self._autoencode(inputs=inputs, from_domain_1=False, to_domain_1=True)

    def autoencode_2_2(self, inputs: tf.Tensor) -> tf.Tensor:
        return self._autoencode(inputs=inputs, from_domain_1=False, to_domain_1=False)

    def cross_autoencode(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
        x_1, x_2 = inputs
        x_1_2 = self.autoencode_1_2(x_1)
        x_2_1 = self.autoencode_2_1(x_2)
        return x_2_1, x_1_2

    def cycle_autoencode(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
        x_1, x_2 = inputs
        x_2_1, x_1_2 = self.cross_autoencode((x_1, x_2))
        x_1_2_1, x_2_1_2 = self.cross_autoencode((x_2_1, x_1_2))
        return x_1_2_1, x_2_1_2

    def encode_1(self, inputs: tf.Tensor):
        return self._encode(inputs=inputs, from_domain_1=True)

    def encode_2(self, inputs: tf.Tensor):
        return self._encode(inputs=inputs, from_domain_1=False)

    def decode_1(self, latent_code: tf.Tensor):
        return self._decode(latent_code=latent_code, to_domain_1=True)

    def decode_2(self, latent_code: tf.Tensor):
        return self._decode(latent_code=latent_code, to_domain_1=False)

    def _autoencode(self, inputs: tf.Tensor, from_domain_1: bool, to_domain_1: bool) -> tf.Tensor:
        return self._decode(self._encode(inputs=inputs, from_domain_1=from_domain_1), to_domain_1=to_domain_1)

    def _encode(self, inputs: tf.Tensor, from_domain_1: bool):
        generator = self.generator_1 if from_domain_1 else self.generator_2
        return generator.encode(inputs)

    def _decode(self, latent_code: tf.Tensor, to_domain_1: bool):
        code_shape = tf.shape(latent_code)
        noise = tf.random.normal(shape=code_shape, mean=0.0, stddev=1.0, seed=self.seed)
        generator = self.generator_1 if to_domain_1 else self.generator_2
        return generator.decode(latent_code + noise)

    # endregion

    # region Trainable variables
    @property
    def generators_trainable_variables(self):
        return self.generator_1.trainable_variables + self.generator_2.trainable_variables

    @property
    def discriminators_trainable_variables(self):
        return self.discriminator_1.trainable_variables + self.discriminator_2.trainable_variables

    # endregion

    # region IDs / Config
    @property
    def models_ids(self) -> Dict[Model, str]:
        return {
            self.generator_1: "generator_{}".format(self.domain_1_name),
            self.generator_2: "generator_{}".format(self.domain_2_name),

            self.discriminator_1: "discriminator_{}".format(self.domain_1_name),
            self.discriminator_2: "discriminator_{}".format(self.domain_2_name),
        }

    @property
    def optimizers_ids(self) -> Dict[OptimizerV2, str]:
        return {
            self.generators_optimizer: "generators_optimizer",
            self.discriminators_optimizer: "discriminators_optimizer",
        }

    @property
    def metrics_names(self):
        return [
            "reconstruction",
            "encoder_divergence",
            "generator_adversarial",
            "discriminator_adversarial",
            "gradient_penalty"
        ]

    def get_config(self):
        return {
            "generator_1": self.generator_1.get_config(),
            "generator_2": self.generator_2.get_config(),
            "discriminator_1": self.discriminator_1.get_config(),
            "discriminator_2": self.discriminator_2.get_config(),

            "generators_optimizer": self.generators_optimizer.get_config(),
            "discriminators_optimizer": self.discriminators_optimizer.get_config(),

            "base_reconstruction_loss_weight": self.base_reconstruction_loss_weight,
            "base_divergence_loss_weight": self.base_divergence_loss_weight,
            "cycle_reconstruction_loss_weight": self.cycle_reconstruction_loss_weight,
            "cycle_divergence_loss_weight": self.cycle_divergence_loss_weight,
            "adversarial_loss_weight": self.adversarial_loss_weight,
            "gradient_penalty_loss_weight": self.gradient_penalty_loss_weight,

            "domain_1_name": self.domain_1_name,
            "domain_2_name": self.domain_2_name,
        }
    # endregion
