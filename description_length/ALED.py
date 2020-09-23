import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.keras.models import Model
from typing import Dict, Any, Union, List

from custom_tf_models.description_length.LED import LED
from custom_tf_models.adversarial.GANLoss import gradient_penalty


# LED : Adversarial Low Energy Descriptors
class ALED(LED):
    def __init__(self,
                 encoder: Model,
                 decoder: Model,
                 generator: Model,
                 # generator_learning_rate: LearningRateType,
                 features_per_block: int,
                 merge_dims_with_features=False,
                 descriptors_activation="tanh",
                 binarization_temperature=50.0,
                 add_binarization_noise_to_mask=True,
                 description_energy_loss_lambda=1e-2,
                 gradient_penalty_weight=1e+0,
                 seed=None,
                 **kwargs
                 ):
        super(ALED, self).__init__(encoder=encoder,
                                   decoder=decoder,
                                   features_per_block=features_per_block,
                                   merge_dims_with_features=merge_dims_with_features,
                                   descriptors_activation=descriptors_activation,
                                   binarization_temperature=binarization_temperature,
                                   add_binarization_noise_to_mask=add_binarization_noise_to_mask,
                                   description_energy_loss_lambda=description_energy_loss_lambda,
                                   use_noise=False,
                                   seed=seed,
                                   **kwargs)

        self.generator = generator
        # self.generator_learning_rate = generator_learning_rate
        self.gradient_penalty_weight = gradient_penalty_weight

        self.generator_noise_distribution = self._make_generator_input_noise_distribution()
        self._gradient_penalty_weight = tf.constant(gradient_penalty_weight, dtype=tf.float32,
                                                    name="gradient_penalty_weight")

    @tf.function
    def sample_generator(self, batch_size: Union[int, tf.Tensor]) -> tf.Tensor:
        input_noise = self.generator_noise_distribution.sample(sample_shape=batch_size)
        return self.generator(input_noise)

    @tf.function
    def gradient_penalty(self, real_inputs, fake_inputs):
        led_gradient_penalty = gradient_penalty(real=real_inputs, fake=fake_inputs, seed=self.seed,
                                                discriminator=self.compute_description_energy)
        return led_gradient_penalty

    @tf.function
    def compute_led_loss(self, inputs: tf.Tensor) -> Dict[str, tf.Tensor]:
        real_inputs = inputs
        fake_inputs = self.sample_generator(tf.shape(real_inputs)[0])
        fake_inputs = tf.stop_gradient(fake_inputs)

        real_encoded = self.encoder(real_inputs)
        real_description_energy = self.description_energy_model(real_encoded)

        fake_encoded = self.encoder(fake_inputs)
        fake_description_energy = self.description_energy_model(fake_encoded)

        description_mask = self.get_description_mask(real_description_energy)
        outputs = self.decode(real_encoded * description_mask)

        real_loss = self.description_energy_loss(real_description_energy)
        fake_loss = tf.nn.relu(- self.description_energy_loss(fake_description_energy))
        description_energy_loss = (real_loss + fake_loss) * self.description_energy_loss_lambda
        # led_gradient_penalty = self.gradient_penalty(real_inputs, fake_inputs) * self._gradient_penalty_weight
        reconstruction_loss = self.reconstruction_loss(inputs, outputs)

        loss = reconstruction_loss + description_energy_loss  # + led_gradient_penalty

        description_length = tf.reduce_mean(tf.stop_gradient(description_mask))

        return {
            "led/loss": loss,
            "led/reconstruction_loss": reconstruction_loss,
            "led/real_loss": real_loss,
            "led/fake_loss": fake_loss,
            # "gradient_penalty": led_gradient_penalty,
            "led/description_length": description_length,
        }

    @tf.function
    def compute_generator_loss(self, inputs: tf.Tensor) -> Dict[str, tf.Tensor]:
        fake_inputs = self.sample_generator(tf.shape(inputs)[0])

        encoded = self.encoder(fake_inputs)
        description_energy = self.description_energy_model(encoded)
        description_mask = self.get_description_mask(description_energy)
        # outputs = self.decode(encoded * description_mask)

        # reconstruction_loss = self.reconstruction_loss(inputs, outputs)
        description_energy_loss = self.description_energy_loss(description_energy)
        loss = self._description_energy_loss_lambda * description_energy_loss
        # loss = reconstruction_loss + self._description_energy_loss_lambda * description_energy_loss

        description_length = tf.reduce_mean(tf.stop_gradient(description_mask))

        generated_mean = tf.reduce_mean(fake_inputs)
        generated_std = tf.math.reduce_std(fake_inputs)

        mean_regularization = tf.abs(generated_mean)
        std_regularization = tf.abs(1.0 - generated_std)
        regularization = (mean_regularization + std_regularization)

        loss += regularization * 1e-2

        metrics = {
            "generator/loss": loss,
            # "generator/reconstruction_loss": reconstruction_loss,
            "generator/description_energy": description_energy_loss,
            "generator/mean": generated_mean,
            "generator/std": generated_std,
            "generator/description_length": description_length,
        }

        return metrics

    @tf.function
    def train_step(self, inputs) -> Dict[str, tf.Tensor]:
        with tf.GradientTape(watch_accessed_variables=False) as led_tape:
            led_tape.watch(self.led_trainable_variables)
            led_metrics = self.compute_led_loss(inputs)
            led_loss = led_metrics["led/loss"]

        led_gradients = led_tape.gradient(led_loss, self.led_trainable_variables)
        self.optimizer.apply_gradients(zip(led_gradients, self.led_trainable_variables))

        with tf.GradientTape(watch_accessed_variables=False) as generator_tape:
            generator_tape.watch(self.generator.trainable_variables)
            generator_metrics = self.compute_generator_loss(inputs)
            generator_loss = generator_metrics["generator/loss"]

        generator_gradients = generator_tape.gradient(generator_loss, self.generator.trainable_variables)
        self.optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))

        metrics = {**led_metrics, **generator_metrics}
        return metrics

    @property
    def led_trainable_variables(self) -> List:
        return self.encoder.trainable_variables + \
               self.decoder.trainable_variables + \
               self.description_energy_model.trainable_variables

    def get_config(self) -> Dict[str, Any]:
        led_config = super(ALED, self).get_config()
        return {
            **led_config,
            "generator": self.generator.get_config(),
            "gradient_penalty_weight": self.gradient_penalty_weight,
        }

    # region Make generator input noise distribution
    def _make_generator_input_noise_distribution(self) -> tfp.distributions.Normal:
        return self.make_generator_input_noise_distribution(self.generator, loc=0.0, scale=1.0)

    @staticmethod
    def make_generator_input_noise_distribution(generator: Model, loc=0.0, scale=1.0) -> tfp.distributions.Normal:
        generator_input_shape = generator.input_shape[1:]
        loc = tf.constant(value=loc, dtype=tf.float32, shape=generator_input_shape)
        scale = tf.constant(value=scale, dtype=tf.float32, shape=generator_input_shape)
        return tfp.distributions.Normal(loc=loc, scale=scale)

    # endregion
