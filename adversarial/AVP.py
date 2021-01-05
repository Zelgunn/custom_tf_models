import tensorflow as tf
from tensorflow.python.keras import Model
from typing import Dict, Any, Tuple

from custom_tf_models.adversarial import compute_gradient_penalty
from custom_tf_models.utils import get_gradient_difference_loss


# AVP : Adversarial Variational Predictor
class AVP(Model):
    """
    A model that aims to predict future frames based on present frames.

    This model uses a variational setup - latent codes produced are used as the mean and variance of a normal random
    variable. This probabilistic setup is used to model uncertainty when trying to predict the future.

    This model also uses an adversarial setup - this allows this model to use high-level features taken from the
    discriminator to train the prediction part and improve details in the output. The Wasserstein loss is used
    with gradient penalty to train the adversarial parts.

        :param encoder: A model which converts inputs into latent codes. Latent codes are expected to be twice as wide
            as those the decoder (predictor) excepts, so that half of them can be used for computing the mean and the
            other half for computing the variance.
        :param decoder: A model which converts latent codes back to the input space. Currently, outputs are excepted
            to have the same shape as inputs.
        :param discriminator: A model with two outputs : [`final_outputs`, `intermediate_outputs`], where
            `intermediate_outputs` will be used for training the encode-predict part of the model.

    """

    def __init__(self,
                 encoder: Model,
                 decoder: Model,
                 discriminator: Model,
                 prediction_lambda: float = 1.0,
                 gradient_difference_lambda: float = 1.0,
                 high_level_prediction_lambda: float = 1e1,
                 kl_divergence_lambda: float = 1e-2,
                 adversarial_lambda: float = 1e-2,
                 gradient_penalty_lambda: float = 1e1,
                 input_length: int = None,
                 **kwargs):
        super(AVP, self).__init__(**kwargs)

        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.prediction_lambda = prediction_lambda
        self.gradient_difference_lambda = gradient_difference_lambda
        self.high_level_prediction_lambda = high_level_prediction_lambda
        self.kl_divergence_lambda = kl_divergence_lambda
        self.adversarial_lambda = adversarial_lambda
        self.gradient_penalty_lambda = gradient_penalty_lambda

        if input_length is None:
            input_length = encoder.input_shape[1]
        self.input_length = input_length

        self._prediction_lambda = tf.constant(prediction_lambda, dtype=tf.float32, name="prediction_lambda")
        self._gradient_difference_lambda = tf.constant(gradient_difference_lambda, dtype=tf.float32,
                                                       name="gradient_difference_lambda")
        self._high_level_prediction_lambda = tf.constant(high_level_prediction_lambda, dtype=tf.float32,
                                                         name="high_level_prediction_lambda")
        self._kl_divergence_lambda = tf.constant(kl_divergence_lambda, dtype=tf.float32, name="kl_divergence_lambda")
        self._adversarial_lambda = tf.constant(adversarial_lambda, dtype=tf.float32, name="adversarial_lambda")
        self._gradient_penalty_lambda = tf.constant(gradient_penalty_lambda, dtype=tf.float32,
                                                    name="gradient_penalty_lambda")

    # region Inference
    @tf.function
    def call(self, inputs, training=None, mask=None):
        deterministic = True if (training is None) else not training
        return self.predict_next_and_concat(inputs, deterministic=deterministic)

    @tf.function
    def get_present(self, inputs: tf.Tensor) -> tf.Tensor:
        return inputs[:, :self.input_length]

    @tf.function
    def get_future(self, inputs: tf.Tensor) -> tf.Tensor:
        return inputs[:, self.input_length:]

    @tf.function
    def encode(self, inputs: tf.Tensor) -> tf.Tensor:
        inputs = self.get_present(inputs)
        return self.encoder(inputs)

    def get_latent_distribution(self, inputs) -> Tuple[tf.Tensor, tf.Tensor]:
        encoder_outputs = self.encode(inputs)
        latent_mean, latent_log_var = tf.split(encoder_outputs, num_or_size_splits=2, axis=-1)
        return latent_mean, latent_log_var

    @tf.function
    def sample_latent_distribution(self, latent_mean: tf.Tensor, latent_log_var: tf.Tensor) -> tf.Tensor:
        epsilon = tf.random.normal(shape=tf.shape(latent_mean))
        # noinspection PyTypeChecker
        latent_code = latent_mean + tf.exp(0.5 * latent_log_var) * epsilon
        return latent_code

    @tf.function
    def decode(self, latent_code: tf.Tensor) -> tf.Tensor:
        return self.decoder(latent_code)

    @tf.function
    def discriminate(self, inputs: tf.Tensor) -> tf.Tensor:
        final_outputs, _ = self.discriminator(inputs)
        return final_outputs

    @tf.function
    def predict_next(self, inputs: tf.Tensor, deterministic: bool = True) -> tf.Tensor:
        latent_mean, latent_log_var = self.get_latent_distribution(inputs)
        if deterministic:
            latent_code = latent_mean
        else:
            latent_code = self.sample_latent_distribution(latent_mean, latent_log_var)
        outputs = self.decode(latent_code)
        return outputs

    @tf.function
    def concat_present_with_prediction(self, inputs: tf.Tensor, prediction: tf.Tensor) -> tf.Tensor:
        present = self.get_present(inputs)
        outputs = tf.concat([present, prediction], axis=1)
        return outputs

    @tf.function
    def predict_next_and_concat(self, inputs: tf.Tensor, deterministic: bool = True) -> tf.Tensor:
        prediction = self.predict_next(inputs, deterministic)
        return self.concat_present_with_prediction(inputs, prediction)

    # endregion

    # region Losses
    @tf.function
    def compute_predictor_metrics(self, inputs: tf.Tensor) -> Dict[str, tf.Tensor]:
        # region Forward
        # latent_distribution = self.get_latent_distribution(inputs)
        latent_mean, latent_log_var = self.get_latent_distribution(inputs)
        latent_code = self.sample_latent_distribution(latent_mean, latent_log_var)

        # latent_code = latent_distribution.sample()
        prediction = self.decode(latent_code)
        outputs = self.concat_present_with_prediction(inputs, prediction)
        discriminated, outputs_features = self.discriminator(outputs)
        # endregion

        future = self.get_future(inputs)
        pixel_wise_l1 = tf.reduce_mean(tf.abs(future - prediction))
        gradient_difference = tf.reduce_mean(get_gradient_difference_loss(future, prediction, alpha=1))

        _, inputs_features = self.discriminator(inputs)
        high_level_l1 = tf.reduce_mean(tf.abs(inputs_features - outputs_features))

        kl_divergence = tf.square(latent_mean) + tf.exp(latent_log_var) - tf.constant(1.0) - latent_log_var
        kl_divergence = 0.5 * tf.reduce_mean(kl_divergence)

        adversarial_loss = tf.reduce_mean(discriminated)

        # tf.print(tf.reduce_mean(latent_distribution.mean()), tf.reduce_mean(latent_distribution.variance()))

        # region Total loss
        loss = (pixel_wise_l1 * self._prediction_lambda +
                gradient_difference * self._gradient_difference_lambda +
                high_level_l1 * self._high_level_prediction_lambda +
                kl_divergence * self._kl_divergence_lambda +
                adversarial_loss * self._adversarial_lambda)
        # endregion

        return {
            "predictor/loss": loss,
            "predictor/pixel_wise_l1": pixel_wise_l1,
            "predictor/gradient_difference": gradient_difference,
            "predictor/high_level_l1": high_level_l1,
            "predictor/kl_divergence": kl_divergence,
            "predictor/adversarial": adversarial_loss,
        }

    @tf.function
    def compute_discriminator_metrics(self, inputs: tf.Tensor) -> Dict[str, tf.Tensor]:
        synth_prediction = tf.stop_gradient(self.predict_next_and_concat(inputs, deterministic=False))

        synth_discriminated = self.discriminate(synth_prediction)
        real_discriminated = self.discriminate(inputs)

        synth_energy = tf.reduce_mean(synth_discriminated)
        real_energy = tf.reduce_mean(real_discriminated)
        gradient_penalty = compute_gradient_penalty(inputs, synth_prediction, self.discriminator)

        loss = (real_energy - synth_energy) + gradient_penalty * self._gradient_penalty_lambda
        return {
            "discriminator/loss": loss,
            "discriminator/synth_energy": synth_energy,
            "discriminator/real_energy": real_energy,
            "discriminator/gradient_penalty": gradient_penalty,
        }

    # endregion

    @tf.function
    def train_step(self, data) -> Dict[str, tf.Tensor]:
        with tf.GradientTape(watch_accessed_variables=False) as predictor_tape:
            predictor_tape.watch(self.predictor_trainable_variables)
            predictor_metrics = self.compute_predictor_metrics(data)
            predictor_loss = predictor_metrics["predictor/loss"]

        predictor_gradients = predictor_tape.gradient(predictor_loss, self.predictor_trainable_variables)
        self.optimizer.apply_gradients(zip(predictor_gradients, self.predictor_trainable_variables))

        with tf.GradientTape(watch_accessed_variables=False) as discriminator_tape:
            discriminator_tape.watch(self.discriminator.trainable_variables)
            discriminator_metrics = self.compute_discriminator_metrics(data)
            discriminator_loss = discriminator_metrics["discriminator/loss"]

        discriminator_gradients = discriminator_tape.gradient(discriminator_loss,
                                                              self.discriminator.trainable_variables)
        self.optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.trainable_variables))

        return {**predictor_metrics, **discriminator_metrics}

    @tf.function
    def test_step(self, data):
        predictor_metrics = self.compute_predictor_metrics(data)
        discriminator_metrics = self.compute_discriminator_metrics(data)
        return {**predictor_metrics, **discriminator_metrics}

    @property
    def predictor_variables(self):
        return self.encoder.variables + self.decoder.variables

    @property
    def predictor_trainable_variables(self):
        return self.encoder.trainable_variables + self.decoder.trainable_variables

    def get_config(self) -> Dict[str, Any]:
        config = {
            "encoder": self.encoder.get_config(),
            "decoder": self.decoder.get_config(),
            "discriminator": self.discriminator.get_config(),
            "input_length": self.input_length,
            "prediction_lambda": self.prediction_lambda,
            "kl_divergence_lambda": self.kl_divergence_lambda,
            "adversarial_lambda": self.adversarial_lambda,
            "gradient_penalty_lambda": self.gradient_penalty_lambda,
        }
        return config
