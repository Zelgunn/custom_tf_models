import tensorflow as tf
from tensorflow.python.keras.models import Model
from typing import Dict, Any

from custom_tf_models.description_length.LED import LED, LEDGoal
from custom_tf_models.basic.AEP import get_temporal_loss_weights
from misc_utils.math_utils import reduce_mean_from


# PredLED : Predicting with Low Energy Descriptors
class PreLED(LED):
    def __init__(self,
                 encoder: Model,
                 decoder: Model,
                 predictor: Model,
                 input_length: int,
                 use_temporal_reconstruction_loss=True,
                 features_per_block=1,
                 merge_dims_with_features=False,
                 description_energy_loss_lambda=1e-2,
                 use_noise=True,
                 noise_stddev=0.1,
                 reconstruct_noise=False,
                 goal_schedule: LEDGoal = None,
                 allow_negative_description_loss_weight=True,
                 goal_delta_factor=4.0,
                 unmasked_reconstruction_weight=1.0,
                 energy_margin=5.0,
                 **kwargs
                 ):
        super(PreLED, self).__init__(encoder=encoder,
                                     decoder=decoder,
                                     features_per_block=features_per_block,
                                     merge_dims_with_features=merge_dims_with_features,
                                     description_energy_loss_lambda=description_energy_loss_lambda,
                                     use_noise=use_noise,
                                     noise_stddev=noise_stddev,
                                     reconstruct_noise=reconstruct_noise,
                                     goal_schedule=goal_schedule,
                                     allow_negative_description_loss_weight=allow_negative_description_loss_weight,
                                     goal_delta_factor=goal_delta_factor,
                                     unmasked_reconstruction_weight=unmasked_reconstruction_weight,
                                     energy_margin=energy_margin,
                                     **kwargs)
        self.predictor = predictor
        self.input_length = input_length
        self.use_temporal_reconstruction_loss = use_temporal_reconstruction_loss

    # region Forward
    @tf.function
    def encode(self, inputs):
        inputs = inputs[:, :self.input_length]
        encoded = super(PreLED, self).encode(inputs)
        return encoded

    @tf.function
    def predict_next(self, inputs):
        encoded = self.encode(inputs)
        return self.predictor(encoded)

    @tf.function
    def decode_and_predict_next(self, encoded):
        decoded = self.decode(encoded)
        predicted = self.predictor(encoded)
        outputs = tf.concat([decoded, predicted], axis=1)
        return outputs

    @tf.function
    def autoencode_and_predict_next(self, inputs):
        encoded = self.encode(inputs)
        outputs = self.decode_and_predict_next(encoded)
        return outputs

    # endregion

    # region Loss
    @tf.function
    def compute_loss(self, inputs) -> Dict[str, tf.Tensor]:
        inputs, target, _ = self.add_training_noise(inputs)
        encoder_inputs = inputs[:, :self.input_length]

        # region Forward
        encoded = self.encoder(encoder_inputs)
        description_energy = self.description_energy_model(encoded)
        description_mask = self.get_description_mask(description_energy)
        masked_encoded = encoded * description_mask

        decoded = self.decode(masked_encoded)
        predicted = self.predictor(masked_encoded)

        # endregion

        # region Loss
        reconstruction_metrics = self.get_reconstruction_metrics(inputs, decoded, predicted)
        reconstruction_loss = reconstruction_metrics["rec/combined"]

        description_energy_loss = self.description_energy_loss(description_energy)
        description_energy_loss_weight = self.get_description_energy_loss_weight(reconstruction_loss)
        description_energy_loss *= description_energy_loss_weight

        loss = reconstruction_loss + description_energy_loss

        if self.perform_unmasked_reconstruction:
            unmasked_decoded = self.decoder(encoded)
            unmasked_predicted = self.predictor(encoded)
            unmasked_metrics = self.get_reconstruction_metrics(inputs, unmasked_decoded, unmasked_predicted)
            unmasked_reconstruction_error = unmasked_metrics["rec/combined"]
            loss += unmasked_reconstruction_error * self._unmasked_reconstruction_weight
        else:
            unmasked_reconstruction_error = None
        # endregion

        # region Metrics
        if self.goal_schedule is not None:
            reconstruction_goal = self.goal_schedule(self.train_step_index)
            reconstruction_goal_delta = reconstruction_loss - reconstruction_goal
            reconstruction_metrics["rec/goal"] = reconstruction_goal
            reconstruction_metrics["rec/goal_delta"] = reconstruction_goal_delta

        if self.perform_unmasked_reconstruction:
            reconstruction_metrics["rec/unmasked_combined"] = unmasked_reconstruction_error

        description_energy = tf.reduce_mean(description_energy)
        description_length = tf.reduce_mean(description_mask)
        description_metrics = {
            "description/energy": description_energy,
            "description/loss_weight": description_energy_loss_weight,
            "description/length": description_length,
        }

        metrics = {
            "loss": loss,
            **reconstruction_metrics,
            **description_metrics,
        }
        # endregion

        return metrics

    @tf.function
    def get_reconstruction_metrics(self, inputs: tf.Tensor, decoded: tf.Tensor, predicted: tf.Tensor):
        decoder_target = inputs[:, :self.input_length]
        predictor_target = inputs[:, self.input_length:]

        decoder_loss = self.get_reconstruction_loss(decoder_target, decoded)
        predictor_loss = self.get_reconstruction_loss(predictor_target, predicted)
        combined_loss = tf.concat([decoder_loss, predictor_loss], axis=1)

        if self.use_temporal_reconstruction_loss:
            output_length = tf.shape(predicted)[1]
            weights = get_temporal_loss_weights(self.input_length, output_length)
            combined_loss = tf.reduce_mean(combined_loss * weights)
        else:
            combined_loss = tf.reduce_mean(combined_loss)

        decoder_loss = tf.reduce_mean(decoder_loss)
        predictor_loss = tf.reduce_mean(predictor_loss)

        return {"rec/combined": combined_loss, "rec/decoder": decoder_loss, "rec/predictor": predictor_loss}

    @tf.function
    def get_reconstruction_loss(self, inputs: tf.Tensor, outputs: tf.Tensor):
        return reduce_mean_from(tf.square(inputs - outputs), start_axis=2)

    # endregion

    @tf.function
    def compute_description_energy(self, inputs: tf.Tensor) -> tf.Tensor:
        inputs = inputs[:, :self.input_length]
        return super(PreLED, self).compute_description_energy(inputs)

    def get_config(self) -> Dict[str, Any]:
        base_config = super(PreLED, self).get_config()
        config = {
            **base_config,
            "predictor": self.predictor.get_config(),
            "input_length": self.input_length,
            "use_temporal_reconstruction_loss": self.use_temporal_reconstruction_loss,
        }
        return config
