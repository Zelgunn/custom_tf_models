import tensorflow as tf
from tensorflow.python.keras import Model
from typing import Dict, List, Callable

from custom_tf_models.basic.MinimalistDescriptor import MinimalistDescriptor


class MinimalistDescriptorV3(MinimalistDescriptor):
    def __init__(self,
                 encoders: List[Model],
                 decoders: List[Model],
                 stop_encoder: Model,
                 max_steps: int,
                 learning_rate,
                 stop_lambda=1e-3,
                 stop_residual_gradients=True,
                 train_stride=1000,
                 noise_type="dense",
                 noise_factor_distribution="normal",
                 binarization_temperature=50.0,
                 seed=None,
                 **kwargs):
        if isinstance(encoders, list) and isinstance(decoders, list):
            if len(encoders) != len(decoders):
                raise ValueError("Lists of encoders and decoders must have the same length. Found {} and {}"
                                 .format(len(encoders), len(decoders)))

        encoder = encoders[0] if isinstance(encoders, list) else encoders
        decoder = decoders[0] if isinstance(decoders, list) else encoders
        super(MinimalistDescriptorV3, self).__init__(encoder=encoder,
                                                     decoder=decoder,
                                                     stop_encoder=stop_encoder,
                                                     max_steps=max_steps,
                                                     learning_rate=learning_rate,
                                                     stop_lambda=stop_lambda,
                                                     stop_residual_gradients=stop_residual_gradients,
                                                     train_stride=train_stride,
                                                     noise_type=noise_type,
                                                     noise_factor_distribution=noise_factor_distribution,
                                                     binarization_temperature=binarization_temperature,
                                                     seed=seed,
                                                     **kwargs)

        self.encoders = encoders
        self.decoders = decoders

    @tf.function
    def encode_at_step(self, step, inputs):
        if isinstance(self.encoders, list):
            branch_encoders = self.make_branch_models(self.encoders, inputs)
            encoded = tf.switch_case(step, branch_encoders)
        else:
            encoded = self.encoder(inputs)

        return encoded

    @tf.function
    def decode_at_step(self, step, inputs):
        if isinstance(self.decoders, list):
            branch_decoders = self.make_branch_models(self.decoders, inputs)
            decoded = tf.switch_case(step, branch_decoders)
        else:
            decoded = self.decoder(inputs)

        return decoded

    @tf.function
    def main_loop_step(self, step, residual):
        encoded = self.encode_at_step(step, residual)

        keep_going = self.stop_encoder(encoded)
        keep_going = tf.squeeze(keep_going, axis=-1)
        keep_going = self.binarize(keep_going)

        latent_code_rank = encoded.shape.rank
        keep_going_multiplier = tf.reshape(keep_going, [-1] + [1] * (latent_code_rank - 1))

        encoded *= keep_going_multiplier
        decoded = self.decode_at_step(step, encoded)
        residual -= decoded

        return residual, keep_going

    @staticmethod
    def make_branch_models(models: List[Model], inputs: tf.Tensor) -> Dict[int, Callable]:
        return {i: lambda: models[i](inputs) for i in range(len(models))}
