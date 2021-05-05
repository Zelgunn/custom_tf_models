import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Activation, Input, Concatenate, Add, Layer
from typing import List, Dict, Union

from custom_tf_models.basic.AE import AE
from CustomKerasLayers import SplitLayer


class IterativeAE(AE):
    def __init__(self,
                 encoders: List[Model],
                 decoders: List[Model],
                 output_activation: Union[Activation, str, Layer],
                 stop_accumulator_gradients=True,
                 **kwargs):
        if not isinstance(output_activation, Layer):
            output_activation = Activation(output_activation)

        encoder = self.zip_encoders(encoders)
        decoder = self.zip_decoders(decoders, output_activation)
        super(IterativeAE, self).__init__(encoder=encoder,
                                          decoder=decoder,
                                          **kwargs)
        self.encoders = encoders
        self.decoders = decoders
        self.output_activation = output_activation
        self.stop_accumulator_gradients = stop_accumulator_gradients

    @staticmethod
    def zip_encoders(encoders: List[Model], model_name="ZippedEncoder") -> Model:
        input_shape = encoders[0].input_shape
        inputs = Input(batch_input_shape=input_shape, name="{}_InputLayer".format(model_name))
        split_encoded = [encoder(inputs) for encoder in encoders]
        encoded = Concatenate()(split_encoded)
        model = Model(inputs=inputs, outputs=encoded, name=model_name)
        return model

    @staticmethod
    def zip_decoders(decoders: List[Model], output_activation: Layer, model_name="ZippedDecoder") -> Model:
        base_input_shape = decoders[0].input_shape[:-1]

        code_sizes = [decoder.input_shape[-1] for decoder in decoders]
        total_code_size = sum(code_sizes)
        input_shape = (*base_input_shape, total_code_size)

        encoded = Input(batch_input_shape=input_shape, name="{}_InputLayer".format(model_name))
        split_encoded = SplitLayer(num_or_size_splits=code_sizes, axis=-1, num=len(decoders))(encoded)
        split_decoded = [decoders[i](split_encoded[i]) for i in range(len(decoders))]
        decoded = Add()(split_decoded)
        outputs = output_activation(decoded)

        model = Model(inputs=encoded, outputs=outputs, name=model_name)
        return model

    @tf.function
    def compute_loss(self,
                     inputs: tf.Tensor
                     ) -> Dict[str, tf.Tensor]:
        accumulator = tf.zeros_like(inputs)
        losses = []

        for encoder, decoder in zip(self.encoders, self.decoders):
            accumulator += decoder(encoder(inputs))
            step_outputs = self.output_activation(accumulator)
            step_loss = self.compute_reconstruction_loss(inputs, step_outputs)
            losses.append(step_loss)

            if self.stop_accumulator_gradients:
                accumulator = tf.stop_gradient(accumulator)

        losses = tf.stack(losses)

        # test_w = tf.range(losses.shape[0], dtype=tf.float32)
        # test_w /= tf.reduce_sum(test_w)
        #
        # loss = tf.reduce_sum(losses * test_w)
        # loss = tf.reduce_mean(losses)

        return {
            "loss": losses[-1],
            "reconstruction_loss": losses[-1]
        }

    def get_config(self):
        base_config = super(IterativeAE, self).get_config()
        config = {
            **base_config,
            "output_activation": self.output_activation.get_config(),
            "stop_accumulator_gradients": self.stop_accumulator_gradients,
        }
        return config
