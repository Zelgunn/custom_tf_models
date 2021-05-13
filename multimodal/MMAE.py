# MMAE : Multi-modal Autoencoder
import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Input, Concatenate, Flatten, Reshape
from tensorflow.python.keras.losses import mean_absolute_error
import numpy as np
from typing import List, Union, Callable

from custom_tf_models.basic.AE import AE
from CustomKerasLayers import SplitLayer


class MMAE(AE):
    def __init__(self,
                 encoders: List[Model],
                 decoders: List[Model],
                 fusion_model: Model,
                 reconstruction_loss_function: Callable = mean_absolute_error,
                 multi_modal_loss=False,
                 **kwargs):
        name = kwargs["name"] if "name" in kwargs else "MMAE"
        encoder = self.join_encoders(encoders, fusion_model, name)
        decoder = self.join_decoders(decoders, fusion_model, name)

        super(MMAE, self).__init__(encoder=encoder,
                                   decoder=decoder,
                                   reconstruction_loss_function=reconstruction_loss_function,
                                   **kwargs)

        self.encoders = encoders
        self.decoders = decoders
        self.fusion_model = fusion_model
        self.multi_modal_loss = multi_modal_loss

    @staticmethod
    def join_encoders(encoders: List[Model], fusion_model: Model, name: str) -> Model:
        inputs = [encoder.input for encoder in encoders]
        codes = [encoder.output for encoder in encoders]

        if len(fusion_model.input_shape) == 2:
            for i in range(len(codes)):
                if len(codes[i].shape) != 2:
                    codes[i] = Flatten()(codes[i])
            codes = Concatenate()(codes)

        output = fusion_model(codes)
        return Model(inputs=inputs, outputs=[output], name=name)

    @staticmethod
    def join_decoders(decoders: List[Model], fusion_model: Model, name: str) -> Model:
        code_shapes = [decoder.input_shape[1:] for decoder in decoders]
        code_sizes = [int(np.prod(code_shape)) for code_shape in code_shapes]

        input_layer = Input(batch_input_shape=fusion_model.output_shape)
        codes = SplitLayer(num_or_size_splits=code_sizes, axis=-1, num=len(decoders))(input_layer)

        outputs = []
        for decoder, code, code_shape in zip(decoders, codes, code_shapes):
            reshape_layer = Reshape(target_shape=code_shape)
            code = reshape_layer(code)
            decoded = decoder(code)
            outputs.append(decoded)

        return Model(inputs=[input_layer], outputs=outputs, name=name)

    @tf.function
    def compute_reconstruction_loss(self,
                                    inputs: List[tf.Tensor],
                                    outputs: List[tf.Tensor],
                                    axis=None) -> Union[tf.Tensor, List[tf.Tensor]]:
        if self.multi_modal_loss:
            return super(MMAE, self).compute_reconstruction_loss(inputs, outputs)

        loss = []
        for i in range(len(inputs)):
            modality_loss = super(MMAE, self).compute_reconstruction_loss(inputs[i], outputs[i])
            if axis is not None:
                modality_loss = tf.reduce_mean(modality_loss, axis=axis)
            else:
                modality_loss = tf.reduce_mean(modality_loss)
            loss.append(modality_loss)

        if axis is None:
            loss = tf.reduce_mean(loss)

        return loss

    def get_config(self):
        super_config = super(MMAE, self).get_config()
        return {
            **super_config,
            "multi_modal_loss": self.multi_modal_loss,
        }
