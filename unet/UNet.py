from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Layer, Input, Dense, Flatten, Reshape, Add
from tensorflow.python.keras.layers import Conv1D, Conv2D, Conv3D
from tensorflow.python.keras.layers import Conv2DTranspose, Conv3DTranspose
from tensorflow.python.keras.layers import UpSampling1D, UpSampling2D, UpSampling3D
from tensorflow.python.keras.layers.pooling import Pooling1D, Pooling2D, Pooling3D
from typing import List, Union

from custom_tf_models.basic.AE import AE
from custom_tf_models.utils import LearningRateType
from CustomKerasLayers import Conv1DTranspose
from CustomKerasLayers import ResBlockND, ResBlockNDTranspose


class UNet(AE):
    def __init__(self,
                 encoder_layers: List[Layer],
                 output_activation=None,
                 connection_map: List[bool] = None,
                 learning_rate: LearningRateType = 1e-3,
                 name: str = None,
                 **kwargs):
        self._init_set_name(name)
        if connection_map is None:
            connection_map = [True] * len(encoder_layers)
        self.connection_map = connection_map
        decoder_layers = transpose_layers(encoder_layers, output_activation=output_activation)

        encoder = self.init_encoder(encoder_layers)
        decoder = self.init_decoder(encoder_layers, decoder_layers, encoder)

        super(UNet, self).__init__(encoder=encoder,
                                   decoder=decoder,
                                   learning_rate=learning_rate,
                                   **kwargs)

    def init_encoder(self, encoder_layers: List[Layer]) -> Model:
        input_shape = get_input_shape(encoder_layers[0])
        input_layer = Input(batch_shape=input_shape, name=self.name + "_encoder_input")

        inputs = input_layer
        outputs = []

        for layer in encoder_layers:
            output = layer(inputs)
            if has_trainable_weights(layer):
                outputs.append(output)
            inputs = output

        encoder = Model(inputs=input_layer, outputs=outputs, name=self.name + "_Encoder")
        return encoder

    def init_decoder(self, encoder_layers: List[Layer], decoder_layers: List[Layer], encoder: Model):
        depth = len(decoder_layers)

        input_layers = [
            Input(batch_shape=output.shape, name=self.name + "_decoder_input_{}".format(i))
            for i, output in enumerate(encoder.outputs)
        ]

        inputs = input_layers[-1]
        skip_index = len(input_layers) - 1
        outputs = None
        for i in range(depth):
            encoder_index = depth - i - 1
            if has_trainable_weights(encoder_layers[encoder_index]):
                if i > 0:
                    inputs = Add(name="skip_{}".format(skip_index))([inputs, input_layers[skip_index]])
                skip_index -= 1

            outputs = decoder_layers[i](inputs)
            inputs = outputs

        decoder = Model(inputs=input_layers, outputs=outputs, name=self.name + "_Decoder")
        return decoder

    def train_step(self, inputs, *args, **kwargs):
        raise NotImplementedError

    def compute_loss(self, inputs, *args, **kwargs):
        raise NotImplementedError


def transpose_layers(layers: List[Layer], output_activation=None) -> List[Layer]:
    shape = get_input_shape(layers[0])
    shapes = [shape]
    for layer in layers:
        shape = layer.compute_output_shape(shape)
        shapes.append(shape)

    transposed_layers = []
    for i in reversed(range(len(layers))):
        activation = output_activation if (i == 0) else None
        transposed_layer = transpose_layer(layers[i], shapes[i], activation=activation)
        transposed_layers.append(transposed_layer)

    return transposed_layers


def transpose_layer(layer, input_shape, activation=None):
    if isinstance(layer, ResBlockND):
        return transpose_resblock(layer, layer_input_shape=input_shape, activation=activation)

    elif isinstance(layer, (Conv1D, Conv2D, Conv3D)):
        return transpose_conv_layer(layer, layer_input_shape=input_shape, activation=activation)

    elif isinstance(layer, Dense):
        return transpose_dense_layer(layer, layer_input_shape=input_shape, activation=activation)

    elif isinstance(layer, (Pooling1D, Pooling2D, Pooling3D)):
        return transpose_pool_layer(layer)

    elif isinstance(layer, (Flatten, Reshape)):
        return transpose_reshape_layer(layer, input_shape)

    else:
        raise TypeError


def transpose_resblock(layer: ResBlockND,
                       layer_input_shape=None,
                       activation=None
                       ) -> ResBlockNDTranspose:
    output_shape = get_input_shape(layer, layer_input_shape)
    input_shape = layer.compute_output_shape(output_shape)
    activation = layer.activation if activation is None else activation

    transposed_layer = ResBlockNDTranspose(rank=layer.rank,
                                           filters=output_shape[-1],
                                           basic_block_count=layer.basic_block_count,
                                           basic_block_depth=layer.basic_block_depth,
                                           kernel_size=layer.kernel_size,
                                           strides=layer.strides,
                                           activation=activation,
                                           use_residual_bias=layer.use_residual_bias,
                                           use_conv_bias=layer.use_conv_bias,
                                           use_batch_norm=layer.use_batch_norm,
                                           kernel_initializer=layer.kernel_initializer,
                                           bias_initializer=layer.bias_initializer,
                                           name=transposed_layer_name(layer),
                                           batch_input_shape=input_shape,
                                           )
    return transposed_layer


def transpose_conv_layer(layer: Union[Conv1D, Conv2D, Conv3D],
                         layer_input_shape=None,
                         activation=None
                         ) -> Union[Conv1DTranspose, Conv2DTranspose, Conv3DTranspose]:
    output_shape = get_input_shape(layer, layer_input_shape)
    input_shape = layer.compute_output_shape(output_shape)

    conv_transpose_map = [Conv1DTranspose, Conv2DTranspose, Conv3DTranspose]
    transposed_layer_class = conv_transpose_map[layer.rank - 1]
    activation = layer.activation if activation is None else activation

    transposed_layer = transposed_layer_class(filters=output_shape[-1],
                                              kernel_size=layer.kernel_size,
                                              strides=layer.strides,
                                              use_bias=layer.use_bias,
                                              padding="same",
                                              name=transposed_layer_name(layer),
                                              activation=activation,
                                              batch_input_shape=input_shape)

    return transposed_layer


def transpose_pool_layer(layer: Union[Pooling1D, Pooling2D, Pooling3D]
                         ) -> Union[UpSampling1D, UpSampling2D, UpSampling3D]:
    transpose_map = {Pooling1D: UpSampling1D, Pooling2D: UpSampling2D, Pooling3D: UpSampling3D}

    transposed_layer_class = None
    for pool_type in transpose_map:
        if isinstance(layer, pool_type):
            transposed_layer_class = transpose_map[pool_type]

    if transposed_layer_class is None:
        raise TypeError("Layer must be a instance of PoolingND, got {}".format(type(layer)))

    size = layer.pool_size
    if isinstance(layer, Pooling1D):
        size = size[0]

    transposed_layer = transposed_layer_class(size=size, name=transposed_layer_name(layer))
    return transposed_layer


def transpose_dense_layer(layer: Dense,
                          layer_input_shape=None,
                          activation=None
                          ) -> Dense:
    output_shape = get_input_shape(layer, layer_input_shape)
    input_shape = layer.compute_output_shape(output_shape)
    units = output_shape[-1]
    activation = layer.activation if activation is None else activation

    transposed_layer = Dense(units=units,
                             activation=activation,
                             use_bias=layer.use_bias,
                             kernel_initializer=layer.kernel_initializer,
                             bias_initializer=layer.bias_initializer,
                             kernel_regularizer=layer.kernel_regularizer,
                             bias_regularizer=layer.bias_regularizer,
                             activity_regularizer=layer.activity_regularizer,
                             kernel_constraint=layer.kernel_constraint,
                             bias_constraint=layer.bias_constraint,
                             name=transposed_layer_name(layer),
                             batch_input_shape=input_shape)

    return transposed_layer


def transpose_reshape_layer(layer: Union[Flatten, Reshape], layer_input_shape=None) -> Reshape:
    output_shape = get_input_shape(layer, layer_input_shape)
    transposed_layer = Reshape(target_shape=output_shape[1:],
                               name=transposed_layer_name(layer))
    return transposed_layer


def get_input_shape(layer: Layer, default=None):
    if default is None:
        # noinspection PyProtectedMember
        input_shape = layer._batch_input_shape
    else:
        input_shape = default
    return input_shape


def transposed_layer_name(layer: Layer) -> str:
    return layer.name + "_transpose"


def has_trainable_weights(layer: Layer) -> bool:
    return len(layer.trainable_weights) > 0
