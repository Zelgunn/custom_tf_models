import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Lambda, TimeDistributed
from typing import Optional, Dict, Union, List

from custom_tf_models.basic.AE import AE
from transformers import Transformer


class CNNTransformer(Model):
    def __init__(self,
                 input_length: int,
                 output_length: int,
                 autoencoder: AE,
                 transformer: Transformer,
                 autoencoder_input_shape: Union[tf.TensorShape, List[int]] = None,
                 train_only_embeddings=True,
                 **kwargs):
        super(CNNTransformer, self).__init__(**kwargs)

        if autoencoder_input_shape is None:
            if autoencoder.encoder.built:
                autoencoder_input_shape = autoencoder.encoder.input_shape
            else:
                raise ValueError("You must provide `autoencoder_input_shape` when"
                                 " the encoder of the autoencoder is not built.")

        self.input_length = input_length
        self.output_length = output_length
        self.autoencoder_input_shape = autoencoder_input_shape
        self.autoencoder = autoencoder
        self.transformer = transformer
        self.train_only_embeddings = train_only_embeddings
        if train_only_embeddings:
            autoencoder.trainable = False

        # region Layers
        self.split_inputs = Lambda(lambda x: self.split_steps(x, True), name="SplitInputs")
        self.split_outputs = Lambda(lambda x: self.split_steps(x, False), name="SplitOutputs")

        self.merge_outputs = Lambda(self.merge_steps, name="MergeOutputs")

        encoded_shape = autoencoder.compute_encoded_shape(autoencoder_input_shape)
        parts_shape = [output_length, *encoded_shape[1:-1]]
        self.parts_to_batch = Lambda(parts_to_batch, name="PartsToBatch")
        self.batch_to_parts = Lambda(lambda x: batch_to_parts(x, parts_shape), name="BatchToParts")

        self.step_encoder = TimeDistributed(Lambda(autoencoder.encode), name="StepEncoder")
        self.step_decoder = TimeDistributed(Lambda(autoencoder.decode), name="StepDecoder")
        # endregion

        self.transformer_evaluator = transformer.make_evaluator(output_length)
        self._evaluator: Optional[tf.keras.models.Sequential] = None

    @tf.function
    def call(self, inputs, training=None, mask=None):
        encoder_embedding_input = self.split_inputs(inputs)
        decoder_embedding_input = self.split_outputs(inputs)

        encoder_latent_code = self.step_encoder(encoder_embedding_input)
        decoder_latent_code = self.step_encoder(decoder_embedding_input)

        encoder_latent_code = self.parts_to_batch(encoder_latent_code)
        decoder_latent_code = self.parts_to_batch(decoder_latent_code)

        transformed = self.transformer([encoder_latent_code, decoder_latent_code])

        decoded = self.batch_to_parts(transformed)
        decoded = self.step_decoder(decoded)
        decoded = self.merge_outputs(decoded)

        return decoded

    @tf.function
    def train_step(self, inputs) -> Dict[str, tf.Tensor]:
        with tf.GradientTape() as tape:
            metrics = self.compute_loss(inputs)
            loss = metrics["loss"]
            if self.transformer.add_copy_regularization:
                loss += metrics["copy_regularization"]

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        if self.transformer.add_copy_regularization and self.train_only_embeddings:
            loss, copy_regularization = loss
            copy_regularization /= self.transformer.copy_regularization_factor
            metrics["copy_regularization"] = copy_regularization

        return metrics

    @tf.function
    def test_step(self, inputs) -> Dict[str, tf.Tensor]:
        return self.compute_loss(inputs)

    # region Loss
    @tf.function
    def compute_loss(self, inputs) -> Dict[str, tf.Tensor]:
        if self.train_only_embeddings:
            return self.compute_embeddings_loss(inputs)
        else:
            return self.compute_reconstruction_loss(inputs)

    @tf.function
    def compute_embeddings_loss(self, inputs: tf.Tensor) -> Dict[str, tf.Tensor]:
        encoder_embedding_input = self.split_inputs(inputs)
        decoder_embedding_input = self.split_outputs(inputs)

        encoder_latent_code = self.step_encoder(encoder_embedding_input)
        decoder_latent_code = self.step_encoder(decoder_embedding_input)

        encoder_latent_code = self.parts_to_batch(encoder_latent_code)
        decoder_latent_code = self.parts_to_batch(decoder_latent_code)

        transformer_loss = self.transformer.compute_loss(encoder_latent_code, decoder_latent_code)
        return transformer_loss

    @tf.function
    def compute_reconstruction_loss(self, inputs: tf.Tensor) -> Dict[str, tf.Tensor]:
        encoder_embedding_input = self.split_inputs(inputs)
        decoder_embedding_input = self.split_outputs(inputs)

        encoder_latent_code = self.step_encoder(encoder_embedding_input)
        decoder_latent_code = self.step_encoder(decoder_embedding_input)

        encoder_latent_code = self.parts_to_batch(encoder_latent_code)
        decoder_latent_code = self.parts_to_batch(decoder_latent_code)

        transformed = self.transformer([encoder_latent_code, decoder_latent_code])

        decoded = self.batch_to_parts(transformed)
        decoded = self.step_decoder(decoded)

        reconstruction_loss = tf.reduce_mean(tf.square(decoder_embedding_input - decoded))
        return {"loss": reconstruction_loss}

    # endregion

    # @property
    # def anomaly_metrics(self) -> List:
    #     @tf.function
    #     def embeddings_loss(_, inputs):
    #         transformer_loss = self.compute_embeddings_loss(inputs)
    #         if self.transformer.add_copy_regularization:
    #             transformer_loss = transformer_loss[0]
    #         return transformer_loss
    #
    #     @tf.function
    #     def combined_loss(_, inputs):
    #         return self.compute_reconstruction_loss(inputs) + embeddings_loss(_, inputs)
    #
    #     return [embeddings_loss, combined_loss]

    # region Split / Merge steps
    @tf.function
    def split_steps(self, tensor: tf.Tensor, is_inputs: bool) -> tf.Tensor:
        step_count = self.input_length + self.output_length

        tensor_shape = tf.shape(tensor)
        batch_size, length, *dimensions = tf.unstack(tensor_shape)
        step_size = length // step_count
        tensor = tf.reshape(tensor, [batch_size, step_count, step_size, *dimensions])

        if is_inputs:
            tensor = tensor[:, :self.input_length]
        else:
            tensor = tensor[:, -self.output_length:]

        return tensor

    @staticmethod
    def merge_steps(tensor: tf.Tensor) -> tf.Tensor:
        tensor_shape = tf.shape(tensor)
        batch_size, step_count, step_size, *dimensions = tf.unstack(tensor_shape)
        tensor = tf.reshape(tensor, [batch_size, step_count * step_size, *dimensions])
        return tensor

    # endregion

    # region Evaluate sequence
    @tf.function
    def evaluate_sequence(self, inputs: tf.Tensor) -> tf.Tensor:
        inputs = self.split_inputs(inputs)
        encoder_latent_code = self.step_encoder(inputs)
        encoder_latent_code = self.parts_to_batch(encoder_latent_code)

        transformed = self.transformer_evaluator(encoder_latent_code)

        decoded = self.batch_to_parts(transformed)
        decoded = self.step_decoder(decoded)
        decoded = self.merge_outputs(decoded)

        return decoded

    @property
    def evaluator(self) -> tf.keras.models.Sequential:
        if self._evaluator is None:
            self._evaluator = tf.keras.models.Sequential(layers=[
                Lambda(self.evaluate_sequence)
            ],
                name="{}Evaluator".format(self.name))

            if self._evaluator in self._layers:
                self._layers.remove(self._evaluator)

        return self._evaluator

    # endregion

    def get_config(self):
        config = {
            "input_length": self.input_length,
            "output_length": self.output_length,
            "autoencoder": self.autoencoder.get_config(),
            "transformer": self.transformer.get_config(),
        }
        return config


def parts_to_batch(tensor):
    batch_size, length, *dimensions, features = tf.unstack(tf.shape(tensor))
    total_dimension = tf.reduce_prod(dimensions)
    tensor = tf.reshape(tensor, [batch_size, length, total_dimension, features])
    tensor = tf.transpose(tensor, perm=[0, 2, 1, 3])
    tensor = tf.reshape(tensor, [batch_size * total_dimension, length, features])
    return tensor


def batch_to_parts(tensor, parts_shape):
    length, *dimensions = parts_shape

    features = tf.shape(tensor)[-1]
    total_dimension = tf.reduce_prod(dimensions)
    tensor = tf.reshape(tensor, [-1, total_dimension, length, features])
    tensor = tf.transpose(tensor, perm=[0, 2, 1, 3])
    tensor = tf.reshape(tensor, [-1, length, *dimensions, features])
    return tensor
