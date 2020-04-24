import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Dense, Input
from typing import List
from enum import IntEnum


class DenseFusionModelMode(IntEnum):
    ALL_TO_ONE = 0,
    ONE_TO_ONE = 1


class DenseFusionModel(Model):
    def __init__(self,
                 latent_code_sizes: List[int],
                 mode: DenseFusionModelMode
                 ):
        super(DenseFusionModel, self).__init__()

        self.latent_code_sizes = latent_code_sizes
        self.mode = mode
        self.projection_layers: List[Dense] = []

        if self.mode == DenseFusionModelMode.ALL_TO_ONE:
            self.init_all_to_one()
        elif self.mode == DenseFusionModelMode.ONE_TO_ONE:
            self.init_one_to_one()
        else:
            raise ValueError("Unknown mode : {}".format(self.mode))

        base_inputs = [Input(shape=[code_size], name="FusionInputBase_{}".format(i))
                       for i, code_size in enumerate(latent_code_sizes)]

        fuse_with_inputs = [Input(shape=[code_size], name="FusionInputFuseWith_{}".format(i))
                            for i, code_size in enumerate(latent_code_sizes)]

        inputs = [base_inputs, fuse_with_inputs]
        outputs = self.fuse(inputs)

        self._init_graph_network(inputs=inputs, outputs=outputs)

    def init_all_to_one(self):
        for i, latent_code_size in enumerate(self.latent_code_sizes):
            layer_name = "Project_All_To_{}".format(i)
            self.projection_layers.append(Dense(units=latent_code_size, activation="tanh", name=layer_name))

    def init_one_to_one(self):
        for output_mod_index, latent_code_size in enumerate(self.latent_code_sizes):
            for input_mod_index in range(len(self.latent_code_sizes)):
                if output_mod_index == input_mod_index:
                    continue
                layer_name = "Project_{}_To_{}".format(input_mod_index, output_mod_index)
                self.projection_layers.append(Dense(units=latent_code_size, activation="tanh", name=layer_name))

    def fuse(self, inputs):
        # Temporary equality
        fuse_with = inputs

        if self.mode == DenseFusionModelMode.ALL_TO_ONE:
            return self.fuse_all_to_one(inputs, fuse_with)
        elif self.mode == DenseFusionModelMode.ONE_TO_ONE:
            return self.fuse_one_to_one(inputs, fuse_with)
        else:
            raise ValueError("Unknown mode : {}".format(self.mode))

    def fuse_all_to_one(self, inputs, fuse_with=None):
        code_shapes = [tf.shape(modality_latent_code) for modality_latent_code in inputs]

        inputs_latent_codes, fuse_with_latent_codes = self.get_call_flat_latent_codes(inputs, fuse_with)

        fused_latent_codes = []
        for i in range(self.modality_count):
            latent_codes_to_fuse = []
            for j in range(self.modality_count):
                if i == j:
                    code = inputs_latent_codes[i]
                else:
                    code = fuse_with_latent_codes[i]
                latent_codes_to_fuse.append(code)
            fuse_with_latent_codes.append(tf.concat(latent_codes_to_fuse, axis=-1))

        outputs = []
        for i in range(self.modality_count):
            refined_latent_code = self.projection_layers[i](fused_latent_codes[i])
            refined_latent_code = tf.reshape(refined_latent_code, code_shapes[i])
            outputs.append(refined_latent_code)

        return outputs

    def fuse_one_to_one(self, inputs, fuse_with=None):
        code_shapes = [tf.shape(modality_latent_code) for modality_latent_code in inputs]

        inputs_latent_codes, fuse_with_latent_codes = self.get_call_flat_latent_codes(inputs, fuse_with)

        outputs = []
        i = 0
        for output_mod_index in range(self.modality_count):
            output_mod_latent_code = fuse_with_latent_codes[output_mod_index]
            refined_latent_code = inputs_latent_codes[output_mod_index]
            for input_mod_index in range(self.modality_count):
                if output_mod_index == input_mod_index:
                    continue
                refined_latent_code += self.projection_layers[i](output_mod_latent_code)

                i += 1

            refined_latent_code = tf.reshape(refined_latent_code, code_shapes[output_mod_index])
            outputs.append(refined_latent_code)

        return outputs

    def get_call_flat_latent_codes(self, inputs, fuse_with=None):
        inputs_latent_codes = self.get_flat_latent_codes(inputs)
        if fuse_with is None:
            fuse_with_latent_codes = inputs_latent_codes
        else:
            fuse_with_latent_codes = self.get_flat_latent_codes(fuse_with)
        return inputs_latent_codes, fuse_with_latent_codes

    def get_flat_latent_codes(self, latent_codes):
        batch_size = tf.shape(latent_codes[0])[0]
        flat_latent_codes = []
        for i in range(self.modality_count):
            latent_code = tf.reshape(latent_codes[i], [batch_size, self.latent_code_sizes[i]])
            flat_latent_codes.append(latent_code)
        return flat_latent_codes

    @property
    def modality_count(self):
        return len(self.projection_layers)
