import tensorflow as tf
from abc import abstractmethod
from typing import Union, List, Tuple

InputsTensor = Union[tf.Tensor, List[tf.Tensor], List[List[tf.Tensor]]]


class EnergyStateFunction(object):
    def __init__(self,
                 is_low_energy: bool,
                 ground_truth_from_inputs: bool,
                 ):
        self.is_low_energy = is_low_energy
        self.ground_truth_from_inputs = ground_truth_from_inputs

    def __call__(self,
                 inputs: InputsTensor
                 ) -> Tuple[InputsTensor, InputsTensor]:
        state = self.call(inputs)
        if self.ground_truth_from_inputs:
            inputs = ground_truth = state
        else:
            inputs, ground_truth = state
        return inputs, ground_truth

    @abstractmethod
    def call(self,
             inputs: InputsTensor
             ) -> Union[Tuple[InputsTensor, InputsTensor], InputsTensor]:
        raise NotImplementedError


