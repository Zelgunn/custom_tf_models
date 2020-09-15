import tensorflow as tf
from abc import abstractmethod
from typing import Union, List, Tuple

InputsTensor = Union[tf.Tensor, List[tf.Tensor], List[List[tf.Tensor]]]


class EnergyStateFunction(object):
    def __init__(self, is_low_energy: bool):
        self.is_low_energy = is_low_energy

    def __call__(self,
                 inputs: InputsTensor
                 ) -> Tuple[InputsTensor, InputsTensor]:
        state = self.call(inputs)
        return state

    @abstractmethod
    def call(self,
             inputs: InputsTensor
             ) -> Union[Tuple[InputsTensor, InputsTensor], InputsTensor]:
        raise NotImplementedError
