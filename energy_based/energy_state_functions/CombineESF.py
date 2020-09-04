from typing import List, Union, Tuple

from custom_tf_models.energy_based import EnergyStateFunction, InputsTensor


class CombineESF(EnergyStateFunction):
    def __init__(self,
                 energy_state_functions: List[EnergyStateFunction],
                 is_low_energy: bool = None):
        self.energy_state_functions = energy_state_functions

        if is_low_energy is None:
            is_low_energy = all([esf.is_low_energy for esf in energy_state_functions])

        super(CombineESF, self).__init__(is_low_energy=is_low_energy)

    def call(self,
             inputs: InputsTensor
             ) -> Union[Tuple[InputsTensor, InputsTensor], InputsTensor]:
        for esf in self.energy_state_functions:
            inputs = esf.call(inputs)
        return inputs
