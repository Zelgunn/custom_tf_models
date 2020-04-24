from typing import List, Union, Tuple

from energy_based import EnergyStateFunction, InputsTensor


class CombineESF(EnergyStateFunction):
    def __init__(self,
                 energy_state_functions: List[EnergyStateFunction],
                 is_low_energy: bool = None,
                 ground_truth_from_inputs: bool = None,
                 ):
        self.energy_state_functions = energy_state_functions

        if is_low_energy is None:
            is_low_energy = all([esf.is_low_energy for esf in energy_state_functions])

        self.raise_if_ground_truth_from_inputs_expect_for_last(energy_state_functions)

        if ground_truth_from_inputs is None:
            ground_truth_from_inputs = energy_state_functions[-1].ground_truth_from_inputs

        super(CombineESF, self).__init__(is_low_energy=is_low_energy,
                                         ground_truth_from_inputs=ground_truth_from_inputs)

    def call(self,
             inputs: InputsTensor
             ) -> Union[Tuple[InputsTensor, InputsTensor], InputsTensor]:
        for esf in self.energy_state_functions:
            inputs = esf.call(inputs)
        return inputs

    @staticmethod
    def raise_if_ground_truth_from_inputs_expect_for_last(energy_state_functions: List[EnergyStateFunction]):
        for esf in energy_state_functions[:-1]:
            if not esf.ground_truth_from_inputs:
                raise ValueError("At the moment, only the last provided EnergyStateFunction can "
                                 "output different ground_truth from inputs. Got `false` for {}.".format(esf))
