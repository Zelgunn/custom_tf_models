from energy_based import EnergyStateFunction


class IdentityESF(EnergyStateFunction):
    def __init__(self):
        super(IdentityESF, self).__init__(is_low_energy=True,
                                          ground_truth_from_inputs=True)

    def call(self, inputs):
        return inputs
