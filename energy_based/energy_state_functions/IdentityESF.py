from custom_tf_models.energy_based import EnergyStateFunction


class IdentityESF(EnergyStateFunction):
    def __init__(self):
        super(IdentityESF, self).__init__(is_low_energy=True)

    def call(self, inputs):
        return inputs
