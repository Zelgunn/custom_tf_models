from tensorflow.python.keras.models import Model

from custom_tf_models.description_length.LED import LED
from custom_tf_models.utils import LearningRateType


# LED : Adversarial Low Energy Descriptors
class ALED(LED):
    def __init__(self,
                 encoder: Model,
                 decoder: Model,
                 generator: Model,
                 learning_rate: LearningRateType,
                 features_per_block: int,
                 merge_dims_with_features=False,
                 binarization_temperature=50.0,
                 add_binarization_noise_to_mask=False,
                 description_energy_loss_lambda=1e-2,
                 seed=None,
                 **kwargs
                 ):
        super(ALED, self).__init__(encoder=encoder,
                                   decoder=decoder,
                                   learning_rate=learning_rate,
                                   features_per_block=features_per_block,
                                   merge_dims_with_features=merge_dims_with_features,
                                   binarization_temperature=binarization_temperature,
                                   add_binarization_noise_to_mask=add_binarization_noise_to_mask,
                                   description_energy_loss_lambda=description_energy_loss_lambda,
                                   seed=seed,
                                   **kwargs)
        self.generator = generator
