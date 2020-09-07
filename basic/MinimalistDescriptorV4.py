import tensorflow as tf
from tensorflow.python.keras import Model
from typing import Dict

from custom_tf_models import AE
from misc_utils.general import expand_dims_to_rank
from misc_utils.math_utils import lerp

"""
Unlike MinimalistDescriptor, all latent codes are produced in 1 step
Then, an autoregressive model produces a mask bit by bit (or a few at each step)
"""


class MinimalistDescriptorV4(AE):
    pass
