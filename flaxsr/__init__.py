import jax.numpy as jnp

a = jnp.ones((1,))  # To prevent CUDA-related error when import jax and tensorflow in the same process
del a

from . import layers
from . import models
from . import losses
from . import training  # Is this really necessary?

from ._utils import get
