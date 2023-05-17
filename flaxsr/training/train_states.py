import jax
import jax.lax as lax
import jax.numpy as jnp

import flax
import flax.linen as nn
from flax.training import train_state

import numpy as np
import einops

from functools import partial
from typing import Sequence, Literal

from flaxsr._utils import register
from flaxsr.losses.utils import loss_wrapper


class TrainState(train_state.TrainState):
    """
    Custom TrainState for the SR model.
    Include losses inside to be able to use them in the training loop.
    """
    losses: loss_wrapper
    reduce: Literal['sum', 'mean', None]
