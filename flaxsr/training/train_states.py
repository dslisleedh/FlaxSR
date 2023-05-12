import jax
import jax.lax as lax
import jax.numpy as jnp

import flax
import flax.linen as nn
from flax.training.train_state import TrainState

import numpy as np
import einops

from functools import partial
from typing import Sequence, Literal

from flaxsr._utils import register
from flaxsr.losses.utils import loss_wrapper


class CustomTrainState(TrainState):
    losses: loss_wrapper
    reduce: Literal['sum', 'mean', None]
