import jax
import jax.lax as lax
import jax.numpy as jnp

import flax
import flax.linen as nn

import numpy as np
import einops

from functools import partial
from typing import Sequence, Literal


def reduce_fn(loss, mode: Literal['sum', 'mean', None]) -> jnp.ndarray:
    if mode == 'sum':
        return jnp.sum(loss)
    elif mode == 'mean':
        return jnp.mean(loss)
    else:
        return loss
