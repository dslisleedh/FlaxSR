import jax
import jax.lax as lax
import jax.numpy as jnp

import flax
import flax.linen as nn

import numpy as np
import einops

from functools import partial
from typing import Sequence, Literal

from flaxsr.losses.utils import reduce_fn
from flaxsr._utils import register


@register('losses', 'l1')
def l1_loss(hr: jnp.ndarray, sr: jnp.ndarray, mode: Literal['sum', 'mean', None] = 'mean') -> jnp.ndarray:
    loss = jnp.abs(hr - sr)
    return reduce_fn(loss, mode)


@register('losses', 'l2')
def l2_loss(hr: jnp.ndarray, sr: jnp.ndarray, mode: Literal['sum', 'mean', None] = 'mean') -> jnp.ndarray:
    loss = jnp.square(hr - sr)
    return reduce_fn(loss, mode)


@register('losses', 'charbonnier')
def charbonnier_loss(
        hr: jnp.ndarray, sr: jnp.ndarray, eps: float = 1e-3, mode: Literal['sum', 'mean', None] = 'mean'
) -> jnp.ndarray:
    loss = jnp.sqrt(jnp.square(hr - sr) + eps ** 2)
    return reduce_fn(loss, mode)
