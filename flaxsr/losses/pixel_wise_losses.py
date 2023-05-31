import jax
import jax.lax as lax
import jax.numpy as jnp

import flax
import flax.linen as nn

import numpy as np
import einops

from functools import partial
from typing import Sequence, Literal

from flaxsr.losses.utils import reduce_fn, Reduces
from flaxsr._utils import register


def l1_loss(hr: jnp.ndarray, sr: jnp.ndarray, reduce: str | Reduces = 'mean') -> jnp.ndarray:
    loss = jnp.abs(hr - sr)
    return reduce_fn(loss, reduce)


@register('losses', 'l1')
class L1Loss:
    def __init__(self, reduce: str | Reduces = 'mean'):
        self.reduce = reduce

    def __call__(self, hr: jnp.ndarray, sr: jnp.ndarray) -> jnp.ndarray:
        return l1_loss(hr, sr, reduce=self.reduce)


def l2_loss(hr: jnp.ndarray, sr: jnp.ndarray, reduce: str | Reduces = 'mean') -> jnp.ndarray:
    loss = jnp.square(hr - sr)
    return reduce_fn(loss, reduce)


@register('losses', 'l2')
class L2Loss:
    def __init__(self, reduce: str | Reduces = 'mean'):
        self.reduce = reduce

    def __call__(self, hr: jnp.ndarray, sr: jnp.ndarray) -> jnp.ndarray:
        return l2_loss(hr, sr, reduce=self.reduce)


def charbonnier_loss(
        hr: jnp.ndarray, sr: jnp.ndarray, eps: float = 1e-3, reduce: str | Reduces = 'mean'
) -> jnp.ndarray:
    loss = jnp.sqrt(jnp.square(hr - sr) + eps ** 2)
    return reduce_fn(loss, reduce)


@register('losses', 'charbonnier')
class CharbonnierLoss:
    def __init__(self, eps: float = 1e-3, reduce: str | Reduces = 'mean'):
        self.eps = eps
        self.reduce = reduce

    def __call__(self, hr: jnp.ndarray, sr: jnp.ndarray) -> jnp.ndarray:
        return charbonnier_loss(hr, sr, eps=self.eps, reduce=self.reduce)
