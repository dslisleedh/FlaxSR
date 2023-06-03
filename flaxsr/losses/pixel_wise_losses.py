import jax
import jax.lax as lax
import jax.numpy as jnp

import flax
import flax.linen as nn

import numpy as np
import einops

from functools import partial
from typing import Sequence, Literal, Optional

from flaxsr.losses.utils import reduce_fn, Reduce, Loss, apply_mask
from flaxsr._utils import register


@partial(jax.jit, static_argnums=(3,))
def l1_loss(
        sr: jnp.ndarray, hr: jnp.ndarray, mask: Optional[jnp.ndarray] = None, reduce: str | Reduce = 'mean'
) -> jnp.ndarray:
    hr, sr = apply_mask(hr, sr, mask=mask)
    loss = jnp.abs(hr - sr)
    return reduce_fn(loss, reduce)


@register('losses', 'l1')
class L1Loss(Loss):
    def __init__(self, reduce: str | Reduce = 'mean'):
        super().__init__(reduce=reduce)

    def __call__(self, sr: jnp.ndarray, hr: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        return l1_loss(sr, hr, mask=mask, reduce=self.reduce)


@partial(jax.jit, static_argnums=(3,))
def l2_loss(
        sr: jnp.ndarray, hr: jnp.ndarray, mask: Optional[jnp.ndarray] = None, reduce: str | Reduce = 'mean'
) -> jnp.ndarray:
    hr, sr = apply_mask(hr, sr, mask=mask)
    loss = jnp.square(hr - sr)
    return reduce_fn(loss, reduce)


@register('losses', 'l2')
class L2Loss(Loss):
    def __init__(self, reduce: str | Reduce = 'mean'):
        super().__init__(reduce)

    def __call__(self, sr: jnp.ndarray, hr: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        return l2_loss(sr, hr, mask=mask, reduce=self.reduce)


@partial(jax.jit, static_argnums=(4,))
def charbonnier_loss(
        sr: jnp.ndarray, hr: jnp.ndarray,  eps: float = 1e-3, mask: Optional[jnp.ndarray] = None,
        reduce: str | Reduce = 'mean'
) -> jnp.ndarray:
    hr, sr = apply_mask(hr, sr, mask=mask)
    loss = jnp.sqrt(jnp.square(hr - sr) + eps ** 2)
    return reduce_fn(loss, reduce)


@register('losses', 'charbonnier')
class CharbonnierLoss(Loss):
    def __init__(self, eps: float = 1e-3, reduce: str | Reduce = 'mean'):
        super().__init__(reduce)
        self.eps = eps

    def __call__(self, sr: jnp.ndarray, hr: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        return charbonnier_loss(sr, hr, self.eps, mask, self.reduce)
