import jax
import jax.lax as lax
import jax.numpy as jnp

import flax
import flax.linen as nn

import numpy as np
import einops

from functools import partial
from typing import Sequence, Literal, Optional

from flaxsr.losses.utils import reduce_fn, Reduce, Loss
from flaxsr._utils import register


@partial(jax.jit, static_argnums=(2,))
def l1_loss(
        sr: jnp.ndarray, hr: jnp.ndarray, reduce: str | Reduce = 'mean'
) -> jnp.ndarray:
    loss = jnp.abs(hr - sr)
    return reduce_fn(loss, reduce)


@register('losses', 'l1')
class L1Loss(Loss):
    def __init__(self, reduce: str | Reduce = 'mean', weight: float = 1.):
        super().__init__(reduce, weight)

    def __call__(self, sr: jnp.ndarray, hr: jnp.ndarray) -> jnp.ndarray:
        return l1_loss(sr, hr, reduce=self.reduce) * self.weight


@partial(jax.jit, static_argnums=(2,))
def l2_loss(
        sr: jnp.ndarray, hr: jnp.ndarray, reduce: str | Reduce = 'mean'
) -> jnp.ndarray:
    loss = jnp.square(hr - sr)
    return reduce_fn(loss, reduce)


@register('losses', 'l2')
class L2Loss(Loss):
    def __init__(self, reduce: str | Reduce = 'mean', weight: float = 1.):
        super().__init__(reduce, weight)

    def __call__(self, sr: jnp.ndarray, hr: jnp.ndarray) -> jnp.ndarray:
        return l2_loss(sr, hr, reduce=self.reduce) * self.weight


@partial(jax.jit, static_argnums=(3,))
def charbonnier_loss(
        sr: jnp.ndarray, hr: jnp.ndarray, eps: float = 1e-3, reduce: str | Reduce = 'mean'
) -> jnp.ndarray:
    loss = jnp.sqrt(jnp.square(hr - sr) + eps ** 2)
    return reduce_fn(loss, reduce)


@register('losses', 'charbonnier')
class CharbonnierLoss(Loss):
    def __init__(self, eps: float = 1e-3, reduce: str | Reduce = 'mean', weight: float = 1.):
        super().__init__(reduce, weight)
        self.eps = eps

    def __call__(self, sr: jnp.ndarray, hr: jnp.ndarray) -> jnp.ndarray:
        return charbonnier_loss(sr, hr, self.eps, self.reduce) * self.weight


@partial(jax.jit, static_argnums=(3, 6))
def outlier_aware_loss(
        sr: jnp.ndarray, hr: jnp.ndarray, alpha: float = 0.1, p: int = 1,
        mu: float = 1., b: float = 1., reduce: str | Reduce = 'mean'
) -> jnp.ndarray:
    if p == 1:
        delta = jnp.abs(hr - sr)
    else:
        delta = jnp.power(hr - sr, p)
    loss = delta * (1 - jnp.exp(-alpha * (delta - mu) / b))
    return reduce_fn(loss, reduce)


@register('losses', 'outlier_aware')
class OutlierAwareLoss(Loss):
    def __init__(
            self, alpha: float = 0.1, p: int = 1, mu: float = 1., b: float = 1.,
            reduce: str | Reduce = 'mean', weight: float = 1.
    ):
        super().__init__(reduce, weight)
        self.alpha = alpha
        self.p = p
        self.mu = mu
        self.b = b

    def __call__(self, sr: jnp.ndarray, hr: jnp.ndarray) -> jnp.ndarray:
        return outlier_aware_loss(sr, hr, self.alpha, self.p, self.mu, self.b, self.reduce) * self.weight
