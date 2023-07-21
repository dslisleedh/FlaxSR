import jax
import jax.lax as lax
import jax.numpy as jnp

import flax
import flax.linen as nn

import numpy as np
import einops

from functools import partial
from typing import Sequence, Literal, Optional

from flaxsr.layers import PixelShuffle
from flaxsr._utils import register


class RCAB(nn.Module):
    r: Optional[int]

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        c = x.shape[-1]

        x = nn.Conv(c, (3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(c, (3, 3), padding='SAME')(x)

        if self.r is not None:
            mu = jnp.mean(x, axis=(1, 2), keepdims=True)
            mu = nn.Dense(c // self.r)(mu)
            mu = nn.relu(mu)
            mu = nn.Dense(c)(mu)
            mu = nn.sigmoid(mu)
            x = x * mu
        else:
            x = x * 0.1  # Residual scaling

        return x


class ResidualGroup(nn.Module):
    r: Optional[int]
    n: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        res = x
        for _ in range(self.n):
            res = RCAB(self.r)(res)
        res = nn.Conv(x.shape[-1], (3, 3), padding='SAME')(res)
        return x + res


@register('models', 'rcan')
class RCAN(nn.Module):
    n_filters: int
    n_groups: int
    n_blocks: int
    r: Optional[int]
    scale: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        out_c = x.shape[-1]
        feat = nn.Conv(self.n_filters, (3, 3), padding='SAME')(x)
        res = feat

        for _ in range(self.n_groups):
            res = ResidualGroup(self.r, self.n_blocks)(res)
        res = nn.Conv(self.n_filters, (3, 3), padding='SAME')(res)
        feat = res + feat

        feat = PixelShuffle(self.scale)(feat)
        out = nn.Conv(out_c, (3, 3), padding='SAME')(feat)
        return out
