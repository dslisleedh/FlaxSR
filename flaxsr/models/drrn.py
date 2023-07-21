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


class BAC(nn.Module):
    n_filters: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        x = nn.BatchNorm()(x, use_running_average=not training)
        x = nn.relu(x)
        x = nn.Conv(self.n_filters, (3, 3), padding='SAME')(x)
        return x


class ResidualBlock(nn.Module):
    U: int
    n_filters: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        res = nn.Conv(self.n_filters, (3, 3), padding='SAME')(x)

        for _ in range(self.U):
            res = res + BAC(self.n_filters)(res, training=training)

        return res + x


@register('models', 'drrn')
class DRRN(nn.Module):
    B: int
    U: int
    n_filters: int
    scale: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        lr_shape = x.shape
        hr_shape = (lr_shape[0], lr_shape[1] * self.scale, lr_shape[2] * self.scale, lr_shape[3])

        x = jax.image.resize(x, hr_shape, method='bicubic')
        res = x

        for _ in range(self.B):
            res = ResidualBlock(self.U, self.n_filters)(res, training=training)

        x = x + res
        return x
