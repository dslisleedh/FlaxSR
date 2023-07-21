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


class SE(nn.Module):
    sr: float = .25

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        c = x.shape[-1]

        mu = jnp.mean(x, axis=[1, 2], keepdims=True)
        mu = nn.Dense(c // self.sr)(mu)
        mu = nn.gelu(mu)
        score = nn.Dense(c)(mu)
        attn = nn.sigmoid(score)

        return x * attn


class MLP(nn.Module):
    gr: int = 2

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        c = x.shape[-1]

        x = nn.Dense(c * self.gr)(x)
        x = nn.gelu(x)
        x = nn.Dense(c)(x)

        return x


# Conv1x1 -> DW Conv 3x3 -> SE -> Conv1x1
class MBConv(nn.Module):
    gr: int = 2

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        c = x.shape[-1]

        x = nn.Conv(c * self.gr, (1, 1))(x)
        x = nn.gelu(x)
        x = nn.Conv(c * self.gr, (3, 3), padding='SAME', feature_group_count=c * self.gr)(x)
        x = nn.gelu(x)
        x = SE()(x)
        x = nn.Conv(c, (1, 1))(x)

        return x


class CCM(nn.Module):
    gr: int = 2

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        c = x.shape[-1]

        x = nn.Conv(c * self.gr, (3, 3), padding='SAME')(x)
        x = nn.gelu(x)
        x = nn.Conv(c, (1, 1))(x)

        return x


class SAFM(nn.Module):
    n_levels: int = 4

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        b, h, w, c = x.shape
        chunk_size = c // self.n_levels

        xs = jnp.split(x, chunk_size, axis=-1)
        out = []
        for i in range(self.n_lavels):
            if i > 0:
                pool_size = (h // 2 ** i, w // 2 ** i)
                s = nn.max_pool(xs[i], pool_size=pool_size, strides=pool_size)
                s = nn.Conv(chunk_size, (3, 3), padding='SAME', feature_group_count=chunk_size)(s)
                s = jax.image.resize(s, shape=(b, h, w, chunk_size), method='nearest')
            else:
                s = nn.Conv(chunk_size, (3, 3), padding='SAME', feature_group_count=chunk_size)(xs[i])
            out.append(s)

        out = nn.Conv(c, (1, 1))(jnp.concatenate(out, axis=-1))
        out = nn.gelu(out) * x
        return out


class AttBlock(nn.Module):
    gr: int = 2

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = SAFM()(nn.LayerNorm()(x)) + x
        x = CCM(self.gr)(nn.LayerNorm()(x)) + x
        return x


@register('models', 'safmn')
class SAFMN(nn.Module):
    n_filters: int = 64
    n_levels: int = 4
    scale: int = 4

    @nn.compact
    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        out_c = x.shape[-1]

        feat = nn.Conv(self.n_filters, (3, 3), padding='SAME')(x)
        skip = feat

        for _ in range(self.n_levels):
            feat = AttBlock()(feat)
        feat = feat + skip

        out = nn.Conv(out_c * (self.scale ** 2), (3, 3), padding='SAME')(feat)
        out = PixelShuffle(scale=self.scale)(out)
        return out
