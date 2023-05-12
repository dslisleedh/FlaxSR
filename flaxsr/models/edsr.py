import jax
import jax.lax as lax
import jax.numpy as jnp

import flax
import flax.linen as nn

import numpy as np
import einops

from functools import partial
from typing import Sequence, Literal

from flaxsr.layers import PixelShuffle
from flaxsr._utils import register


class ResBlock(nn.Module):
    n_filters: int
    scale_factor: float = .1
    kernel_size: Sequence[int] = (3, 3)

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        residual = nn.Conv(self.n_filters, self.kernel_size, padding='SAME')(inputs)
        residual = nn.relu(residual)
        residual = nn.Conv(self.n_filters, self.kernel_size, padding='SAME')(residual)
        return inputs + residual * self.scale_factor


class Upscale(nn.Module):
    scale: Literal[2, 3, 4]
    n_filters: int

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        if self.scale in [2, 3]:
            inputs = nn.Conv(self.n_filters * self.scale ** 2, (3, 3), padding='SAME')(inputs)
            inputs = PixelShuffle(self.scale)(inputs)
        elif self.scale == 4:
            inputs = nn.Conv(self.n_filters * 4, (3, 3), padding='SAME')(inputs)
            inputs = PixelShuffle(2)(inputs)
            inputs = nn.Conv(self.n_filters * 4, (3, 3), padding='SAME')(inputs)
            inputs = PixelShuffle(2)(inputs)
        else:
            raise ValueError(f'Invalid scale: {self.scale}, only support 2, 3, 4')
        return inputs


@register('models', 'edsr')
class EDSR(nn.Module):
    n_filters: int
    n_blocks: int
    scale: int  # Upscale ratio
    scale_factor: float = .1  # LayerScale

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        shape = inputs.shape

        feats = nn.Conv(self.n_filters, (3, 3), padding='SAME')(inputs)
        feats_skip = feats

        for _ in range(self.n_blocks):
            feats = ResBlock(self.n_filters, self.scale_factor)(feats)
        feats = feats + feats_skip

        feats = Upscale(self.scale, self.n_filters)(feats)
        out = nn.Conv(shape[-1], (3, 3), padding='SAME')(feats)
        return out


class Preprocessing(nn.Module):
    n_filters: int
    scale_factor: float = .1

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        inputs = inputs + ResBlock(self.n_filters, self.scale_factor, (5, 5))(inputs)
        inputs = inputs + ResBlock(self.n_filters, self.scale_factor, (5, 5))(inputs)
        return inputs


@register('models', 'mdsr')
class MDSR(nn.Module):
    n_filters: int
    n_blocks: int
    scale_factor: float = .1

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> Sequence[jnp.ndarray]:
        shape = inputs.shape
        feats = Preprocessing(self.n_filters, self.scale_factor)(inputs)

        feats_2x = Preprocessing(self.n_filters, self.scale_factor)(feats)
        feats_3x = Preprocessing(self.n_filters, self.scale_factor)(feats)
        feats_4x = Preprocessing(self.n_filters, self.scale_factor)(feats)

        feats = jnp.concatenate([feats_2x, feats_3x, feats_4x], axis=0)
        feats_skip = feats

        for _ in range(self.n_blocks):
            feats = ResBlock(self.n_filters, self.scale_factor)(feats)

        feats = feats + feats_skip
        feats_2x, feats_3x, feats_4x = jnp.split(feats, shape[0], axis=0)

        feats_2x = Upscale(2, self.n_filters)(feats_2x)
        feats_3x = Upscale(3, self.n_filters)(feats_3x)
        feats_4x = Upscale(4, self.n_filters)(feats_4x)

        out_2x = nn.Conv(shape[-1], (3, 3), padding='SAME')(feats_2x)
        out_3x = nn.Conv(shape[-1], (3, 3), padding='SAME')(feats_3x)
        out_4x = nn.Conv(shape[-1], (3, 3), padding='SAME')(feats_4x)

        return out_2x, out_3x, out_4x
