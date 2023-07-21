import jax
import jax.lax as lax
import jax.numpy as jnp

import flax
import flax.linen as nn

import numpy as np
import einops

from functools import partial
from typing import Sequence, Literal, Optional, List

from flaxsr.layers import PixelShuffle
from flaxsr._utils import register


class ShallowFeatureExtraction(nn.Module):
    n_filters: int = 64

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        x = nn.Conv(self.n_filters, (3, 3), padding='SAME')(x)
        x = nn.Conv(self.n_filters, (3, 3), padding='SAME')(x)
        return x


class ResidualDenseBlock(nn.Module):
    n_filters: int = 64
    growth_rate: int = 32
    n_layers: int = 6

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        x_skip = x
        x = [x]
        for _ in range(self.n_layers):
            x.append(
                nn.relu(
                    nn.Conv(self.growth_rate, (3, 3), padding='SAME')(jnp.concatenate(x, axis=-1))
                )
            )
        x = jnp.concatenate(x, axis=-1)
        x = nn.Conv(self.n_filters, (1, 1), padding='SAME')(x)
        return x + x_skip


class DenseFeatureFusion(nn.Module):
    n_filters: int = 64

    @nn.compact
    def __call__(self, x: Sequence[jnp.ndarray]) -> jnp.ndarray:
        x = jnp.concatenate(x, axis=-1)
        x = nn.Conv(self.n_filters, (1, 1), padding='SAME')(x)
        return x


class ResidualDenseBlocks(nn.Module):
    n_filters: int = 64
    growth_rate: int = 32
    n_blocks: int = 23
    n_rdb_layers: int = 6

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Conv(self.n_filters, (3, 3), padding='SAME')(x)

        x_skip = x
        dense_feats = []
        for _ in range(self.n_blocks):
            x = ResidualDenseBlock(self.n_filters, self.growth_rate, self.n_rdb_layers)(x)
            dense_feats.append(x)
        x = DenseFeatureFusion(self.n_filters)(dense_feats)
        x = nn.Conv(self.n_filters, (3, 3), padding='SAME')(x)
        return x + x_skip


class UPsamplingNet(nn.Module):
    n_filters: int = 64
    scale: int = 4

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Conv(self.n_filters * (self.scale ** 2), (3, 3), padding='SAME')(x)
        x = PixelShuffle(self.scale)(x)
        return x


class RDN(nn.Module):
    n_filters: int = 64
    growth_rate: int = 32
    n_blocks: int = 23
    n_rdb_layers: int = 6
    scale: int = 4

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        c = x.shape[-1]
        x = ShallowFeatureExtraction(self.n_filters)(x)
        x = ResidualDenseBlocks(self.n_filters, self.growth_rate, self.n_blocks, self.n_rdb_layers)(x)
        x = UPsamplingNet(self.n_filters, self.scale)(x)
        x = nn.Conv(c, (3, 3), padding='SAME')(x)
        return x
