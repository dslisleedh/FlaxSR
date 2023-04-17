import jax
import jax.lax as lax
import jax.numpy as jnp

import flax
import flax.linen as nn

import numpy as np
import einops

from functools import partial
from typing import Sequence, Literal, Optional

from FlaxSR.layers import PixelShuffle


"""
Implemented SRResNet only.
SRGAN will be implemented later.
"""


class ResBlock(nn.Module):

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, training: bool = False, **kwargs) -> jnp.ndarray:
        shape = inputs.shape
        residual = nn.Conv(shape[-1], (3, 3), padding='SAME')(inputs)
        residual = nn.BatchNorm()(residual, use_running_average=not training)
        residual = nn.PReLU(residual)
        residual = nn.Conv(shape[-1], (3, 3), padding='SAME')(residual)
        residual = nn.BatchNorm()(residual, use_running_average=not training)
        return inputs + residual


class SRResNet(nn.Module):
    n_filters: int
    n_blocks: int
    scale: int

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, training: bool, **kwargs) -> jnp.ndarray:
        shape = inputs.shape
        feats = nn.Conv(self.n_filters, (9, 9), padding='SAME')(inputs)
        feats = nn.PReLU(feats)
        feats_skip = feats

        for _ in range(self.n_blocks):
            feats = ResBlock()(feats, training=training)
        feats = nn.Conv(self.n_filters, (3, 3), padding='SAME')(feats)
        feats = nn.BatchNorm()(feats, use_running_average=not training)
        feats = feats + feats_skip

        if self.scale in [2, 3]:
            feats = nn.Conv(self.n_filters * (self.scale ** 2), (3, 3), padding='SAME')(feats)
            feats = PixelShuffle(self.scale)(feats)
            feats = nn.PReLU(feats)
        elif self.scale in [4, 8]:
            for _ in range(int(np.log2(self.scale))):
                feats = nn.Conv(self.n_filters * 4, (3, 3), padding='SAME')(feats)
                feats = PixelShuffle(2)(feats)
                feats = nn.PReLU(feats)
        else:
            raise ValueError(f"Scale {self.scale} is not supported., use 2, 3, 4, 8.")

        out = nn.Conv(shape[-1], (9, 9), padding='SAME')(feats)
        return out
    