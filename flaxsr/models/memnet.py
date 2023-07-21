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


class BAC(nn.Module):  # In paper, they not refer to this as BAC. Just a name I gave.
    n_filters: int = 64
    kernel_size: int = 3

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        x = nn.BatchNorm()(x, use_running_average=not training)
        x = nn.relu(x)
        x = nn.Conv(self.n_filters, (self.kernel_size, self.kernel_size), padding='SAME')(x)
        return x


class ResidualBlock(nn.Module):
    n_filters: int = 64
    n_res_blocks: int = 2

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False):
        for _ in range(self.n_res_blocks):
            x = x + BAC(self.n_filters)(x, training=training)
        return x


class MEMBlock(nn.Module):
    n_filters: int = 64
    n_res_blocks: int = 2
    n_recursions: int = 6

    @nn.compact
    def __call__(
            self, x: jnp.ndarray, long_transitions: List, training: bool = False
    ) -> jnp.ndarray:
        concat_feat = [x]
        for _ in range(self.n_recursions):
            x = ResidualBlock(self.n_filters, self.n_res_blocks)(x, training=training)
            concat_feat.append(x)

        concat_feat = jnp.concatenate(long_transitions + concat_feat, axis=-1)
        refined = BAC(self.n_filters, kernel_size=1)(concat_feat, training=training)
        return refined


class MEMNet(nn.Module):
    n_filters: int = 64
    n_res_blocks: int = 2
    n_recursions: int = 6
    n_mem_blocks: int = 6
    scale: int = 4

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        lr_shape = x.shape
        c = lr_shape[-1]
        hr_shape = (lr_shape[0], lr_shape[1] * self.scale, lr_shape[2] * self.scale, lr_shape[3])
        x = jax.image.resize(x, hr_shape, method='bicubic')
        x_skip = x

        # Output Weight
        w = self.param(  # B, H, W, C, N_mem_blocks
            'w', nn.initializers.constant(1. / float(self.n_mem_blocks)), (1, 1, 1, c, self.n_mem_blocks))

        # Feat projection
        feat = BAC(self.n_filters)(x, training=training)

        # MEM Blocks
        long_transitions = []
        for _ in range(self.n_mem_blocks):
            feat = MEMBlock(self.n_filters, self.n_res_blocks, self.n_recursions)(
                feat, long_transitions, training=training)
            long_transitions.append(feat)

        # Reconstruction
        recons = [
            BAC(c)(intermediate_feat, training=training) + x_skip for intermediate_feat in long_transitions
        ]
        recons = jnp.concatenate(recons, axis=-1)
        recons = jnp.sum(recons * w, axis=-1)
        return recons
