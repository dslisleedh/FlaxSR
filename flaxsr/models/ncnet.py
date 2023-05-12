import jax
import jax.lax as lax
import jax.numpy as jnp

import flax
import flax.linen as nn

import numpy as np
import einops

from functools import partial
from typing import Sequence, Literal

from flaxsr.layers import PixelShuffle, NearestConv
from flaxsr._utils import register


@register('models', 'ncnet')
class NCNet(nn.Module):
    n_filters: int
    n_blocks: int
    scale: int

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        shape = inputs.shape
        feats_skip = NearestConv(self.scale, out_c=shape[-1], return_upscaled=False)(inputs)
        feats = inputs
        for i in range(self.n_blocks):
            feats = nn.Conv(
                self.n_filters if i <= self.n_blocks - 2 else shape[-1] * self.scale ** 2, (3, 3), padding='SAME'
            )
            if i != self.n_blocks - 1:
                feats = nn.relu(feats)

        feats = feats + feats_skip
        output = PixelShuffle(self.scale)(feats)
        return output
