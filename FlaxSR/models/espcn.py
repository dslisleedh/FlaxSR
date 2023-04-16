import jax
import jax.lax as lax
import jax.numpy as jnp

import flax
import flax.linen as nn

import numpy as np
import einops

from functools import partial
from typing import Sequence, Literal

from FlaxSR.layers import PixelShuffle


class ESPCN(nn.Module):
    n_filters: Sequence[int]
    kernel_size: Sequence[int]
    scale: int

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        shape = inputs.shape
        inputs = nn.Conv(self.n_filters[0], self.kernel_size[0], padding='SAME')(inputs)
        inputs = nn.relu(inputs)
        inputs = nn.Conv(self.n_filters[1], self.kernel_size[1], padding='SAME')(inputs)
        inputs = nn.relu(inputs)
        inputs = nn.Conv(shape[-1] * self.scale ** 2, self.kernel_size[1], padding='SAME')(inputs)
        inputs = PixelShuffle(self.scale)(inputs)
        return inputs
