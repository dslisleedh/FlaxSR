import jax
import jax.lax as lax
import jax.numpy as jnp

import flax
import flax.linen as nn

import numpy as np
import einops

from functools import partial
from typing import Sequence, Literal

from flaxsr._utils import register


@register('models', 'vdsr')
class VDSR(nn.Module):
    n_filters: int
    n_blocks: int
    scale: int

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        shape = inputs.shape
        inputs = jax.image.resize(
            inputs, (shape[0], shape[1] * self.scale, shape[2] * self.scale, shape[3]), method='bicubic'
        )
        inputs_skip = inputs
        for i in range(self.n_blocks):
            if (i + 1) != self.n_blocks:
                inputs = nn.Conv(self.n_filters, (3, 3), padding='SAME')(inputs)
                inputs = nn.relu(inputs)
            else:
                inputs = nn.Conv(shape[-1], (3, 3), padding='SAME')(inputs)
        return inputs + inputs_skip
