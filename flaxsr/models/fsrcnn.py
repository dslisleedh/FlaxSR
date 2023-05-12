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


@register('models', 'fsrcnn')
class FSRCNN(nn.Module):
    """
    Original paper only upscale Y channel of YCbCr image.
    """
    d: int
    s: int
    m: int
    scale: int

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, **kwargs) -> jnp.ndarray:
        shape = inputs.shape
        inputs = nn.Conv(self.d, (5, 5), padding='SAME')(inputs)
        inputs = nn.activation.PReLU(inputs)
        inputs = nn.Conv(self.s, (1, 1), padding='VALID')(inputs)
        inputs = nn.activation.PReLU(inputs)
        for _ in range(self.m):
            inputs = nn.Conv(self.s, (3, 3), padding='SAME')(inputs)
        inputs = nn.activation.PReLU(inputs)
        inputs = nn.Conv(self.d, (1, 1), padding='VALID')(inputs)
        inputs = nn.activation.PReLU(inputs)
        inputs = nn.ConvTranspose(shape[-1], (9, 9), strides=(self.scale, self.scale), padding='SAME')(inputs)
        return inputs
