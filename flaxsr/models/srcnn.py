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


@register('models', 'srcnn')
class SRCNN(nn.Module):
    """
    I fixed n_layers as 3 and kernel_size as 9, 1, 5.
    In paper, there's 4-5 layers versions.
    """
    n_filters: Sequence[int]  # 2 length. Last one is the number of output channels.
    scale: int

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, **kwargs) -> jnp.ndarray:
        shape = inputs.shape
        inputs = jax.image.resize(
            inputs, (shape[0], shape[1] * self.scale, shape[2] * self.scale, shape[3]), method="bicubic"
        )
        x = nn.Conv(features=self.n_filters[0], kernel_size=(9, 9), padding="SAME")(inputs)
        x = nn.relu(x)
        x = nn.Conv(features=self.n_filters[1], kernel_size=(1, 1), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(shape[-1], kernel_size=(5, 5), padding="SAME")(x)
        return x
