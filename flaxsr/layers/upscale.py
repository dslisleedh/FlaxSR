import jax
import jax.lax as lax
import jax.numpy as jnp

import flax
import flax.linen as nn
from flaxsr._utils import register

import numpy as np
import einops

from functools import partial
from typing import Sequence, Literal, Optional


def _pixel_shuffle(inputs: jnp.ndarray, scale: int) -> jnp.ndarray:
    output = einops.rearrange(inputs, 'b (h s1) (w s2) c -> b h w (s1 s2 c)', s1=scale, s2=scale)
    return output


def _nearest_conv_init(input_c: int, out_c: int, scale: int) -> jnp.ndarray:
    kernel = jnp.transpose(
        jnp.reshape(
            jnp.repeat(
                jnp.eye(input_c), (out_c * (scale ** 2)) // input_c, axis=0
            ), (1, 1, -1, input_c)
        ), (0, 1, 3, 2)
    )
    return kernel


@register('layers', 'pixelshuffle')
class PixelShuffle(nn.Module):
    scale: int

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        return _pixel_shuffle(inputs, self.scale)


@register('layers', 'nearestconv')
class NearestConv(nn.Module):
    scale: int
    out_c: Optional[int] = None
    return_upscaled: bool = True

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        input_c = inputs.shape[-1]
        out_c = self.out_c or input_c

        nc_kernel = self.variable(
            'nearest_conv', 'kernel', lambda: _nearest_conv_init(input_c, out_c, self.scale)
        )

        outputs = lax.conv_general_dilated(
            inputs, nc_kernel, (1, 1), 'VALID',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        )
        if self.return_upscaled:
            return _pixel_shuffle(outputs, self.scale)
        else:
            return outputs
