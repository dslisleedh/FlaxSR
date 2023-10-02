import jax
import jax.lax as lax
import jax.numpy as jnp

import flax
import flax.linen as nn

import numpy as np
import einops

from functools import partial
from typing import Sequence, Literal, Optional
from enum import Enum

from flaxsr.losses.utils import reduce_fn, Reduce, Loss
from flaxsr._utils import register


class Kernel(Enum):
    SOBEL = 'sobel'
    LAPLACIAN = 'laplacian'


@partial(jax.jit, static_argnums=(1,))
def total_variation_loss(x: jnp.ndarray, reduce: str | Reduce = 'sum') -> jnp.ndarray:
    """
    Not include eps in the loss function. You should multipy eps after the loss function.
    """
    if reduce == 'none' or reduce == Reduce.NONE:
        raise ValueError('reduce must be sum or mean for tv loss, but got none')
    diff_h = reduce_fn(
        jnp.square(x[:, :-1, :, :] - x[:, 1:, :, :]), reduce=reduce
    )
    diff_w = reduce_fn(
        jnp.square(x[:, :, :-1, :] - x[:, :, 1:, :]), reduce=reduce
    )

    loss = diff_h + diff_w
    return loss


@register('losses', 'tv')
class TotalVariationLoss(Loss):
    def __init__(self, reduce: str | Reduce = 'sum', weight: float = 1e-8):
        super().__init__(reduce, weight)

    def __call__(self, x: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        return total_variation_loss(x, self.reduce) * self.weight


@partial(jax.jit, static_argnums=(2,))
def frequency_reconstruction_loss(sr: jnp.ndarray, hr: jnp.ndarray, reduce: str | Reduce = 'mean') -> jnp.ndarray:
    hr_frequency = jnp.fft.rfft2(hr, axes=(1, 2))
    sr_frequency = jnp.fft.rfft2(sr, axes=(1, 2))

    loss = jnp.abs(hr_frequency - sr_frequency)
    return reduce_fn(loss, reduce)


@register('losses', 'freq_recon')
class FrequencyReconstructionLoss(Loss):
    def __init__(self, reduce: str | Reduce = 'mean', weight: float = 5e-2):
        super().__init__(reduce, weight)

    def __call__(self, sr: jnp.ndarray, hr: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        return frequency_reconstruction_loss(hr, sr, self.reduce) * self.weight


def _magnitude(x: jnp.ndarray, y: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    return jnp.sqrt(jnp.square(x) + jnp.square(y) + eps)


def _load_sobel_kernels(kernel_size: int) -> Sequence[jnp.ndarray]:
    if kernel_size == 3:
        kernel_init = jnp.array([[-1., 0., 1.],
                                 [-2., 0., 2.],
                                 [-1., 0., 1.]], dtype=jnp.float32)
    elif kernel_size == 5:
        kernel_init = jnp.array([[-1., -2., 0., 2., 1.],
                                 [-4., -8., 0., 8., 4.],
                                 [-6., -12., 0., 12., 6.],
                                 [-4., -8., 0., 8., 4.],
                                 [-1., -2., 0., 2., 1.]], dtype=jnp.float32)
    else:
        raise ValueError(f'kernel_size must be 3 or 5, but got {kernel_size}')

    kernel_x = kernel_init[:, :, jnp.newaxis, jnp.newaxis]
    kernel_y = kernel_init[:, :, jnp.newaxis, jnp.newaxis].transpose((1, 0, 2, 3))

    return kernel_x, kernel_y


@partial(jax.jit, static_argnums=(2,))
def filter_sobel(sr: jnp.ndarray, hr: jnp.ndarray, kernel_size: int) -> Sequence[jnp.ndarray]:
    c = sr.shape[-1]
    kernel_x, kernel_y = _load_sobel_kernels(kernel_size)
    kernel_x = jnp.repeat(kernel_x, c, axis=-1)
    kernel_y = jnp.repeat(kernel_y, c, axis=-1)

    sr_x = lax.conv_general_dilated(
        sr, kernel_x, window_strides=(1, 1), padding='SAME',
        feature_group_count=c, dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    )
    sr_y = lax.conv_general_dilated(
        sr, kernel_y, window_strides=(1, 1), padding='SAME',
        feature_group_count=c, dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    )
    hr_x = lax.conv_general_dilated(
        hr, kernel_x, window_strides=(1, 1), padding='SAME',
        feature_group_count=c, dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    )
    hr_y = lax.conv_general_dilated(
        hr, kernel_y, window_strides=(1, 1), padding='SAME',
        feature_group_count=c, dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    )

    sr_filtered = _magnitude(sr_x, sr_y)
    hr_filtered = _magnitude(hr_x, hr_y)

    return sr_filtered, hr_filtered


def _load_laplacian_kernels(kernel_size: int) -> jnp.ndarray:
    if kernel_size == 3:
        kernel_init = jnp.array([[0., 1., 0.],
                                 [1., -4., 1.],
                                 [0., 1., 0.]])
    elif kernel_size == 5:
        kernel_init = jnp.array([[0., 0., 1., 0., 0.],
                                 [0., 1., 2., 1., 0.],
                                 [1., 2., -16., 2., 1.],
                                 [0., 1., 2., 1., 0.],
                                 [0., 0., 1., 0., 0.]])
    else:
        raise ValueError(f'kernel_size must be 3 or 5, but got {kernel_size}')

    kernel = kernel_init[:, :, jnp.newaxis, jnp.newaxis]
    return kernel


@partial(jax.jit, static_argnums=(2,))
def filter_laplacian(sr: jnp.ndarray, hr: jnp.ndarray, kernel_size: int) -> Sequence[jnp.ndarray]:
    c = sr.shape[-1]
    kernel = _load_laplacian_kernels(kernel_size)
    kernel = jnp.repeat(kernel, c, axis=-1)

    sr_filtered = lax.conv_general_dilated(
        sr, kernel, window_strides=(1, 1), padding='SAME',
        feature_group_count=c, dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    )
    hr_filtered = lax.conv_general_dilated(
        hr, kernel, window_strides=(1, 1), padding='SAME',
        feature_group_count=c, dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    )

    return sr_filtered, hr_filtered


@partial(jax.jit, static_argnums=(2, 3, 4))
def edge_loss(
        sr: jnp.ndarray, hr: jnp.ndarray, kernel_size: int, kernel_type: str | Kernel, reduce: str | Reduce = 'mean'
) -> jnp.ndarray:
    if isinstance(kernel_type, str):
        kernel_type = Kernel(kernel_type)

    if kernel_type == Kernel.SOBEL:
        sr_filtered, hr_filtered = filter_sobel(sr, hr, kernel_size)
    elif kernel_type == Kernel.LAPLACIAN:
        sr_filtered, hr_filtered = filter_laplacian(sr, hr, kernel_size)
    else:
        raise ValueError(f'kernel_type must be sobel or laplacian, but got {kernel_type}')

    loss = jnp.abs(sr_filtered - hr_filtered)
    return reduce_fn(loss, reduce)


@register('losses', 'edge')
class EdgeLoss(Loss):
    def __init__(
            self, kernel_size: int = 3, kernel_type: str | Kernel = 'sobel', reduce: str | Reduce = 'mean',
            weight: float = 1e-2
    ):
        super().__init__(reduce, weight)
        self.kernel_size = kernel_size
        self.kernel_type = kernel_type

    def __call__(self, sr: jnp.ndarray, hr: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        return edge_loss(sr, hr, self.kernel_size, self.kernel_type, self.reduce) * self.weight
