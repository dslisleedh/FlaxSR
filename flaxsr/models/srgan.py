import jax
import jax.lax as lax
import jax.numpy as jnp

import flax
import flax.linen as nn

import numpy as np
import einops

from functools import partial
from typing import Sequence, Literal, Optional

from flaxsr.layers import PixelShuffle
from flaxsr._utils import register


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


@register('models', 'srresnet')
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


class DiscriminatorBlock(nn.Module):
    n_filters: int
    kernel_size: int
    strides: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool) -> jnp.ndarray:
        x = nn.Conv(self.n_filters, (self.kernel_size, self.kernel_size), strides=self.strides)(x)
        x = nn.BatchNorm()(x, use_running_average=not training)
        x = nn.activation.leaky_relu(x, negative_slope=0.2)
        return x


class Discriminator(nn.Module):
    n_filters: int = 64

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool) -> jnp.ndarray:
        x = nn.Conv(self.n_filters, (3, 3))(x)
        x = nn.activation.leaky_relu(x, negative_slope=0.2)

        x = DiscriminatorBlock(self.n_filters, 3, 2)(x, training=training)
        x = DiscriminatorBlock(self.n_filters * 2, 3, 1)(x, training=training)
        x = DiscriminatorBlock(self.n_filters * 2, 3, 2)(x, training=training)
        x = DiscriminatorBlock(self.n_filters * 4, 3, 1)(x, training=training)
        x = DiscriminatorBlock(self.n_filters * 4, 3, 2)(x, training=training)
        x = DiscriminatorBlock(self.n_filters * 8, 3, 1)(x, training=training)
        x = DiscriminatorBlock(self.n_filters * 8, 3, 2)(x, training=training)

        x = einops.rearrange(x, '... h w c -> ... (h w c)')
        x = nn.Dense(1024)(x)
        x = nn.activation.leaky_relu(x, negative_slope=0.2)
        x = nn.Dense(1)(x)
        return x


@register('models', 'srgan')
class SRGAN(nn.Module):
    generator_spec: dict
    discriminator_spec: dict

    def setup(self) -> None:
        self.generator = SRResNet(**self.generator_spec)
        self.discriminator = Discriminator(**self.discriminator_spec)

    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        return self.generator(x, training=training)

    def discriminate(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        return self.discriminator(x, training=training)
