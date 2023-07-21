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


class RB(nn.Module):
    n_filters: int
    residual_scale: float = .2

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        res = nn.Conv(self.n_filters, (3, 3), padding='SAME')(x)
        res = nn.relu(res)
        res = nn.Conv(self.n_filters, (3, 3), padding='SAME')(res)
        return x + res * self.residual_scale


class ResidualDenseBlock(nn.Module):
    n_filters: int = 64
    gc: int = 32
    use_bias: bool = True
    residual_scale: float = .2

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x1 = nn.Conv(self.n_filters, (3, 3), padding='SAME', use_bias=self.use_bias)(x)
        x1 = nn.leaky_relu(x1, negative_slope=.2)
        x2 = nn.Conv(self.n_filters, (3, 3), padding='SAME', use_bias=self.use_bais)(
            jnp.concatenate([x, x1], axis=-1))
        x2 = nn.leaky_relu(x2, negative_slope=.2)
        x3 = nn.Conv(self.n_filters, (3, 3), padding='SAME', use_bias=self.use_bias)(
            jnp.concatenate([x, x1, x2], axis=-1))
        x3 = nn.leaky_relu(x3, negative_slope=.2)
        x4 = nn.Conv(self.n_filters, (3, 3), padding='SAME', use_bias=self.use_bias)(
            jnp.concatenate([x, x1, x2, x3], axis=-1))
        x4 = nn.leaky_relu(x4, negative_slope=.2)
        x5 = nn.Conv(self.n_filters, (3, 3), padding='SAME', use_bias=self.use_bias)(
            jnp.concatenate([x, x1, x2, x3, x4], axis=-1))
        return x + x5 * self.residual_scale


class RRDB(nn.Module):
    n_filters: int = 64
    gc: int = 32
    use_bias: bool = True
    residual_scale: float = .2

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        res = ResidualDenseBlock(self.n_filters, self.gc, self.use_bias, self.residual_scale)(x)
        res = ResidualDenseBlock(self.n_filters, self.gc, self.use_bias, self.residual_scale)(res)
        res = ResidualDenseBlock(self.n_filters, self.gc, self.use_bias, self.residual_scale)(res)
        return x + res * self.residual_scale


@register('models', 'rrdbnet')
class RRDBNet(nn.Module):
    n_filters: int = 64
    gd: int = 32
    n_blocks: int = 23
    use_bias: bool = True
    residual_scale: float = .2
    use_rb: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        # Feat Projection
        feat = nn.Conv(self.n_filters, (3, 3), padding='SAME', use_bias=self.use_bias)(x)

        # Feat Refinement
        feat_res = feat
        for _ in range(self.n_blocks):
            feat_res = RB(self.n_filters, self.residual_scale)(feat_res) if self.use_rb\
                else RRDB(self.n_filters, self.gd, self.use_bias, self.residual_scale)(feat_res)
        feat_res = nn.Conv(self.n_filters, (3, 3), padding='SAME', use_bias=self.use_bias)(feat_res)
        feat = feat_res + feat

        # Upsampling
        shape = feat.shape
        feat = jax.image.resize(feat, (shape[0], shape[1] * 2, shape[2] * 2, shape[3]), method='nearest')
        feat = nn.Conv(self.n_filters, (3, 3), padding='SAME', use_bias=self.use_bias)(feat)
        feat = nn.leaky_relu(feat, negative_slope=.2)
        feat = jax.image.resize(feat, (shape[0], shape[1] * 4, shape[2] * 4, shape[3]), method='nearest')
        feat = nn.Conv(self.n_filters, (3, 3), padding='SAME', use_bias=self.use_bias)(feat)
        feat = nn.leaky_relu(feat, negative_slope=.2)

        # Reconstruction
        feat = nn.Conv(3, (3, 3), padding='SAME', use_bias=self.use_bias)(feat)
        feat = nn.leaky_relu(feat, negative_slope=.2)
        out = nn.Conv(3, (3, 3), padding='SAME', use_bias=self.use_bias)(feat)
        return out


# There's no information about discriminator's architecture in the paper or the official repo
# So I referenced https://github.com/Lornatang/ESRGAN-PyTorch/blob/main/model.py


class Discriminator(nn.Module):
    n_filters: int = 64

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        # Input = 128 x 128 x 3
        x = nn.Conv(self.n_filters, (3, 3), padding='SAME')(x)
        x = nn.leaky_relu(x, negative_slope=.2)

        x = nn.Conv(self.n_filters, (4, 4), strides=(2, 2), padding='SAME', use_bias=False)(x)  # 128 -> 64
        x = nn.BatchNorm()(x, use_running_average=not training)
        x = nn.leaky_relu(x, negative_slope=.2)
        x = nn.Conv(self.n_filters * 2, (3, 3), padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm()(x, use_running_average=not training)
        x = nn.leaky_relu(x, negative_slope=.2)

        x = nn.Conv(self.n_filters * 2, (4, 4), strides=(2, 2), padding='SAME', use_bias=False)(x)  # 64 -> 32
        x = nn.BatchNorm()(x, use_running_average=not training)
        x = nn.leaky_relu(x, negative_slope=.2)
        x = nn.Conv(self.n_filters * 4, (3, 3), padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm()(x, use_running_average=not training)
        x = nn.leaky_relu(x, negative_slope=.2)

        x = nn.Conv(self.n_filters * 4, (4, 4), strides=(2, 2), padding='SAME', use_bias=False)(x)  # 32 -> 16
        x = nn.BatchNorm()(x, use_running_average=not training)
        x = nn.leaky_relu(x, negative_slope=.2)
        x = nn.Conv(self.n_filters * 8, (3, 3), padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm()(x, use_running_average=not training)
        x = nn.leaky_relu(x, negative_slope=.2)

        x = nn.Conv(self.n_filters * 8, (4, 4), strides=(2, 2), padding='SAME', use_bias=False)(x)  # 16 -> 8
        x = nn.BatchNorm()(x, use_running_average=not training)
        x = nn.leaky_relu(x, negative_slope=.2)
        x = nn.Conv(self.n_filters * 8, (3, 3), padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm()(x, use_running_average=not training)
        x = nn.leaky_relu(x, negative_slope=.2)

        x = nn.Conv(self.n_filters * 8, (4, 4), strides=(2, 2), padding='SAME', use_bias=False)(x)  # 8 -> 4
        x = nn.BatchNorm()(x, use_running_average=not training)
        x = nn.leaky_relu(x, negative_slope=.2)

        # Classification
        x = einops.rearrange(x, 'b h w c -> b (h w c)')
        x = nn.Dense(100)(x)
        x = nn.leaky_relu(x, negative_slope=.2)
        x = nn.Dense(1)(x)
        return x


@register('models', 'esrgan')
class ESRGAN(nn.Module):
    generator_spec: dict
    discriminator_spec: dict

    def setup(self):
        self.generator = RRDBNet(**self.generator_spec)
        self.discriminator = Discriminator(**self.discriminator_spec)

    def __call__(
            self, x: jnp.ndarray, hr: jnp.ndarray, training: bool = False, discriminate: bool = False
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | jnp.ndarray:
        sr = self.generator(x, training=training)
        sr = jnp.clip(sr, 0, 1)

        if discriminate:
            sr_score = self.discriminator(sr, training=training)
            hr_score = self.discriminator(hr, training=training)
            return sr, sr_score, hr_score
        else:
            return sr
