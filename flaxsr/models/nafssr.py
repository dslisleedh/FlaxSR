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


def _simple_gate(inputs: jnp.ndarray) -> jnp.ndarray:
    x1, x2 = jnp.split(inputs, inputs.shape[-1] // 2, axis=-1)
    return x1 * x2


class NAFBlock(nn.Module):
    tlsc_patch_size: Sequence[int]  # (H, W)
    dw_expansion_ratio: int = 2
    ffn_expansion_ratio: int = 2
    # Skip Dropout because NAFSSR uses DropPath instead.

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, training: bool = False, **kwargs) -> jnp.ndarray:
        shape = inputs.shape
        beta = self.param('beta', nn.initializers.zeros, (1, 1, 1, shape[-1]))
        gamma = self.param('gamma', nn.initializers.zeros, (1, 1, 1, shape[-1]))

        spatial = nn.LayerNorm()(inputs)
        spatial = nn.Conv(shape[-1] * self.dw_expansion_ratio, (1, 1), padding='VALID')(spatial)
        spatial = nn.Conv(shape[-1] * self.dw_expansion_ratio, (3, 3), padding='SAME')(spatial)
        spatial = _simple_gate(spatial)

        # Simple Attn and TLSC
        if training:
            statistic = jnp.mean(spatial, axis=(1, 2), keepdims=True)
        else:
            s = spatial.cumsum(2).cumsum(1)
            s = jnp.pad(s, [[0, 0], [1, 0], [1, 0], [0, 0]])
            kh, kw = self.tlsc_patch_size
            statistic = (
                s[:, kh:, kw:, :] + s[:, :-kh, :-kw, :] - s[:, :-kh, kw:, :] - s[:, kh:, :-kw, :]
            ) / (kh * kw)

            h, w = shape[1], shape[2]
            if (h != kh) or (w != kw):
                _, h_s, w_s, _ = statistic.shape
                h_pad, w_pad = [(h - h_s) // 2, (h - h_s + 1) // 2], [(w - w_s) // 2, (w - w_s + 1) // 2]
                statistic = jnp.pad(statistic, [[0, 0], h_pad, w_pad, [0, 0]], mode='edge')
        spatial_attn = nn.Dense(shape[-1] * self.dw_expansion_ratio // 2)(statistic)  # Why not use sigmoid here?

        spatial = spatial * spatial_attn
        spatial = nn.Conv(shape[-1], (1, 1), padding='VALID')(spatial)
        inputs = inputs + (spatial * beta)

        channel = nn.LayerNorm()(inputs)
        channel = nn.Conv(shape[-1] * self.ffn_expansion_ratio, (1, 1), padding='VALID')(channel)
        channel = _simple_gate(channel)
        channel = nn.Conv(shape[-1], (1, 1), padding='VALID')(channel)
        inputs = inputs + (channel * gamma)
        return inputs


class SCAM(nn.Module):
    attn_scale: float

    @nn.compact
    def __call__(self, feats: Sequence[jnp.ndarray]) -> Sequence[jnp.ndarray]:
        shape = feats[0].shape
        beta = self.param('beta', nn.initializers.zeros, (1, 1, 1, feats[0].shape[-1]))
        gamma = self.param('gamma', nn.initializers.zeros, (1, 1, 1, feats[0].shape[-1]))

        x_l, x_r = feats
        q_l = nn.LayerNorm()(x_l)
        q_l = nn.Dense(shape[-1])(q_l)  # B H W C
        q_r = nn.LayerNorm()(x_r)
        q_r_t = nn.Dense(shape[-1])(q_r).transpose((0, 1, 3, 2))  # B H C W

        v_l = nn.Dense(shape[-1])(x_l)
        v_r = nn.Dense(shape[-1])(x_r)

        attn = jnp.matmul(q_l, q_r_t) * self.attn_scale

        f_r2l = jnp.matmul(nn.softmax(attn, axis=-1), v_r) * beta
        f_l2r = jnp.matmul(nn.softmax(attn.transpose((0, 1, 3, 2)), axis=-1), v_l) * gamma
        return x_l + f_r2l, x_r + f_l2r


class NAFBlockSR(nn.Module):
    fusion: bool
    tlsc_patch_size: Sequence[int]
    attn_scale: float
    survival_prob: float
    dw_expansion_ratio: int = 2
    ffn_expansion_ratio: int = 2

    def setup(self):
        self.block = NAFBlock(self.tlsc_patch_size, self.dw_expansion_ratio, self.ffn_expansion_ratio)
        if self.fusion:
            self.scam = SCAM(self.attn_scale)

    def forward(self, feats: Sequence[jnp.ndarray], training: bool) -> Sequence[jnp.ndarray]:
        feats = [self.block(f, training=training) for f in feats]
        if self.fusion:
            feats = self.scam(feats)
        return feats

    def __call__(self, feats: Sequence[jnp.ndarray], training: bool = False, **kwargs) -> Sequence[jnp.ndarray]:
        if self.survival_prob == 0.:
            return feats
        elif (self.survival_prob == 1.) or not training:
            return self.forward(feats, training=training)
        else:
            rng = self.make_rng('DropPath')
            survival_state = jax.random.bernoulli(rng, self.survival_prob, ())
            if survival_state:
                feats_new = self.forward(feats, training=training)
                return [f + ((f_n - f) / self.survival_prob) for f, f_n in zip(feats, feats_new)]
            else:
                return feats


@register('models', 'nafssr')
class NAFSSR(nn.Module):
    n_filters: int
    n_blocks: int
    train_patch_size: Sequence[int]
    attn_scale: float
    drop_rate: float
    scale: int
    dw_expansion_ratio: int = 2
    ffn_expansion_ratio: int = 2
    fusion_from: int = -1
    fusion_to: int = 1000
    tlsc_ratio: float = 1.5
    out_c: Optional[int] = 3

    def setup(self):
        self.intro = nn.Conv(self.n_filters, (3, 3), padding='SAME')

        tlsc_patch_size = [int(size * self.tlsc_ratio) for size in self.train_patch_size]
        self.middles = nn.Sequential([
            NAFBlockSR(
                (i >= self.fusion_from) and (i < self.fusion_to), tlsc_patch_size,
                self.dw_expansion_ratio, self.ffn_expansion_ratio, self.attn_scale, 1. - self.drop_rate  # They use isotropic drop_path rate
            ) for i in range(self.n_blocks)
        ])
        self.end = nn.Sequential([
            nn.Conv(self.out_c * self.scale ** 2, (3, 3), padding='SAME'),
            PixelShuffle(self.scale)
        ])

    def __call__(
            self, inputs: Sequence[jnp.ndarray] | jnp.ndarray, training: bool = False, **kwargs
    ) -> Sequence[jnp.ndarray] | jnp.ndarray:
        is_ndarray = isinstance(inputs, jnp.ndarray)
        if is_ndarray:
            inputs = [inputs]
        shape = inputs[0].shape
        inputs_skip = [
            jax.image.resize(i, (shape[0], shape[1] * self.scale, shape[2] * self.scale, shape[3]), method='bilinear')
            for i in inputs
        ]  # FIXME: is list comprehension works well in jit-compiled function??
        feats = [self.intro(f, training=training) for f in inputs]
        feats = self.middles(feats, training=training)
        outputs = [self.end(f, training=training) for f in feats]
        if is_ndarray:
            return outputs[0] + inputs_skip[0]
        else:
            return [o + s for o, s in zip(outputs, inputs_skip)]
