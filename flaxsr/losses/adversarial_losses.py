import jax
import jax.lax as lax
import jax.numpy as jnp

import flax
import flax.linen as nn

import numpy as np
import einops

from functools import partial
from typing import Sequence, Literal

from flaxsr.losses.utils import reduce_fn
from flaxsr._utils import register


"""
TODO: Implement VanillaGAN loss, LSGAN loss, relativistic GAN loss(from ESRGAN)
"""


@register('losses', 'gan')
def gan_loss(true_logit: jnp.ndarray, fake_logit):
    pass


@register('losses', 'lsgan')
def lsgan_loss():
    pass


@register('losses', 'relativistic')
def relativistic_gan_loss():
    pass
