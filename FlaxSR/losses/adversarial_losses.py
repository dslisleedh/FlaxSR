import jax
import jax.lax as lax
import jax.numpy as jnp

import flax
import flax.linen as nn

import numpy as np
import einops

from functools import partial
from typing import Sequence, Literal

from FlaxSR.losses.utils import reduce_fn


"""
TODO: Implement VanillaGAN loss, LSGAN loss, relativistic GAN loss(from ESRGAN)
"""


def gan_loss(true_logit: jnp.ndarray, fake_logit):
    pass


def lsgan_loss():
    pass


def relativistic_gan_loss():
    pass
