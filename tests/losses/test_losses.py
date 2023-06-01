import jax
import jax.numpy as jnp
import tensorflow as tf
import numpy as np
from absl.testing import parameterized, absltest

from functools import partial
from itertools import product
import pickle
from tqdm import tqdm

import flaxsr


jax.config.parse_flags_with_absl()


class TestLosses(absltest.TestCase):
    def test_0_loss_none_includes(self):
        names = ['l1', 'l2', 'charbonnier']
        weights = [1.0, 2.0, 3.0]

        reduces = 'none'
        flaxsr.losses.LossWrapper(names, weights, reduces)

        reduces = ['none', 'none', 'none']
        flaxsr.losses.LossWrapper(names, weights, reduces)

        reduces = ['sum', 'sum', 'sum']
        flaxsr.losses.LossWrapper(names, weights, reduces)

        with self.assertRaises(AssertionError):
            reduces = ['none', 'mean', 'sum']
            flaxsr.losses.LossWrapper(names, weights, reduces)

    def test_1_loss_calculation(self):
        names = ['l1', 'l2', 'charbonnier']
        weights = [1.0, 2.0, 3.0]
        reduces = 'mean'

        loss_wrapper = flaxsr.losses.LossWrapper(names, weights, reduces)
        hr = jnp.ones((1, 3, 3, 3))
        sr = jnp.zeros((1, 3, 3, 3))

        rng = jax.random.PRNGKey(0)
        mask = jax.random.bernoulli(rng, 0.5, hr.shape)

        loss = loss_wrapper(hr, sr, mask)

        loss_from_get = 0.
        for name, weight in zip(names, weights):
            loss_from_get += weight * flaxsr.get('losses', name)(hr * mask, sr * mask)

        self.assertEqual(loss, loss_from_get)


if __name__ == '__main__':
    absltest.main()
