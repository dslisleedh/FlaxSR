import jax
import jax.lax as lax
import jax.numpy as jnp

import flax
import flax.linen as nn
from flax.training import train_state

import numpy as np
import einops

from functools import partial
from typing import Sequence, Literal

import optax

from absl.testing import parameterized, absltest

import flaxsr


class TestTrainStep(absltest.TestCase):
    def test_0_discriminative_train_step(self):
        model_kwargs = {
            'n_filters': 64,
            'n_blocks': 8,
            'scale': 4
        }
        model = flaxsr.get('models', 'vdsr', **model_kwargs)
        losses = flaxsr.losses.LossWrapper(
            ['l1', 'l2'],
            [.9, .1]
        )
        params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 32, 32, 3), dtype=jnp.float32))
        tx = optax.adam(1e-3)

        state = flaxsr.training.TrainState.create(
            apply_fn=model.apply, params=params, tx=tx, losses=losses,
        )

        hr = jnp.ones((1, 32, 32, 3), dtype=jnp.float32)
        lr = jnp.ones((1, 8, 8, 3), dtype=jnp.float32)

        batch = (lr, hr)

        state_new, loss = flaxsr.training.discriminative_train_step(state, batch)

        self.assertEqual(state_new.step, 1)
        np.not_equal(state_new.params['params']['Conv_0']['kernel'], state.params['params']['Conv_0']['kernel'])


if __name__ == '__main__':
    absltest.main()
    