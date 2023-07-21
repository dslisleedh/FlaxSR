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

from omegaconf import OmegaConf

from absl.testing import parameterized, absltest

import flaxsr


class TestTrainState(absltest.TestCase):
    def test_0_from_scratch_discriminative(self):
        model_kwargs = {
            'n_filters': 64,
            'n_blocks': 8,
            'scale': 4
        }
        model = flaxsr.get('models', 'vdsr', **model_kwargs)
        losses = [
            flaxsr.losses.L1Loss(reduce='sum'),
            flaxsr.get('losses', 'vgg', feats_from=(6, 8, 14,), before_act=False, reduce='mean')
        ]
        weights = (.1, 1.,)
        loss_wrapper = flaxsr.losses.LossWrapper(losses, weights)
        params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 32, 32, 3), dtype=jnp.float32))
        tx = optax.adam(1e-3)

        state = flaxsr.training.discriminative.TrainState.create(
            apply_fn=model.apply, params=params, tx=tx, losses=loss_wrapper,
        )

        hr = jnp.ones((1, 32, 32, 3), dtype=jnp.float32)
        lr = jnp.ones((1, 8, 8, 3), dtype=jnp.float32)

        def loss_fn(p):
            sr = state.apply_fn(state.params, lr)
            return state.losses(hr, sr)

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grad = grad_fn(state.params)
        state_new = state.apply_gradients(grads=grad)

        self.assertEqual(state_new.step, 1)
        np.not_equal(state_new.params['params']['Conv_0']['kernel'], state.params['params']['Conv_0']['kernel'])

    def test_1_config_discriminative(self):
        config_paths = [
            './configs/srcnn/srcnn_x4.yaml',
            './configs/edsr/edsr_x4.yaml',
            './configs/vdsr/vdsr_x4.yaml',
        ]

        for config_path in config_paths:
            config = OmegaConf.load(config_path)
            init_key = jax.random.PRNGKey(0)
            state = flaxsr.training.discriminative.create_train_state(config, init_key)

            lr = jnp.ones((1, 8, 8, 3), dtype=jnp.float32)
            hr = jnp.ones((1, 32, 32, 3), dtype=jnp.float32)

            def loss_fn(p):
                sr = state.apply_fn(state.params, lr)
                return state.losses(hr, sr)

            grad_fn = jax.value_and_grad(loss_fn)
            loss, grad = grad_fn(state.params)
            state_new = state.apply_gradients(grads=grad)

            self.assertEqual(state_new.step, 1)
            np.not_equal(state_new.params['params']['Conv_0']['kernel'], state.params['params']['Conv_0']['kernel'])

    def test_3_config_generative(self):
        config_paths = [
            './configs/esrgan/esrgan_rrdb.yaml'
        ]

        for config_path in config_paths:
            config = OmegaConf.load(config_path)
            init_key = jax.random.PRNGKey(0)
            state = flaxsr.training.generative.create_train_state(config, init_key)

            lr = jnp.ones((1, 8, 8, 3), dtype=jnp.float32)
            hr = jnp.ones((1, 32, 32, 3), dtype=jnp.float32)

            def disc_loss_fn(p):
                sr = state.apply_fn(state.params, lr)



if __name__ == '__main__':
    absltest.main()
