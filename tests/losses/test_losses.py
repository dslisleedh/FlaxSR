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


search_space = {
    'reduce': ['none', 'mean', 'sum'],
    'same_input': [True, False],
    'jit': [True, False],
}
search_space_ = list(product(*search_space.values()))
pixel_wise_loss_search_space = [dict(zip(search_space.keys(), v)) for v in search_space_]

flaxsr.losses.utils.check_vgg_params_exists()
search_space = {
    'feats_from': [(16,), (0,), (2, 5, 6), (6, 8, 14)],
    'before_act': [True, False],
    'reduce': ['mean', 'sum', 'sum'],
    'jit': [True, False],
    'same_input': [True, False],
}
search_space_ = list(product(*search_space.values()))
perceptual_loss_search_space = [dict(zip(search_space.keys(), v)) for v in search_space_]

search_space = {
    'reduce': ['none', 'mean', 'sum'],
    'from_logits': [True, False],
    'same_input': [True, False],
    'jit': [True, False],
}
search_space_ = list(product(*search_space.values()))
adversarial_loss_search_space = [dict(zip(search_space.keys(), v)) for v in search_space_]

search_space = {
    'reduce': ['none', 'mean', 'sum'],
    'jit': [True, False],
}
search_space_ = list(product(*search_space.values()))
regularization_loss_search_space = [dict(zip(search_space.keys(), v)) for v in search_space_]


class TestPixelWiseLosses(parameterized.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.hr_ones = jnp.ones((1, 32, 32, 3))
        self.hr_zeros = jnp.zeros((1, 32, 32, 3))
        self.sr_ones = jnp.ones((1, 32, 32, 3))
        self.sr_zeros = jnp.zeros((1, 32, 32, 3))

        self.mask = jnp.ones((1, 32, 32, 3))

    @parameterized.parameters(*pixel_wise_loss_search_space)
    def test_0_l1(self, reduce, same_input, jit):
        Loss = flaxsr.get('losses', 'l1', reduce)
        if jit:
            Loss = jax.jit(Loss)

        if same_input:
            hr = self.hr_ones
            sr = self.sr_ones
        else:
            hr = self.hr_ones
            sr = self.sr_zeros

        loss = Loss(sr, hr)

        if reduce == 'none':
            self.assertEqual(loss.shape, (1, 32, 32, 3))
        else:
            self.assertEqual(loss.shape, ())

        if same_input:
            np.equal(np.zeros(()), np.sum(loss))
        else:
            np.not_equal(np.zeros(()), np.sum(loss))

    @parameterized.parameters(*pixel_wise_loss_search_space)
    def test_1_l2(self, reduce, same_input, jit):
        Loss = flaxsr.get('losses', 'l2', reduce)
        if jit:
            Loss = jax.jit(Loss)

        if same_input:
            hr = self.hr_ones
            sr = self.sr_ones
        else:
            hr = self.hr_ones
            sr = self.sr_zeros

        loss = Loss(sr, hr)

        if reduce == 'none':
            self.assertEqual(loss.shape, (1, 32, 32, 3))
        else:
            self.assertEqual(loss.shape, ())

        if same_input:
            np.equal(np.zeros(()), np.sum(loss))
        else:
            np.not_equal(np.zeros(()), np.sum(loss))

    @parameterized.parameters(*pixel_wise_loss_search_space)
    def test_2_charbonnier(self, reduce, same_input, jit):
        Loss = flaxsr.get('losses', 'charbonnier', .001, reduce)  # eps
        if jit:
            Loss = jax.jit(Loss)

        if same_input:
            hr = self.hr_ones
            sr = self.sr_ones
        else:
            hr = self.hr_ones
            sr = self.sr_zeros

        loss = Loss(sr, hr)

        if reduce == 'none':
            self.assertEqual(loss.shape, (1, 32, 32, 3))
        else:
            self.assertEqual(loss.shape, ())

        if same_input:
            np.equal(np.zeros(()), np.sum(loss))
        else:
            np.not_equal(np.zeros(()), np.sum(loss))


class TestPerceptualLoss(parameterized.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.hr_ones = jnp.ones((1, 32, 32, 3))
        self.hr_zeros = jnp.zeros((1, 32, 32, 3))
        self.sr_ones = jnp.ones((1, 32, 32, 3))
        self.sr_zeros = jnp.zeros((1, 32, 32, 3))

        self.mask = jnp.ones((1, 32, 32, 3))

    @parameterized.parameters(*perceptual_loss_search_space)
    def test_0_vgg(self, feats_from, before_act, reduce, same_input, jit):
        Loss = flaxsr.get('losses', 'vgg', feats_from, before_act, reduce)
        if jit:
            Loss = jax.jit(Loss)

        if same_input:
            hr = self.hr_ones
            sr = self.sr_ones
        else:
            hr = self.hr_ones
            sr = self.sr_zeros

        loss = Loss(sr, hr)

        if reduce == 'none':
            self.assertEqual(loss.shape, (1, 32, 32, 3))
        else:
            self.assertEqual(loss.shape, ())

        if same_input:
            np.equal(np.zeros(()), np.sum(loss))
        else:
            np.not_equal(np.zeros(()), np.sum(loss))


class TestAdversarialLoss(parameterized.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hr_ones = jnp.ones((16, 1))
        self.hr_zeros = jnp.zeros((16, 1))
        self.sr_ones = jnp.ones((16, 1))
        self.sr_zeros = jnp.zeros((16, 1))

    @parameterized.parameters(*adversarial_loss_search_space)
    def test_0_minmax(self, reduce, from_logits, same_input, jit):
        discriminator_loss = flaxsr.get('losses', 'minmax_discriminator', from_logits, reduce)
        generator_loss = flaxsr.get('losses', 'minmax_generator', from_logits, reduce)

        if jit:
            discriminator_loss = jax.jit(discriminator_loss)
            generator_loss = jax.jit(generator_loss)

        if same_input:
            hr = self.hr_ones
            sr = self.sr_ones
        else:
            hr = self.hr_ones
            sr = self.sr_zeros

        discriminator_loss = discriminator_loss(sr, hr)
        generator_loss = generator_loss(sr)

        if reduce == 'none':
            self.assertEqual(discriminator_loss.shape, (16, 1))
            self.assertEqual(generator_loss.shape, (16, 1))
        else:
            self.assertEqual(discriminator_loss.shape, ())
            self.assertEqual(generator_loss.shape, ())

        if same_input:
            np.equal(np.zeros(()), np.sum(discriminator_loss))
        else:
            np.not_equal(np.zeros(()), np.sum(discriminator_loss))

    @parameterized.parameters(*adversarial_loss_search_space)
    def test_1_least_square(self, reduce, from_logits, same_input, jit):
        discriminator_loss = flaxsr.get('losses', 'least_square_discriminator', from_logits, reduce)
        generator_loss = flaxsr.get('losses', 'least_square_generator', from_logits, reduce)

        if jit:
            discriminator_loss = jax.jit(discriminator_loss)
            generator_loss = jax.jit(generator_loss)

        if same_input:
            hr = self.hr_ones
            sr = self.sr_ones
        else:
            hr = self.hr_ones
            sr = self.sr_zeros

        discriminator_loss = discriminator_loss(sr, hr)
        generator_loss = generator_loss(sr)

        if reduce == 'none':
            self.assertEqual(discriminator_loss.shape, (16, 1))
            self.assertEqual(generator_loss.shape, (16, 1))
        else:
            self.assertEqual(discriminator_loss.shape, ())
            self.assertEqual(generator_loss.shape, ())

        if same_input:
            np.equal(np.zeros(()), np.sum(discriminator_loss))
        else:
            np.not_equal(np.zeros(()), np.sum(discriminator_loss))

    @parameterized.parameters(*adversarial_loss_search_space)
    def test_2_relativistic(self, reduce, from_logits, same_input, jit):
        # There's no from_logits for relativistic loss
        discriminator_loss = flaxsr.get('losses', 'relativistic_discriminator', reduce)
        generator_loss = flaxsr.get('losses', 'relativistic_generator', reduce)

        if jit:
            discriminator_loss = jax.jit(discriminator_loss)
            generator_loss = jax.jit(generator_loss)

        if same_input:
            hr = self.hr_ones
            sr = self.sr_ones
        else:
            hr = self.hr_ones
            sr = self.sr_zeros

        discriminator_loss = discriminator_loss(sr, hr)
        generator_loss = generator_loss(sr, hr)

        if reduce == 'none':
            self.assertEqual(discriminator_loss.shape, (16, 1))
            self.assertEqual(generator_loss.shape, (16, 1))
        else:
            self.assertEqual(discriminator_loss.shape, ())
            self.assertEqual(generator_loss.shape, ())

        if same_input:
            np.equal(np.zeros(()), np.sum(discriminator_loss))
        else:
            np.not_equal(np.zeros(()), np.sum(discriminator_loss))


class TestLossWrapper(absltest.TestCase):
    def test_0_check_none_assert(self):
        weights = (.1, 1., .5)

        losses = [
            flaxsr.get('losses', 'l1', 'none', weight=weights[0]),
            flaxsr.get('losses', 'l2', 'none', weight=weights[1]),
            flaxsr.get('losses', 'l1', 'none', weight=weights[2])
        ]
        flaxsr.losses.LossWrapper(losses)

        losses = [
            flaxsr.get('losses', 'l1', 'mean', weight=weights[0]),
            flaxsr.get('losses', 'l2', 'mean', weight=weights[1]),
            flaxsr.get('losses', 'l1', 'mean', weight=weights[2])
        ]
        flaxsr.losses.LossWrapper(losses)

        losses = [
            flaxsr.get('losses', 'l1', 'sum', weight=weights[0]),
            flaxsr.get('losses', 'l2', 'sum', weight=weights[1]),
            flaxsr.get('losses', 'l1', 'sum', weight=weights[2])
        ]
        flaxsr.losses.LossWrapper(losses)

        losses = [
            flaxsr.get('losses', 'l1', 'sum', weight=weights[0]),
            flaxsr.get('losses', 'l2', 'sum', weight=weights[1]),
            flaxsr.get('losses', 'l1', 'mean', weight=weights[2])
        ]
        flaxsr.losses.LossWrapper(losses)

        with self.assertRaises(AssertionError):
            losses = [
                flaxsr.get('losses', 'l1', 'sum', weight=weights[0]),
                flaxsr.get('losses', 'l2', 'sum', weight=weights[1]),
                flaxsr.get('losses', 'l1', 'none', weight=weights[2])
            ]
            flaxsr.losses.LossWrapper(losses)

    def test_discriminative_loss_calculation(self):
        weights = (.1, 1.,)
        losses = [
            flaxsr.get('losses', 'l1', 'mean', weight=weights[0]),
            flaxsr.get('losses', 'vgg', (6, 8, 14,), False, weight=weights[1]),
        ]
        loss_wrapper = flaxsr.losses.LossWrapper(losses)

        hr = jnp.ones((16, 32, 32, 3))
        sr = jnp.zeros((16, 32, 32, 3))

        loss = loss_wrapper(sr, hr)
        self.assertEqual(loss.shape, ())
        np.not_equal(np.zeros(()), loss)


class TestRegularizationLoss(parameterized.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hr = jax.random.bernoulli(jax.random.PRNGKey(0), p=.5, shape=(16, 32, 32, 3)).astype(jnp.float32)
        self.sr = jax.random.bernoulli(jax.random.PRNGKey(0), p=.5, shape=(16, 32, 32, 3)).astype(jnp.float32)

    @parameterized.parameters(*regularization_loss_search_space)
    def test_0_total_variation_loss(self, reduce, jit):
        loss = flaxsr.get('losses', 'tv', reduce)
        if jit:
            loss = jax.jit(loss)

        if reduce == 'none':
            with self.assertRaises(ValueError):
                loss = loss(self.sr)
        else:
            loss = loss(self.sr)
            self.assertEqual(loss.shape, ())

        np.not_equal(np.zeros(()), np.sum(loss))

    @parameterized.parameters(*regularization_loss_search_space)
    def test_1_frequency_reconstruction_loss(self, reduce, jit):
        loss = flaxsr.get('losses', 'freq_recon', reduce)
        if jit:
            loss = jax.jit(loss)

        loss = loss(self.sr, self.hr)

        if reduce == 'none':
            self.assertEqual(loss.shape, (16, 32, 17, 3))
        else:
            self.assertEqual(loss.shape, ())

        np.not_equal(np.zeros(()), np.sum(loss))

    @parameterized.parameters(*regularization_loss_search_space)
    def test_2_edge_loss(self, reduce, jit):
        for kernel_size in (3, 5):
            for kernel_type in ['sobel', 'laplacian']:
                loss = flaxsr.get('losses', 'edge', reduce=reduce, kernel_size=kernel_size, kernel_type=kernel_type)

                if jit:
                    loss = jax.jit(loss)

                loss = loss(self.sr, self.hr)

                if reduce == 'none':
                    self.assertEqual(loss.shape, (16, 32, 32, 3))
                else:
                    self.assertEqual(loss.shape, ())

                np.not_equal(np.zeros(()), np.sum(loss))


if __name__ == '__main__':
    absltest.main()
