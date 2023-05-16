import jax
import jax.numpy as jnp
import tensorflow as tf
import numpy as np
from absl.testing import parameterized, absltest

from functools import partial
from itertools import product
import pickle
from tqdm import tqdm

from flaxsr.losses.utils import check_vgg_params_exists, load_vgg19_params
from flaxsr.losses.perceptual_losses import _get_feats_from_vgg19, vgg_loss

jax.config.parse_flags_with_absl()


"""
1. Check all weight/bias is same
2. Check output feature map is same
"""


check_vgg_params_exists()
search_space = {
    'feats_from': [(16,), (0,), (2, 5, 6), (6, 8, 14)],
    'before_act': [True, False],
    'mode': ['mean', 'sum', None]
}
search_space_list = list(product(*search_space.values()))
search_space = [dict(zip(search_space.keys(), v)) for v in search_space_list]


class TestVGGLoss(parameterized.TestCase):
    def test_1_weights(self):
        tf.keras.backend.clear_session()
        model_tf = tf.keras.applications.VGG19(
            include_top=False, weights="imagenet", input_shape=(None, None, 3)
        )
        weights_jax = load_vgg19_params()

        i = 0
        for layer in tqdm(model_tf.layers):
            if "conv" in layer.name:
                tf_w = layer.get_weights()[0]
                tf_b = layer.get_weights()[1]

                jax_w = weights_jax[i][0]
                jax_b = weights_jax[i][1]

                self.assertTrue(np.allclose(np.asarray(tf_w), np.asarray(jax_w)))
                self.assertTrue(np.allclose(np.asarray(tf_b), np.asarray(jax_b)))
                i += 1

    def test_2_output(self):
        tf.keras.backend.clear_session()
        model_tf = tf.keras.applications.VGG19(
            include_top=False, weights="imagenet", input_shape=(None, None, 3)
        )
        weights_jax = load_vgg19_params()

        for _ in tqdm(range(41)):
            inp = tf.random.normal((16, 256, 256, 3)) * 3.
            tf_out = model_tf(inp)
            jax_out = _get_feats_from_vgg19(jnp.asarray(inp), weights_jax, False)
            np.allclose(np.asarray(tf_out), np.asarray(jax_out[-1]), atol=1e-5, rtol=1e-5)

    @parameterized.parameters(search_space)
    def test_3_loss(self, feats_from, before_act, mode):
        weights_jax = load_vgg19_params()

        inp = np.random.normal(size=(16, 256, 256, 3)) * 3.
        inp = jnp.asarray(inp)

        out = vgg_loss(inp, inp, weights_jax, feats_from, before_act, mode)
        np.allclose(out, 0.0, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    absltest.main()
