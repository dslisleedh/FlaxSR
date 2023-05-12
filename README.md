


# FlaxSR

Super Resolution models with Jax/Flax

## HOW TO USE

### Install
```shell
pip install flaxsr
```

### Usage

```python
from flaxsr.models import VDSR
import jax
import jax.numpy as jnp

inputs = jnp.ones((16, 256, 256, 3))
key = jax.random.PRNGKey(42)
model = VDSR(n_filters=64, n_blocks=20, scale=4)
params = model.init(key, inputs)
outputs = model.apply(params, inputs)
print(outputs.shape)
```

## Models implemented
 - SRCNN
 - FSRCNN
 - ESPCN
 - VDSR
 - EDSR, MDSR,
 - NCNet
 - SRResNet(SRGAN will be implemented in future)
 - NAFSSR

## Feats will be added in future

 - More models
 - Pre-trained parameters
 - Training states(includes Generative-sr models)
