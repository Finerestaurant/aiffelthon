import flax.linen as nn
import jax 
from Conv2d_model import Conv2d_VAE
import jax.numpy as jnp


print(nn.tabulate(Conv2d_VAE(), jax.random.PRNGKey(42))(jnp.ones((128, 48, 1876)), jax.random.PRNGKey(42)))