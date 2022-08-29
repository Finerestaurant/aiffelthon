import jax
import jax.numpy as jnp
from jax import random

from flax import linen as nn
from typing import Callable, Any, Optional


class Encoder(nn.Module):
    
    latents: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(500, name='fc1')(x)
        x = nn.relu(x)
        mean_x = nn.Dense(self.latents, name='fc2_mean')(x)
        logvar_x = nn.Dense(self.latents, name='fc2_logvar')(x)
        
        return mean_x, logvar_x

    
class Decoder(nn.Module):
    
    recon_shape: int = 48 * 1876
    
    @nn.compact
    def __call__(self, z):
        z = nn.Dense(500, name='fc1')(z)
        z = nn.relu(z)
        z = nn.Dense(self.recon_shape, name='fc2')(z)
        return z

    
def reparameterize(rng, mean, logvar):
    std = jnp.exp(0.5 * logvar)
    eps = jax.random.normal(rng, logvar.shape)
    return mean + eps * std    
    
class CVAE(nn.Module):
    latents: int = 20
    recon_shape: int = 48 * 1876
    def setup(self):
        self.encoder = Encoder(self.latents)
        self.decoder = Decoder(self.recon_shape)

    def __call__(self, x, y, z_rng):
     
        x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
        x = jnp.concatenate((x,y), axis=-1)    
        
        mean, logvar = self.encoder(x)
        z = reparameterize(z_rng, mean, logvar)
        recon_x = self.decoder(z)
        return recon_x, mean, logvar

    def generate(self, z):
        return nn.sigmoid(self.decoder(z))


