import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import random
import numpy as np 
import flax
from typing import Callable, Any, Optional
from functools import partial
from jax.nn.initializers import normal as normal_init


class Encoder(nn.Module):
    
    latents: int = 20

    @nn.compact
    def __call__(self, x):
        
        x = nn.Conv(512,kernel_size=(2,), strides=1, padding='same')(x)
        x = nn.relu(x)
        x = nn.normalization.BatchNorm(True)(x)
        # x = nn.max_pool(x, window_shape=(2,), strides=(1,))

        x = nn.Conv(256,kernel_size=(2,), strides=2, padding='same')(x)
        x = nn.relu(x)
        x = nn.normalization.BatchNorm(True)(x)
        x = nn.max_pool(x, window_shape=(2,), strides=(2,))
        
        x = nn.Conv(128,kernel_size=(2,), strides=1, padding='same')(x)
        x = nn.relu(x)
        x = nn.normalization.BatchNorm(True)(x)
        x = nn.max_pool(x, window_shape=(2,), strides=(2,))
        
        x = nn.Conv(128,kernel_size=(2,), strides=1, padding='same')(x)
        x = nn.relu(x)
        x = nn.normalization.BatchNorm(True)(x)
        
        x = nn.Conv(64,kernel_size=(2,), strides=1, padding='same')(x)
        x = nn.relu(x)
        x = nn.normalization.BatchNorm(True)(x)
        x = nn.max_pool(x, window_shape=(2,), strides=(2,))

        x = nn.Conv(32,kernel_size=(2,), strides=1, padding='same')(x)
        x = nn.relu(x)
        x = nn.normalization.BatchNorm(True)(x)
        x = nn.max_pool(x, window_shape=(2,), strides=(2,))

        # x = nn.Conv(1,kernel_size=(2,2), strides=1, padding='same')(x)
        # x = nn.relu(x)
        # x = nn.max_pool(x, window_shape=(2,), strides=(2,))
        # x = nn.normalization.BatchNorm(True)(x)

        
        mean_x = nn.Dense(self.latents, name='fc3_mean')(x)
        logvar_x = nn.Dense(self.latents, name='fc3_logvar')(x)
        
        return mean_x, logvar_x
    
    
class Decoder(nn.Module):

    recon_shape : int = 1848
    
    @nn.compact
    def __call__(self, x):

        x = nn.ConvTranspose(32, kernel_size=(2,), strides=[2,])(x)
        x = nn.relu(x)
        x = nn.normalization.BatchNorm(True)(x)

        x = nn.ConvTranspose(64, kernel_size=(2,), strides=[2,])(x)
        x = nn.relu(x)
        x = nn.normalization.BatchNorm(True)(x)
        
        x = nn.ConvTranspose(128, kernel_size=(2,), strides=[3,])(x)
        x = nn.relu(x)
        x = nn.normalization.BatchNorm(True)(x)
        
        x = nn.ConvTranspose(256, kernel_size=(2,), strides=[2,])(x)
        x = nn.relu(x)
        x = nn.normalization.BatchNorm(True)(x)
        
        x = nn.ConvTranspose(512, kernel_size=(2,), strides=[2,])(x)
        x = nn.relu(x)
        x = nn.normalization.BatchNorm(True)(x)
        
        x = nn.ConvTranspose(self.recon_shape, kernel_size=(2,), strides=[1,])(x)
        
        return x
    
def reparameterize(rng, mean, logvar):
    std = jnp.exp(0.5 * logvar)
    eps = jax.random.normal(rng, logvar.shape)
    return mean + eps * std    

class Conv1d_CVAE(nn.Module):
    latents: int = 20
    recon_shape: int = 1876
    
    def setup(self):
        self.encoder = Encoder(self.latents)
        self.decoder = Decoder(self.recon_shape)

    def __call__(self, x, z_rng):
     
        x = x.reshape(x.shape[0],x.shape[1],x.shape[2])        
        mean, logvar = self.encoder(x)
        z = reparameterize(z_rng, mean, logvar)
        recon_x = self.decoder(z)
        return recon_x, mean, logvar    