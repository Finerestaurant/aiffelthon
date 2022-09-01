import jax
import jax.numpy as jnp
from jax import random

from flax import linen as nn
from typing import Callable, Any, Optional


    
def reparameterize(rng, mean, logvar):
    std = jnp.exp(0.5 * logvar)
    eps = random.normal(rng, logvar.shape)
    return mean + eps * std    

class Encoder(nn.Module):

    @nn.compact
    def __call__(self, x, z_rng):
        
        x = nn.Conv(512, kernel_size=(3,3),  strides=[2,2], padding='same')(x)
        x = jax.nn.leaky_relu(x)
        x = nn.normalization.BatchNorm(True)(x)
        # x = nn.max_pool(x, window_shape=(2,2), strides=(2,2))

        
        x = nn.Conv(512,kernel_size=(3,3),  padding='same')(x)
        x = jax.nn.leaky_relu(x)
        x = nn.normalization.BatchNorm(True)(x)
        x = nn.max_pool(x, window_shape=(2,2), strides=(2,2))

        
        x = nn.Conv(256,kernel_size=(3,3),  padding='same')(x)
        x = jax.nn.leaky_relu(x)
        x = nn.normalization.BatchNorm(True)(x)
 

        x = nn.Conv(128,kernel_size=(3,3), padding='same')(x)
        x = jax.nn.leaky_relu(x)
        x = nn.normalization.BatchNorm(True)(x)
        
        x = nn.Conv(64,kernel_size=(3,3), padding='same')(x)
        x = jax.nn.leaky_relu(x)
        x = nn.normalization.BatchNorm(True)(x)
        
        
        x = nn.Conv(32, kernel_size=(3,3),  padding='same')(x)
        x = jax.nn.leaky_relu(x)
        x = nn.normalization.BatchNorm(True)(x)
        
        
        x = nn.Conv(16, kernel_size=(3,3), strides=[1,1], padding='same')(x)
        x = jax.nn.leaky_relu(x)
        x = nn.normalization.BatchNorm(True)(x)
        
        x = nn.Conv(1,kernel_size=(3,3), strides=[1,1],  padding='same')(x)
        x = jax.nn.leaky_relu(x)
        x = nn.normalization.BatchNorm(True)(x)

        
        x = x.reshape(x.shape[0], -1) 
        
        
        mean_x = nn.Dense(512, name='fc3_mean')(x)
        logvar_x = nn.Dense(512, name='fc3_logvar')(x)  # (128, 12, 469, 20)
        
        z = reparameterize(z_rng, mean_x, logvar_x)
        
        return z, mean_x, logvar_x
    
    
class Decoder(nn.Module):
    
    @nn.compact
    def __call__(self, x):
        
        x = nn.Dense(12 * 469 * 1)(x)
        x = jax.nn.leaky_relu(x)
        
        x = x.reshape(x.shape[0], 12, 469, 1)
        
    
         # (128, 12, 469, 20)
        x = nn.ConvTranspose(128, kernel_size=(3,3), strides=[1,1])(x)
        x = jax.nn.leaky_relu(x)
        x = nn.normalization.BatchNorm(True)(x)
        
        x = nn.ConvTranspose(256, kernel_size=(3,3))(x)
        x = jax.nn.leaky_relu(x)
        x = nn.normalization.BatchNorm(True)(x)        
        
        x = nn.ConvTranspose(512, kernel_size=(3,3), strides=[2,2])(x)
        x = jax.nn.leaky_relu(x)
        x = nn.normalization.BatchNorm(True)(x)
        
        x = nn.ConvTranspose(1024, kernel_size=(3,3))(x)
        x = jax.nn.leaky_relu(x)
        
        x = nn.ConvTranspose(1, kernel_size=(3,3), strides=[2,2])(x)
        x = jax.nn.leaky_relu(x)

        return x
    
    

    
    
    
class Conv2d_VAE(nn.Module):

    def setup(self):
        self.encoder = Encoder()
        self.decoder = Decoder()

    def __call__(self, x, z_rng):
     
        # x = x.reshape(*x.shape, 1)

        z, mean, logvar = self.encoder(x, z_rng)

        recon_x = self.decoder(z)
        return recon_x, mean, logvar
