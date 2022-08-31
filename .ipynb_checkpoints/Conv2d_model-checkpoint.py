import jax
import jax.numpy as jnp
from jax import random

from flax import linen as nn
from typing import Callable, Any, Optional


class Encoder(nn.Module):

    @nn.compact
    def __call__(self, x):
        
        x = nn.Conv(512, kernel_size=(3,3),  strides=[3,3], padding='same')(x)
        x = nn.relu(x)
        # x = nn.normalization.BatchNorm(True)(x)
        x = nn.max_pool(x, window_shape=(2,2), strides=(2,2))

        
        x = nn.Conv(512,kernel_size=(3,3),  padding='same')(x)
        x = nn.relu(x)
        # x = nn.normalization.BatchNorm(True)(x)
        x = nn.max_pool(x, window_shape=(2,2), strides=(2,2))

        
        x = nn.Conv(256,kernel_size=(3,3),  padding='same')(x)
        x = nn.relu(x)
        # x = nn.normalization.BatchNorm(True)(x)
 

        x = nn.Conv(128,kernel_size=(3,3), padding='same')(x)
        x = nn.relu(x)
        # x = nn.normalization.BatchNorm(True)(x)
        
        x = nn.Conv(64,kernel_size=(3,3), padding='same')(x)
        x = nn.relu(x)
        # x = nn.normalization.BatchNorm(True)(x)
        
        
        x = nn.Conv(32, kernel_size=(3, 3),  padding='same')(x)
        x = nn.relu(x)
        # x = nn.normalization.BatchNorm(True)(x)
        
        
        x = nn.Conv(16, kernel_size=(2,2), strides=[1,1], padding='same')(x)
        x = nn.relu(x)
        # x = nn.normalization.BatchNorm(True)(x)
        
        x = nn.Conv(1,kernel_size=(2,2), strides=[1,1])(x)
        x = nn.relu(x)
        # x = nn.normalization.BatchNorm(True)(x)

        
        x = x.reshape(x.shape[0], -1)
        
        
        mean_x = nn.Dense(20, name='fc3_mean')(x)
        logvar_x = nn.Dense(20, name='fc3_logvar')(x)
        
        
        return mean_x, logvar_x
    
    
class Decoder(nn.Module):
    
    @nn.compact
    def __call__(self, x):
        
        x = nn.Dense(12 * 469 * 1)(x)
        x = nn.relu(x)
        
        x = x.reshape(x.shape[0], 12, 469, 1)
        
        x = nn.ConvTranspose(128, kernel_size=(3,3), strides=[1,1])(x)
        x = nn.relu(x)
        # x = nn.normalization.BatchNorm(True)(x)
        
        x = nn.ConvTranspose(256, kernel_size=(3,3))(x)
        x = nn.relu(x)
        # x = nn.normalization.BatchNorm(True)(x)        
        
        x = nn.ConvTranspose(512, kernel_size=(3,3), strides=[2,2])(x)
        x = nn.relu(x)
        # x = nn.normalization.BatchNorm(True)(x)
        
        x = nn.ConvTranspose(1024, kernel_size=(3,3))(x)
        x = nn.relu(x)
        
        x = nn.ConvTranspose(1, kernel_size=(3,3), strides=[2,2])(x)
        x = nn.relu(x)

        return x
    
    
    
def reparameterize(rng, mean, logvar):
    std = jnp.exp(0.5 * logvar)
    eps = random.normal(rng, logvar.shape)
    return mean + eps * std    
    
    
    
class Conv2d_VAE(nn.Module):

    def setup(self):
        self.encoder = Encoder()
        self.decoder = Decoder()

    def __call__(self, x, z_rng):
     
        x = x.reshape(*x.shape, 1)

        mean, logvar = self.encoder(x)
        z = reparameterize(z_rng, mean, logvar)
        
        recon_x = self.decoder(z)
        return recon_x, mean, logvar
