import jax
import jax.numpy as jnp
from jax import random

from flax import linen as nn
from typing import Callable, Any, Optional


# SampleCNN implementation

class SampleCNN(nn.Module):
    
    deterministic:bool = False
    
    @nn.compact
    def __call__(self, x):
        
        # x = jnp.expand_dims(x, axis=-1)
        
        # 1
        x = nn.Conv(features=128, kernel_size=(3,3), strides=(1,3), padding='valid')(x)
        x = nn.normalization.BatchNorm(True)(x)
        x = nn.relu(x)
        
        # 2 
        x = nn.Conv(features=128, kernel_size=(3,3), strides=(1,2), padding='valid')(x)
        x = nn.normalization.BatchNorm(True)(x)
        x = nn.relu(x)
        # x = nn.max_pool(x, window_shape=(3,), strides=(3,))
        
        # 3
        x = nn.Conv(features=128, kernel_size=(3,3), strides=(1,1), padding='valid')(x)
        x = nn.normalization.BatchNorm(True)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3,3), strides=(3,3))

        # 4
        x = nn.Conv(features=256, kernel_size=(3,3), strides=(1,1), padding='valid')(x)
        x = nn.normalization.BatchNorm(True)(x)
        x = nn.relu(x)
        # x = nn.max_pool(x, window_shape=(3,), strides=(3,))
        
        # 5
        x = nn.Conv(features=256, kernel_size=(3,3), strides=(1,1), padding='valid')(x)
        x = nn.normalization.BatchNorm(True)(x)
        x = nn.relu(x)
        # x = nn.max_pool(x, window_shape=(3,), strides=(3,))

        # 6
        x = nn.Conv(features=256, kernel_size=(3,3), strides=(1,1), padding='same')(x)
        x = nn.normalization.BatchNorm(True)(x)
        x = nn.relu(x)
        # x = nn.max_pool(x, window_shape=(3,), strides=(3,))

        # 7
        x = nn.Conv(features=512, kernel_size=(3,3), strides=1, padding='same')(x)
        x = nn.normalization.BatchNorm(True)(x)
        x = nn.relu(x)
        # x = nn.max_pool(x, window_shape=(3,), strides=(3,))
        x = nn.Dropout(rate=0.5)(x, deterministic=self.deterministic)

        # 8
        x = nn.Conv(features=256, kernel_size=(3,3), strides=1, padding='same')(x)
        x = nn.normalization.BatchNorm(True)(x)
        x = nn.relu(x)
        # x = nn.max_pool(x, window_shape=(3,), strides=(3,))
        
        # 9
        x = nn.Conv(features=128, kernel_size=(3,3), strides=1, padding='same')(x)
        x = nn.normalization.BatchNorm(True)(x)
        x = nn.relu(x)
        # x = nn.max_pool(x, window_shape=(3,), strides=(3,))
        
        
        # 10
        x = nn.Conv(features=128, kernel_size=(3,3), strides=1, padding='valid')(x)
        x = nn.normalization.BatchNorm(True)(x)
        x = nn.relu(x)
        # x = nn.max_pool(x, window_shape=(3,), strides=(3,))
        
        
        # 11
        x = nn.Conv(features=64, kernel_size=(3,3), strides=1, padding='valid')(x)
        x = nn.normalization.BatchNorm(True)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=0.5)(x, deterministic=self.deterministic)
        
        # fc
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512)(x)
        x = nn.Dense(512)(x)
        x = nn.Dense(30)(x)
        return x 


# CVAE implementation    

class Encoder(nn.Module):
    
    latents: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(500, name='fc1')(x)
        x = nn.relu(x)
        x = nn.Dense(500, name='fc2')(x)
        x = nn.relu(x)
        x = nn.Dense(500, name='fc3')(x)
        x = nn.relu(x)
        mean_x = nn.Dense(self.latents, name='fc3_mean')(x)
        logvar_x = nn.Dense(self.latents, name='fc3_logvar')(x)
        
        return mean_x, logvar_x

    
class Decoder(nn.Module):
    
    recon_shape: int = 48 * 1876
    
    @nn.compact
    def __call__(self, z):
        z = nn.Dense(500, name='fc4')(z)
        z = nn.relu(z)
        z = nn.Dense(500, name='fc5')(z)
        z = nn.relu(z)
        z = nn.Dense(500, name='fc6')(z)
        z = nn.relu(z)
        z = nn.Dense(self.recon_shape, name='fc7')(z)
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