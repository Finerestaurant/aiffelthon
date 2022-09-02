import jax
import jax.numpy as jnp
from jax import random

from flax import linen as nn
from typing import Callable, Any, Optional


def reparameterize(rng, mean, logvar):
    std = jnp.exp(0.5 * logvar)
    eps = jax.random.normal(rng, logvar.shape)
    return mean + eps * std    

class Encoder(nn.Module):
    
    dilation:bool=False
    @nn.compact
    def __call__(self, x, rng):
        if self.dilation:
            x = nn.Conv(512,kernel_size=(3,), strides=1, padding='same', kernel_dilation=1)(x)
        else:
            x = nn.Conv(512,kernel_size=(3,), strides=1, padding='same')(x)
        x = nn.relu(x)
        x = nn.normalization.BatchNorm(True)(x)
        
        if self.dilation:
            x = nn.Conv(256,kernel_size=(3,), strides=2, padding='same', kernel_dilation=2)(x)
        else:
            x = nn.Conv(256,kernel_size=(3,), strides=2, padding='same')(x)
        x = nn.relu(x)
        x = nn.normalization.BatchNorm(True)(x)
        
        if self.dilation:
            x = nn.Conv(128,kernel_size=(3,), strides=1, padding='same', kernel_dilation=4)(x)
        else:
            x = nn.Conv(128,kernel_size=(3,), strides=1, padding='same')(x)
        x = nn.relu(x)
        x = nn.normalization.BatchNorm(True)(x)
        
        x = nn.Conv(128,kernel_size=(3,), strides=2, padding='same', kernel_dilation=8)(x)
        x = nn.relu(x)
        x = nn.normalization.BatchNorm(True)(x)
        
        if self.dilation:
            x = nn.Conv(64,kernel_size=(3,), strides=1, padding='same', kernel_dilation=8)(x)
        else:
            x = nn.Conv(64,kernel_size=(3,), strides=1, padding='same')(x)
        x = nn.relu(x)
        x = nn.normalization.BatchNorm(True)(x)
        x = nn.max_pool(x, window_shape=(2,), strides=(2,))
        
        if self.dilation:
            x = nn.Conv(32,kernel_size=(3,), strides=2, padding='same',kernel_dilation=16)(x)
        else:
            x = nn.Conv(32,kernel_size=(3,), strides=2, padding='same')(x)
        x = nn.relu(x)
        x = nn.normalization.BatchNorm(True)(x)
        
        if self.dilation:                
            x = nn.Conv(1,kernel_size=(3,), strides=1, padding='same',kernel_dilation=32)(x)
        else:
            x = nn.Conv(1,kernel_size=(3,), strides=1, padding='same')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2,), strides=(2,))
        x = nn.normalization.BatchNorm(True)(x)

        
        mean_x = nn.Dense(20, name='fc3_mean')(x) 
        logvar_x = nn.Dense(20, name='fc3_logvar')(x)
        
        z = reparameterize(rng, mean_x, logvar_x)
        
        return z, mean_x, logvar_x
    
class Decoder(nn.Module):

    recon_shape : int = 1876
    dilation:bool=False
    @nn.compact
    def __call__(self, x):
        
        if self.dilation:            
            x = nn.ConvTranspose(64, kernel_size=(3,), strides=[1,],kernel_dilation=(16,))(x)
        else:
            x = nn.ConvTranspose(64, kernel_size=(3,), strides=[1,])(x)
        x = nn.relu(x)
        x = nn.normalization.BatchNorm(True)(x)

        if self.dilation:            
            x = nn.ConvTranspose(128, kernel_size=(3,), strides=[2,], kernel_dilation=(8,))(x)
        else:
            x = nn.ConvTranspose(128, kernel_size=(3,), strides=[2,])(x)
        x = nn.relu(x)
        x = nn.normalization.BatchNorm(True)(x)
        
        if self.dilation:          
            x = nn.ConvTranspose(256, kernel_size=(3,), strides=[2,], kernel_dilation=(8,))(x)
        else:
            x = nn.ConvTranspose(256, kernel_size=(3,), strides=[2,])(x)
        x = nn.relu(x)
        x = nn.normalization.BatchNorm(True)(x)
        
        if self.dilation:          
            x = nn.ConvTranspose(512, kernel_size=(3,), strides=[3,],  kernel_dilation=(4,))(x)
        else:
            x = nn.ConvTranspose(512, kernel_size=(3,), strides=[3,])(x)
        x = nn.relu(x)
        x = nn.normalization.BatchNorm(True)(x)

        if self.dilation:
            x = nn.ConvTranspose(1024, kernel_size=(3,), strides=[2,], kernel_dilation=(2,))(x)
        else:
            x = nn.ConvTranspose(1024, kernel_size=(3,), strides=[2,])(x)
        x = nn.relu(x)
        x = nn.normalization.BatchNorm(True)(x)
        
        if self.dilation:
            x = nn.ConvTranspose(self.recon_shape, kernel_size=(3,), strides=[2,], kernel_dilation=(1,))(x)
        else:
            x = nn.ConvTranspose(self.recon_shape, kernel_size=(3,), strides=[2,])(x)
        
        return x
        


    
class Conv1d_VAE(nn.Module):
    
    dilation:bool=False
    def setup(self):
        self.encoder = Encoder(dilation=self.dilation)
        self.decoder = Decoder(dilation=self.dilation)

    def __call__(self, x, z_rng):
     
        x = x.reshape(x.shape[0],x.shape[1],x.shape[2])   
        z, mean, logvar = self.encoder(x, z_rng) 
        recon_x = self.decoder(z)
        return recon_x, mean, logvar
    
if __name__=='__main__':
    
    test_input = jnp.ones((12, 48, 1876))
    test_label = jnp.ones((12, 30))
    
    test_latent = jnp.ones((12, 20))

    key = jax.random.PRNGKey(32)
    
    params = Conv1d_VAE(dilation=True).init({'params': key}, test_input , key)
    result = Conv1d_VAE(dilation=True).apply(params, test_input, key)

    params = Conv1d_VAE(dilation=False).init({'params': key}, test_input , key)
    result = Conv1d_VAE(dilation=False).apply(params, test_input, key)

    print('test complete!')