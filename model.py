import jax
import jax.numpy as jnp
from jax import random

from flax import linen as nn
from typing import Callable, Any, Optional


class SampleCNN(nn.Module):
    
    deterministic:bool = False
    
    @nn.compact
    def __call__(self, x):

        x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
        x = jnp.expand_dims(x, axis=2)
        
        # 1
        x = nn.Conv(features=128, kernel_size=(3,), strides=3, padding='valid')(x)
        x = nn.normalization.BatchNorm(True)(x)
        x = nn.relu(x)
        
        # 2 
        x = nn.Conv(features=128, kernel_size=(3,), strides=1, padding='same')(x)
        x = nn.normalization.BatchNorm(True)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3,), strides=(3,))
        
        # 3
        x = nn.Conv(features=128, kernel_size=(3,), strides=1, padding='same')(x)
        x = nn.normalization.BatchNorm(True)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3,), strides=(3,))

        # 4
        x = nn.Conv(features=256, kernel_size=(3,), strides=1, padding='same')(x)
        x = nn.normalization.BatchNorm(True)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3,), strides=(3,))
        
        # 5
        x = nn.Conv(features=256, kernel_size=(3,), strides=1, padding='same')(x)
        x = nn.normalization.BatchNorm(True)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3,), strides=(3,))

        # 6
        x = nn.Conv(features=256, kernel_size=(3,), strides=1, padding='same')(x)
        x = nn.normalization.BatchNorm(True)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3,), strides=(3,))

        # 7
        x = nn.Conv(features=256, kernel_size=(3,), strides=1, padding='same')(x)
        x = nn.normalization.BatchNorm(True)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3,), strides=(3,))
        x = nn.Dropout(rate=0.5)(x, deterministic=self.deterministic)

        # 8
        x = nn.Conv(features=256, kernel_size=(3,), strides=1, padding='same')(x)
        x = nn.normalization.BatchNorm(True)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3,), strides=(3,))
        
        # 9
        x = nn.Conv(features=256, kernel_size=(3,), strides=1, padding='same')(x)
        x = nn.normalization.BatchNorm(True)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3,), strides=(3,))
        
        
        # 10
        x = nn.Conv(features=512, kernel_size=(3,), strides=1, padding='same')(x)
        x = nn.normalization.BatchNorm(True)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3,), strides=(3,))
        
        
        # 11
        x = nn.Conv(features=256, kernel_size=(3,), strides=1, padding='same')(x)
        x = nn.normalization.BatchNorm(True)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=0.5)(x, deterministic=self.deterministic)
        
        # fc
        x = x.reshape((x.shape[0], -1))
        x = nn.sigmoid(nn.Dense(30)(x))
        return x 
