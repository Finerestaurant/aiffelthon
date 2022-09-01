import jax
import jax.numpy as jnp
from jax import random

from flax import linen as nn
from typing import Callable, Any, Optional


class linear_evaluation(nn.Module):
    
    n_features:int = 30
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(512)(x)
        x = jax.nn.leaky_relu(x)
        x = nn.Dense(self.n_features)
        
        return x
            