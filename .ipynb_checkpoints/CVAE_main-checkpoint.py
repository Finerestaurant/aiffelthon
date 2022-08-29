from dataloader import mel_dataset
from torch.utils.data import DataLoader
from CVAE import CVAE

import flax 
import flax.linen as nn
from flax.training import train_state

import jax
import numpy as np
import jax.numpy as jnp
import optax
from tqdm import tqdm

# print(jax.local_devices())

def collate_batch(batch):
    x_train = [x for x, _ in batch]
    y_train = [y for _, y in batch]                  
        
    return np.array(x_train), np.array(y_train)


def init_state(model, x_shape, y_shape, key, lr) -> train_state.TrainState:
    params = model.init({'params': key}, jnp.ones(x_shape), jnp.ones(y_shape), key)
    # Create the optimizer
    optimizer = optax.adam(learning_rate=lr)
    # Create a State
    return train_state.TrainState.create(
        apply_fn=model.apply,
        tx=optimizer,
        params=params)

@jax.vmap
def kl_divergence(mean, logvar):
    return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))


# @jax.vmap
# def binary_cross_entropy_with_logits(logits, labels):
#     logits = nn.log_sigmoid(logits)
#     return -jnp.sum(labels * logits + (1. - labels) * jnp.log(-jnp.expm1(logits)))


@jax.jit
def train_step(state, x, y, z_rng):
    
    def loss_fn(params):
        recon_x, mean, logvar = CVAE().apply({'params': params['params']}, x, y , z_rng)

        bce_loss = optax.sigmoid_binary_cross_entropy(recon_x, x.reshape(x.shape[0], x.shape[1]*x.shape[2])).mean()
        kld_loss = kl_divergence(mean, logvar).mean()
        loss = bce_loss + kld_loss
        return loss
    
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    
    return state.apply_gradients(grads=grads), loss



if __name__ == "__main__":
    batch_size = 128
    lr = 0.00001
    rng = jax.random.PRNGKey(303)
    
    # ---Load dataset---
    print("Loading dataset...")
    dataset_dir = '/home/anthonypark6904/dataset'
    data = mel_dataset(dataset_dir)
    print(f'Loaded data : {len(data)}')
    train_dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_batch)
    print(f'batch_size = {batch_size}')
    print(f'learning rate = {lr}')
    
    print('Data load complete!\n')

    # ---initializing model---
    model = CVAE()
    print("Initializing model....")
    state = init_state(model, 
                       next(iter(train_dataloader))[0].shape, 
                       next(iter(train_dataloader))[1].shape,
                       rng, 
                       lr)
    
    print("Initialize complete!!\n")
    # ---train model---
    epoch = 50
    # checkpoint_dir = str(input('checkpoint dir : '))
    

    for i in range(epoch):
        train_data = iter(train_dataloader)
        loss_mean = 0
        print(f'\nEpoch {i+1}')
        
        for j in range(len(train_dataloader)):
            rng, key = jax.random.split(rng)
            x, y = next(train_data)
            x = (x / 100) + 1
            state, loss = train_step(state, x, y, rng)
            
            loss_mean += loss

            print(f'step : {j}/{len(train_dataloader)}, loss : {loss}', end='\r')

        print(f'epoch {i+1} - average loss : {loss_mean/len(train_dataloader)}')
        




