from dataloader import mel_dataset
from torch.utils.data import DataLoader, random_split
from Conv2d_model import Conv2d_VAE
from linear_evaluation import linear_evaluation

import flax 
import flax.linen as nn
from flax.training import train_state

import jax
import numpy as np
import jax.numpy as jnp
import optax
from tqdm import tqdm
import os
import wandb
import matplotlib.pyplot as plt

wandb.init(
    project='Conv2d_VAE_linear',
    entity='aiffelthon'
)


def collate_batch(batch):
    x_train = [x for x, _ in batch]
    y_train = [y for _, y in batch]                  
        
    return np.array(x_train), np.array(y_train)


def init_state(model, x_shape, key, lr) -> train_state.TrainState:
    params = model.init({'params': key}, jnp.ones(x_shape), key)
    # Create the optimizer
    optimizer = optax.adam(learning_rate=lr)
    # Create a State
    return train_state.TrainState.create(
        apply_fn=model.apply,
        tx=optimizer,
        params=params)


def linear_init_state(model, x_shape, lr) -> train_state.TrainState:
    params = model.init({'params': key}, jnp.ones(x_shape))
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

@jax.vmap
def binary_cross_entropy_with_logits(logits, labels):
    logits = nn.log_sigmoid(logits)
    return -jnp.sum(labels * logits + (1. - labels) * jnp.log(-jnp.expm1(logits)))


@jax.jit
def train_step(state, x, z_rng):
    
    x = jnp.expand_dims(x, axis=-1)
    
    def loss_fn(params):
        recon_x, mean, logvar = Conv2d_VAE().apply(params, x, z_rng)

        mse_loss = ((recon_x - x)**2).mean()
        kld_loss = kl_divergence(mean, logvar).mean()
        loss = mse_loss + kld_loss
        return loss
    
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    
    return state.apply_gradients(grads=grads), loss

@jax.jit
def eval_step(state, x, z_rng):
    x = jnp.expand_dims(x, axis=-1)
    recon_x, mean, logvar = Conv2d_VAE().apply(state.params, x, z_rng)
    mse_loss = ((recon_x -x)**2).mean()
    kld_loss = kl_divergence(mean, logvar).mean()
    loss = mse_loss + kld_loss
    
    return recon_x, loss, mse_loss, kld_loss

@jax.jit
def linear_train_step(state, x, y, z_rng):
    
    x = jnp.expand_dims(x, axis=-1)
    latent_vector = Encoder().apply({'params':enc_params,'batch_stats':enc_batch_stats}, x, rng)
    
    def loss_fn(params):
        logits = linear_evaluation().apply(params, latent_vector, z_rng)

        loss = optax.softmax_cross_entropy(logtis, y)
        return loss
    
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    
    return state.apply_gradients(grads=grads), loss

if __name__ == "__main__":
    batch_size = 16
    lr = 0.0001
    rng = jax.random.PRNGKey(303)
    
    print('\n')
    # ---Load dataset---
    print("Loading dataset...")
    dataset_dir = os.path.join(os.path.expanduser('~'),'dataset')
    data = mel_dataset(dataset_dir)
    print(f'Loaded data : {len(data)}\n')
    target = data[0][0] # (48, 1876)
    target = jnp.expand_dims(target, axis = 0)
    
    dataset_size = len(data)
    train_size = int(dataset_size * 0.8)
    test_size = dataset_size - train_size
    
    train_dataset, test_dataset = random_split(data, [train_size, test_size])

    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_batch)
    test_dataloader = DataLoader(test_dataset, batch_size=int(batch_size/4), shuffle=True, num_workers=0, collate_fn=collate_batch)
    
    print(f'batch_size = {batch_size}')
    print(f'learning rate = {lr}')
    print(f'train_size = {train_size}')
    print(f'test_size = {test_size}')
    
    
    print('Data load complete!\n')

    # ---initializing model---
    model = Conv2d_VAE()
    print("Initializing model....")
    state = init_state(model, 
                       jnp.expand_dims(next(iter(train_dataloader))[0], axis=-1).shape,
                       rng, 
                       lr)
    
    print("Initialize complete!!\n")
    # ---train model---
    epoch = 5
    # checkpoint_dir = str(input('checkpoint dir : '))
    
    
    for i in range(epoch):
        train_data = iter(train_dataloader)
        test_data = iter(test_dataloader)
        
        train_loss_mean = 0
        test_loss_mean = 0
        
        
        print(f'\nEpoch {i+1}')
        
        for j in range(len(train_dataloader)):
            rng, key = jax.random.split(rng)
            x, y = next(train_data)
            test_x, test_y = next(test_data)
            
            # x = x + 100
            # test_x = test_x + 100
            
            state, train_loss = train_step(state, x, rng)           
            _, test_loss, mse_loss, kld_loss = eval_step(state, test_x, rng)
            
            recon_x, _, _, _ = eval_step(state, target, rng)
            wandb.log({'train_loss' : train_loss, 'test_loss' : test_loss, 'mse_loss':mse_loss, 'kld_loss':kld_loss})
            train_loss_mean += train_loss
            test_loss_mean += test_loss
            
            
            
        
                
                
            print(f'step : {j}/{len(train_dataloader)}, train_loss : {round(train_loss, 3)}, test_loss : {round(test_loss, 3)}', end='\r')

        print(f'epoch {i+1} - average loss - train : {round(train_loss_mean/len(train_dataloader), 3)}, test : {round(test_loss_mean/len(test_dataloader), 3)}')
        
    # ---Linear evaluation---
    
            
    print('Prepare linear evaluation...')
    
    enc_params = state.params['params']['encoder']
    enc_batch_stats = state.params['batch_stats']['encoder']


    @jax.jit
    def encoder_apply(data, rng):
        latent_vector = Encoder().apply({'params':enc_params,'batch_stats':enc_batch_stats}, data, rng)
        return latent_vector
    
    whole_latent_vector = []
    for single_data in data:
        latent_vector = encoder_apply(np.expand_dims(single_data[0][0], axis=(0,-1)), rng)
        whole_latent_vector.append(latent_vector)
        
    whole_latent_vector = np.array(whole_latent_vector)
    print(f'Latent vector shape : {whole_latent_vector.shape}'
          
    linear_state = linear_init_state(linear_evaluation(), )
    
    def linear_init_state(model, x_shape, lr) -> train_state.TrainState:
        params = model.init({'params': key}, jnp.ones(x_shape))
        # Create the optimizer
        optimizer = optax.adam(learning_rate=lr)
        # Create a State
        return train_state.TrainState.create(
            apply_fn=model.apply,
            tx=optimizer,
            params=params)

    
wandb.finish()



