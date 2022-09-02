from dataloader import mel_dataset
from torch.utils.data import DataLoader, random_split
from Dilated_Conv1D_vae_model import Dilated_Conv1d_VAE
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



# print(jax.local_devices())

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

@jax.vmap
def kl_divergence(mean, logvar):
    return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))


# @jax.vmap
# def binary_cross_entropy_with_logits(logits, labels):
#     logits = nn.log_sigmoid(logits)
#     return -jnp.sum(labels * logits + (1. - labels) * jnp.log(-jnp.expm1(logits)))


@jax.jit
def train_step(state, x, z_rng):
    
    def loss_fn(params):
        recon_x, mean, logvar = Dilated_Conv1d_VAE().apply(params, x, z_rng)

        # bce_loss = binary_cross_entropy_with_logits(recon_x, x.reshape(x.shape[0], x.shape[1]*x.shape[2])).mean()
        kld_loss = kl_divergence(mean, logvar).mean()
        mse_loss = ((recon_x - jnp.expand_dims(x, axis=-1))**2).mean()
        loss = mse_loss + kld_loss
        return loss
    
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    
    return state.apply_gradients(grads=grads), loss

@jax.jit
def eval_step(state, x, z_rng):
    
    recon_x, mean, logvar = Dilated_Conv1d_VAE().apply(state.params, x, z_rng)
    # bce_loss = binary_cross_entropy_with_logits(recon_x, x.reshape(x.shape[0], x.shape[1]*x.shape[2])).mean()
    kld_loss = kl_divergence(mean, logvar).mean()
    mse_loss = ((recon_x - jnp.expand_dims(x, axis=-1))**2).mean()
    loss = mse_loss + kld_loss
    
    return recon_x, loss, mse_loss, kld_loss


if __name__ == "__main__":
    batch_size = 16
    lr = 0.0001
    rng = jax.random.PRNGKey(303)
    
    
    # ---Load dataset---
    dev_or_train = input('dev or train? : ')
    if dev_or_train == 'dev':
        dataset_dir = os.path.join(os.path.expanduser('~'),'dev_dataset')
    else:
        dataset_dir = os.path.join(os.path.expanduser('~'),'dataset')
        
        
    wandb.init(
    project='dilated_Conv2d_VAE_linear',
    entity='aiffelthon'
    )

    print("Loading dataset...")
    
    data = mel_dataset(dataset_dir)
    print(f'Loaded data : {len(data)}')
    
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
    model = Dilated_Conv2d_VAE()
    print("Initializing model....")
    state = init_state(model, 
                       next(iter(train_dataloader))[0].shape, 
                       rng, 
                       lr)
    
    print("Initialize complete!!\n")
    # ---train model---
    epoch = 100
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
            
#             x = (x + 100) 
#             test_x = (test_x + 100)
            
            state, train_loss = train_step(state, x, rng)           
            recon_x, test_loss, mse_loss, kld_loss = eval_step(state, test_x, rng)
            wandb.log({'train_loss' : train_loss, 'test_loss' : test_loss, 'mse_loss':mse_loss,'kld_loss':kld_loss})
            train_loss_mean += train_loss
            test_loss_mean += test_loss
            
            if j % 100 == 0:
                
                recon_x = recon_x.reshape(recon_x.shape[0], x.shape[1], x.shape[2])       
                
                fig1, ax1 = plt.subplots()
                im1 = ax1.imshow(recon_x[0], aspect='auto', origin='lower', interpolation='none')
                fig1.colorbar(im1)
                fig1.savefig('recon.png')


                fig2, ax2 = plt.subplots()
                im2 = ax2.imshow(test_x[0], aspect='auto', origin='lower', interpolation='none')
                fig2.colorbar(im2)
                fig2.savefig('x.png')
                
                wandb.log({'reconstruction' : [
                            wandb.Image('recon.png')
                            ], 
                           'original image' : [
                            wandb.Image('x.png')
                            ]})
                
            print(f'step : {j}/{len(train_dataloader)}, train_loss : {round(train_loss, 3)}, test_loss : {round(test_loss, 3)}', end='\r')

        print(f'epoch {i+1} - average loss - train : {round(train_loss_mean/len(train_dataloader), 3)}, test : {round(test_loss_mean/len(test_dataloader), 3)}')
    
wandb.finish()
