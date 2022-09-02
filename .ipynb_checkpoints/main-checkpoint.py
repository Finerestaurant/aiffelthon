# 2022-09-02 16:23 Seoul

# --- import dataset ---
from utils.dataloader import mel_dataset
from utils.losses import *
from torch.utils.data import DataLoader, random_split

# --- import model ---
from model.supervised_model import *
from model.Conv1d_model import *
from model.Conv2d_model import *

# --- import framework ---
import flax 
import flax.linen as nn
from flax.training import train_state
import jax
import numpy as np
import jax.numpy as jnp
import optax

import argparse
from tqdm import tqdm
import os
import wandb
import matplotlib.pyplot as plt


# --- define argparser ---
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type = int, default=16, help='batch_size')
parser.add_argument('--learning_rate', type = float, default=0.0001, help='learning_rate')
parser.add_argument('--epoch', type = int, default=5, help='Epoch')
parser.add_argument('--model_type', type = str, default=None, help='model_type')
parser.add_argument('--dilation', type = bool, default=False, help='dilation')
parser.add_argument('--linear_evaluation', type=bool, default=False, help='Linear_evaluation')
parser.add_argument('--num_workers', type=int, default=0, help='num_workers')

args = parser.parse_args()


# --- collate batch for dataloader ---
 
def collate_batch(batch):
    x_train = [x for x, _ in batch]
    y_train = [y for _, y in batch]                  
        
    return np.array(x_train), np.array(y_train)


# --- define init state ---

def init_state(model, x_shape, key, lr) -> train_state.TrainState:
    params = model.init({'params': key}, jnp.ones(x_shape), key)
    optimizer = optax.adam(learning_rate=lr)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        tx=optimizer,
        params=params)

def linear_init_state(model, x_shape, key, lr) -> train_state.TrainState:
    params = model.init({'params': key}, jnp.ones(x_shape))
    optimizer = optax.adam(learning_rate=lr)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        tx=optimizer,
        params=params)






# --- define train_step ---

@jax.jit
def train_step(state, x, z_rng):    
    def loss_fn(params):
        recon_x, mean, logvar = model.apply(params, x, z_rng)
        kld_loss = kl_divergence(mean, logvar).mean()
        mse_loss = ((recon_x - x)**2).mean()
        loss = mse_loss + kld_loss
        return loss
    
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    
    return state.apply_gradients(grads=grads), loss

@jax.jit
def linear_train_step(state, x, y):    
    def loss_fn(params):
        logits = linear_evaluation().apply(params, x)
        loss = jnp.mean(optax.softmax_cross_entropy(logits, y))
        return loss, logits
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == jnp.argmax(labels, -1))
    
    return state.apply_gradients(grads=grads), loss

# --- define eval step ---

@jax.jit
def eval_step(state, x, z_rng):
    
    recon_x, mean, logvar = model.apply(state.params, x, z_rng)
    kld_loss = kl_divergence(mean, logvar).mean()
    mse_loss = ((recon_x - x)**2).mean()
    loss = mse_loss + kld_loss
    
    return recon_x, loss, mse_loss, kld_loss

@jax.jit
def linear_eval_step(state, x, y):
    
    logits = linear_evaluation().apply(state.params, x)
    loss = jnp.mean(optax.softmax_cross_entropy(logits, y))
    accuracy = jnp.mean(jnp.argmax(logits, -1) == jnp.argmax(labels, -1))
    
    return loss, accuracy

if __name__ == "__main__":
    batch_size = args.batch_size
    lr = args.learning_rate
    dilation = args.dilation
    
    if args.model_type == 'Conv1d':
        model = Conv1d_VAE(dilation=args.dilation)
    elif args.model_type == 'Conv2d':
        model = Conv2d_VAE(dilation=args.dilation)
    else: 
        raise Exception('Input Correct model type. Conv1d, Conv2d.')
    
    rng = jax.random.PRNGKey(303)
    
    
    # ---Load dataset---
    dataset_dir = os.path.join(os.path.expanduser('~'),'dataset')            

    print("Loading dataset...")    
    data = mel_dataset(dataset_dir)
    print(f'Loaded data : {len(data)}')
    
    dataset_size = len(data)
    train_size = int(dataset_size * 0.8)
    test_size = dataset_size - train_size
    
    train_dataset, test_dataset = random_split(data, [train_size, test_size])

    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_batch)
    test_dataloader = DataLoader(test_dataset, batch_size=int(batch_size/4), shuffle=True, num_workers=0, collate_fn=collate_batch)
    
    print(f'batch_size = {args.batch_size}')
    print(f'learning rate = {args.learning_rate}')
    print(f'train_size = {train_size}')
    print(f'test_size = {test_size}')
    
    
    print('Data load complete!\n')
    print(nn.tabulate(model, rngs={'params': rng})(next(iter(train_dataloader))[0], rng))
    
#     # ---initializing model---
#     print("Initializing model....")
#     state = init_state(model, 
#                        next(iter(train_dataloader))[0].shape, 
#                        rng, 
#                        lr)
    
#     print("Initialize complete!!\n")
    
#     # ---train model---
    
#     wandb.init(
#     project=args.model_type,
#     entity='aiffelthon',
#     config = args
#     )
#     for i in range(args.epoch):
#         train_data = iter(train_dataloader)
#         test_data = iter(test_dataloader)
        
#         train_loss_mean = 0
#         test_loss_mean = 0
        
#         print(f'\nEpoch {i+1}')
        
#         for j in range(len(train_dataloader)):
#             rng, key = jax.random.split(rng)
#             x, y = next(train_data)
#             test_x, test_y = next(test_data)
            
# #             x = (x + 100) 
# #             test_x = (test_x + 100)
            
#             state, train_loss = train_step(state, x, rng)           
#             recon_x, test_loss, mse_loss, kld_loss = eval_step(state, test_x, rng)
#             wandb.log({'train_loss' : train_loss, 'test_loss' : test_loss, 'mse_loss':mse_loss,'kld_loss':kld_loss})
#             train_loss_mean += train_loss
#             test_loss_mean += test_loss
            
#             if j % 100 == 0:
                
#                 recon_x = recon_x.reshape(recon_x.shape[0], x.shape[1], x.shape[2])       
                
#                 fig1, ax1 = plt.subplots()
#                 im1 = ax1.imshow(recon_x[0], aspect='auto', origin='lower', interpolation='none')
#                 fig1.colorbar(im1)
#                 fig1.savefig('recon.png')


#                 fig2, ax2 = plt.subplots()
#                 im2 = ax2.imshow(test_x[0], aspect='auto', origin='lower', interpolation='none')
#                 fig2.colorbar(im2)
#                 fig2.savefig('x.png')
                
#                 wandb.log({'reconstruction' : [
#                             wandb.Image('recon.png')
#                             ], 
#                            'original image' : [
#                             wandb.Image('x.png')
#                             ]})
                
#             print(f'step : {j}/{len(train_dataloader)}, train_loss : {round(train_loss, 3)}, test_loss : {round(test_loss, 3)}', end='\r')

#         print(f'epoch {i+1} - average loss - train : {round(train_loss_mean/len(train_dataloader), 3)}, test : {round(test_loss_mean/len(test_dataloader), 3)}')
    
                      
                      
# wandb.finish()
