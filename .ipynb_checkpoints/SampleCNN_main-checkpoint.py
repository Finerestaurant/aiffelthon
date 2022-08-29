from dataloader import mel_dataset
from torch.utils.data import DataLoader
from model import SampleCNN

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


def init_state(model, shape, key, lr) -> train_state.TrainState:
    params = model.init({'params': key, 'dropout':key}, jnp.ones(shape))
    # Create the optimizer
    optimizer = optax.adam(learning_rate=lr)
    # Create a State
    return train_state.TrainState.create(
        apply_fn=model.apply,
        tx=optimizer,
        params=params)



@jax.jit
def train_step(state,
               inputs,
               labels,
               dropout_rng=None):
    
    def loss_fn(params):
        logits = SampleCNN().apply(
            params,
            inputs,
            rngs={"dropout": dropout_rng})
        
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=labels))
        return loss, logits
        
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == jnp.argmax(labels, -1))
    new_state = state.apply_gradients(grads=grads)
    

    return new_state, loss, accuracy
    


if __name__ == "__main__":
    batch_size = 128
    lr = 0.0001
    
    # ---Load dataset---
    print("Loading dataset...")
    dataset_dir = '/home/anthonypark6904/dataset'
    data = mel_dataset(dataset_dir)
    train_dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_batch)
    print(f'batch_size = {batch_size}')
    print(f'learning rate = {lr}')
    
    print('Data load complete!\n')

    # ---initializing model---
    model = SampleCNN()
    print("Initializing model....")
    state = init_state(model, next(iter(train_dataloader))[0].shape, jax.random.PRNGKey(353), lr)
    print("Initialize complete!!\n")
    rng = jax.random.PRNGKey(353)
    # ---train model---
    epoch = 50
    # checkpoint_dir = str(input('checkpoint dir : '))
    loss_list = []

    for i in range(epoch):
        train_data = iter(train_dataloader)
        loss_mean = 0
        accuarcy_mean = 0 
        print(f'\nEpoch {i+1}')
        
        for j in range(len(train_dataloader)):
            rng, key = jax.random.split(rng)
            x, y = next(train_data)
            state, loss, accuracy = train_step(state, x, y, dropout_rng=rng)
            loss_list.append(loss)
            loss_mean += loss
            accuarcy_mean += accuracy
            print(f'step : {j}/{len(train_dataloader)}, loss : {loss}, accuracy : {accuracy}', end='\r')
            # if j % 100 == 0:
            #     checkpoints.save_checkpoint(ckpt_dir=checkpoint_dir, target=state, step=state.step, overwrite=True)
            
        print(f'epoch {i+1} - average loss : {loss_mean/len(train_dataloader)} - accuracy : {accuarcy_mean/len(train_dataloader)}')
        




