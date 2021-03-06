#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import os
import sys

import numpy as np
from torch import optim
from torch.utils.data import DataLoader
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from tensorboardX import SummaryWriter

sys.path.append('..')

from model import PGN
import config
from evaluate import evaluate
from dataset import collate_fn, SamplesDataset


def train(dataset, val_dataset, v, start_epoch=0):
    """Train the model, evaluate it and store it.

    Args:
        dataset (dataset.PairDataset): The training dataset.
        val_dataset (dataset.PairDataset): The evaluation dataset.
        v (vocab.Vocab): The vocabulary built from the training dataset.
        start_epoch (int, optional): The starting epoch number. Defaults to 0.
    """
    torch.autograd.set_detect_anomaly(True)
    DEVICE = torch.device("cuda" if config.is_cuda else "cpu")

    model = PGN(v)
    model.load_model()
    model.to(DEVICE)
    if config.fine_tune:
        # In fine-tuning mode, we fix the weights of all parameters except attention.wc.
        print('Fine-tuning mode.')
        for name, params in model.named_parameters():
            if name != 'attention.wc.weight':
                params.requires_grad=False    
    # forward
    print("loading data")
    train_data = dataset
    val_data = val_dataset

    print("initializing optimizer")

    # Define the optimizer.
    optimizer = optim.Adam(model.parameters(),
                           lr=config.learning_rate)
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  collate_fn=collate_fn)

    val_loss = np.inf
    if (os.path.exists(config.losses_path)):
        with open(config.losses_path, 'rb') as f:
            val_losses = pickle.load(f)

#     torch.cuda.empty_cache()
    writer = SummaryWriter(config.log_path)
    with tqdm(total=config.epochs) as epoch_progress:
        early_stopping_count = 0
        for epoch in range(start_epoch, config.epochs):
            batch_losses = []  # Get loss of each batch.
            num_batches = len(train_dataloader)
            with tqdm(total=num_batches//100) as batch_progress:
                for batch, data in enumerate(tqdm(train_dataloader)):
                    x, y, x_len, y_len, oov, len_oovs, img_vec = data
                    assert not np.any(np.isnan(x.numpy()))
                    if config.is_cuda:  # Training with GPUs.
                        x = x.to(DEVICE)
                        y = y.to(DEVICE)
                        x_len = x_len.to(DEVICE)
                        len_oovs = len_oovs.to(DEVICE)
                        img_vec = img_vec.to(DEVICE)

                    model.train()  # Sets the module in training mode.
                    optimizer.zero_grad()  # Clear gradients.
                    # Calculate loss.
#                     print('x: ', x)
                    loss = model(x, y, len_oovs, img_vec, batch=batch, num_batches=num_batches)
                    batch_losses.append(loss.item())
                    loss.backward()  # Backpropagation.

                    # Do gradient clipping to prevent gradient explosion.
                    clip_grad_norm_(model.encoder.parameters(),
                                    config.max_grad_norm)
                    clip_grad_norm_(model.decoder.parameters(),
                                    config.max_grad_norm)
                    clip_grad_norm_(model.attention.parameters(),
                                    config.max_grad_norm)
                    clip_grad_norm_(model.reduce_state.parameters(),
                                    config.max_grad_norm)
                    optimizer.step()  # Update weights.

                    # Output and record epoch loss every 100 batches.
                    if (batch % 100) == 0:
                        batch_progress.set_description(f'Epoch {epoch}')
                        batch_progress.set_postfix(Batch=batch,
                                                   Loss=loss.item())
                        batch_progress.update()
                        # Write loss for tensorboard.
                        writer.add_scalar(f'Average loss for epoch {epoch}',
                                          np.mean(batch_losses),
                                          global_step=batch)
            # Calculate average loss over all batches in an epoch.
            epoch_loss = np.mean(batch_losses)

            epoch_progress.set_description(f'Epoch {epoch}')
            epoch_progress.set_postfix(Loss=epoch_loss)
            epoch_progress.update()

            avg_val_loss = evaluate(model, val_data, epoch)

            print('training loss:{}'.format(epoch_loss),
                  'validation loss:{}'.format(avg_val_loss))

            # Update minimum evaluating loss.
            if (avg_val_loss < val_loss):
                torch.save(model.encoder, config.encoder_save_name)
                torch.save(model.decoder, config.decoder_save_name)
                torch.save(model.attention, config.attention_save_name)
                torch.save(model.reduce_state, config.reduce_state_save_name)
                val_loss = avg_val_loss
            else:
                early_stopping_count += 1
                if early_stopping_count >= config.patience:
                    print(f'Validation loss did not decrease for {config.patience} epochs, stop training.')
                    break
            with open(config.losses_path, 'wb') as f:
                pickle.dump(val_loss, f)

    writer.close()


if __name__ == "__main__":
    DEVICE = torch.device('cuda') if config.is_cuda else torch.device('cpu')
    dataset = SamplesDataset(config.train_data_path)
    val_dataset = SamplesDataset(config.val_data_path)

    vocab = dataset.vocab

    train(dataset, val_dataset, vocab, start_epoch=0)
