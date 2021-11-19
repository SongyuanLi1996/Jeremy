#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import torch

# General
root_path = os.path.abspath(os.path.dirname(__file__))
is_cuda = True
device = torch.device('cuda') if is_cuda else torch.device('cpu')

# Data
max_vocab_size = 20000
embed_file = None  # use pre-trained embeddings
train_data_path = os.path.join(root_path, 'files/train.txt')
val_data_path = os.path.join(root_path, 'files/dev.txt')
test_data_path = os.path.join(root_path, 'files/test.txt')
stop_word_file = os.path.join(root_path, 'files/stopwords.txt')
max_src_len = 200  # exclusive of special tokens such as EOS
max_tgt_len = 80  # exclusive of special tokens such as EOS
truncate_src = True
truncate_tgt = True
min_dec_steps = 30
max_dec_steps = 80
enc_rnn_dropout = 0.5
enc_attn = True
dec_attn = True
dec_in_dropout = 0
dec_rnn_dropout = 0
dec_out_dropout = 0
data_folder = os.path.join(root_path, 'files')
img_vecs = os.path.join(root_path, 'files/img_vecs.txt')
sample_path = os.path.join(root_path, 'files/samples.txt')

# Training
hidden_size = 512
dec_hidden_size = 512
embed_size = 256
trunc_norm_init_std = 1e-4
eps = 1e-31
learning_rate = 1e-4
lr_decay = 0.0
initial_accumulator_value = 0.1
epochs = 8
batch_size = 8
pointer = True
coverage = False
fine_tune = False
img_feat = True
max_grad_norm = 2.0
is_cuda = True
DEVICE = torch.device("cuda" if is_cuda else "cpu")
LAMBDA = 1
patience = 2
img_vec_dim = 1000

model_name = 'baseline'
if pointer:
    model_name = 'pgn'
    if img_feat:
        model_name += '_multimodal'
    if coverage:
        model_name += '_cov'
    if fine_tune:
        model_for_ft = model_name.replace('_cov', '')
        model_name += '_ft'
        

encoder_save_name = os.path.join(root_path, 'saved_model/' + model_name + '/encoder.pt')
decoder_save_name = os.path.join(root_path, 'saved_model/' + model_name + '/decoder.pt')
attention_save_name = os.path.join(root_path, 'saved_model/' + model_name + '/attention.pt')
reduce_state_save_name = os.path.join(root_path, 'saved_model/' + model_name + '/reduce_state.pt')
losses_path = os.path.join(root_path, 'saved_model/' + model_name + '/val_losses.pkl')
log_path = os.path.join(root_path, 'runs/' + model_name)


# Beam search
beam_size = 3
alpha = 0.2
beta = 0.2
gamma = 0.6

#Inference and evaluation
rouge_path = os.path.join(root_path, 'files/rouge_result.txt')