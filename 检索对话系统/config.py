#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FilePath: /config.py
Desciption: 配置文件。

"""

import torch
import os

root_path = os.path.abspath(os.path.dirname(__file__))
dataset_path = os.path.join(
    os.path.dirname(root_path),
    'dataset/对话生成数据集/e8d1386f-5a00-49f9-b56b-4b2e1bfa0554')

train_raw = os.path.join(dataset_path, 'file/chat.txt')
dev_raw = os.path.join(dataset_path, 'file/Develop.txt')
test_raw = os.path.join(dataset_path, 'file/Test.txt')
ware_path = os.path.join(dataset_path, 'file/ware.txt')

sep = '[SEP]'

# main
train_path = os.path.join(root_path, 'data/train_no_blank.csv')
dev_path = os.path.join(root_path, 'data/dev.csv')
test_path = os.path.join(root_path, 'data/test.csv')

# intention
business_train = os.path.join(root_path, 'data/intention/business.train')
business_test = os.path.join(root_path, 'data/intention/business.test')
keyword_path = os.path.join(root_path, 'data/intention/key_word.txt')

# ranking dataset path
ranking_train = os.path.join(root_path, 'data/ranking/train.tsv')
ranking_dev = os.path.join(root_path, 'data/ranking/dev.tsv')
ranking_test = os.path.join(root_path, 'data/ranking/test.tsv')

# generative dataset path
generative_train = os.path.join(root_path, 'data/generative/train.tsv')
generative_dev = os.path.join(root_path, 'data/generative/dev.tsv')
generative_test = os.path.join(root_path, 'data/generative/test.tsv')

# intention
# fasttext
ft_path = os.path.join(root_path, 'model/intention/fastext')

#  Retrival
# Embedding
w2v_path = os.path.join(root_path, 'model/retrieval/word2vec')

# HNSW parameters
ef_construction = 3000  # ef_construction defines a construction time/accuracy trade-off
M = 64  # M defines tha maximum number of outgoing connections in the graph
hnsw_path = os.path.join(root_path, 'model/retrieval/hnsw_index')
hnsw_path_hnswlib = os.path.join(root_path,
                                 'model/retrieval/hnsw_index_hnswlib')

# 通用配置
is_cuda = True
if is_cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

max_sequence_length = 512
# 用于训练的会话数量，（总共有200万条原始对话数据,平均一个会话约有10条对话）
num_train_samples = 10000

base_chinese_bert_vocab = os.path.join(root_path, 'lib/bert-base-chinese/bert-base-chinese-vocab.txt')

max_length = 512
batch_size = 10
lr = 2e-5
bert_chinese_model_path = os.path.join(root_path, 'lib/bert-base-chinese/bert-base-chinese-pytorch_model.bin')
log_path = os.path.join(root_path, 'log/log.txt')
max_grad_norm = 2
gradient_accumulation = 1
distill_log_path = os.path.join(root_path, 'log/distill.txt')
