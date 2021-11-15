#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
FilePath: /generative/train_distil.py
Desciption: Perform knowledge distillation for compressing the BERT model.
'''
import random
import sys
from functools import partial
import csv

import numpy as np
import pandas as pd
import torch
from textbrewer import DistillationConfig, GeneralDistiller, TrainingConfig
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.append('..')
from config import root_path, max_length, distill_log_path
from generative import config_distil
from generative.bert_model import BertConfig
from generative.optimizer import BERTAdam
from generative.seq2seq import Seq2SeqModel
from generative.tokenizer import Tokenizer, load_chinese_base_vocab
from utils.tools import create_logger, divide_parameters
import config


def read_corpus(data_path):
    df = pd.read_csv(data_path,
                     sep='\t',
                     header=None,
                     names=['src', 'tgt'],
                     quoting=csv.QUOTE_NONE
                     ).dropna()
    sents_src = []
    sents_tgt = []
    for index, row in df.iterrows():
        query = row['src']
        answer = row['tgt']
        if len(query) + len(answer) >= max_length:
            continue
        sents_src.append(query)
        sents_tgt.append(answer)
    return sents_src, sents_tgt


class SelfDataset(Dataset):
    """
    针对数据集，定义一个相关的取数据的方式
    """

    def __init__(self, path, max_length):
        # 一般init函数是加载所有数据
        super(SelfDataset, self).__init__()
        # 读原始数据
        self.sents_src, self.sents_tgt = read_corpus(path)
        self.word2idx = load_chinese_base_vocab()
        self.idx2word = {k: v for v, k in self.word2idx.items()}
        self.tokenizer = Tokenizer(self.word2idx)

        self.max_length = max_length

    def __getitem__(self, i):
        # 得到单个数据

        src = self.sents_src[i] if len(
            self.sents_src[i]
        ) < self.max_length else self.sents_src[i][:self.max_length]
        tgt = self.sents_tgt[i] if len(
            self.sents_tgt[i]
        ) < self.max_length else self.sents_tgt[i][:self.max_length]

        token_ids, token_type_ids = self.tokenizer.encode(src, tgt)
        output = {
            'token_ids': token_ids,
            'token_type_ids': token_type_ids,
        }
        return output

    def __len__(self):
        return len(self.sents_src)


def collate_fn(batch):
    """
    动态padding， batch为一部分sample
    """

    def padding(indice, max_length, pad_idx=0):
        """
        pad 函数
        注意 token type id 右侧pad是添加1而不是0，1表示属于句子B
        """
        pad_indice = [
            item + [pad_idx] * max(0, max_length - len(item))
            for item in indice
        ]
        return torch.tensor(pad_indice).to('cuda:0' if torch.cuda.is_available() else 'cpu')

    token_ids = [data['token_ids'] for data in batch]
    max_length = max([len(t) for t in token_ids])
    token_type_ids = [data['token_type_ids'] for data in batch]

    token_ids_padded = padding(token_ids, max_length)
    token_type_ids_padded = padding(token_type_ids, max_length)
    target_ids_padded = token_ids_padded[:, 1:].contiguous()

    return token_ids_padded, token_type_ids_padded, target_ids_padded


def evaluate(model, devloader, step, args):
    logger.info('start evaluating model')
    model.eval()
    logger.info('starting evaluating')
    total_loss = 0
    count = 0
    with torch.no_grad():
        for token_ids, token_type_ids, target_ids in tqdm(devloader, position=0, leave=True):
            token_ids = token_ids.to(args.device)
            token_type_ids = token_type_ids.to(args.device)
            target_ids = target_ids.to(args.device)

            predictions, loss, _, _, _ = model(token_ids,
                                               token_type_ids,
                                               labels=target_ids
                                               )

            # loss, accuracy = calculate_loss_and_accuracy(outputs, labels=input_ids, device=device)

            total_loss = total_loss + loss.mean()
            count += 1
    logger.info('evaluate step{},loss {}'.format(step, total_loss / count))
    logger.info('finishing evaluating')


def simple_adaptor(batch, model_outputs):
    return {'logits': model_outputs[2][-1],
            'attention': model_outputs[3],
            'losses': model_outputs[1],
            'hidden': model_outputs[2],
            'logits_mask': model_outputs[4],
            'inputs_mask': model_outputs[4]}


def main():
    # parse arguments
    config_distil.parse()
    args = config_distil.args
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    forward_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)
    args.forward_batch_size = forward_batch_size

    word2idx = load_chinese_base_vocab()

    # load bert config
    bert_config_T = BertConfig(vocab_size=len(word2idx))
    bert_config_S = BertConfig(vocab_size=len(word2idx),
                               num_hidden_layers=3)

    model_T = Seq2SeqModel(config=bert_config_T)
    model_S = Seq2SeqModel(config=bert_config_S)

    train = SelfDataset(config.generative_train, max_length=max_length)
    dev = SelfDataset(config.generative_dev, max_length=max_length)

    trainloader = DataLoader(train,
                             batch_size=args.forward_batch_size,
                             shuffle=True,
                             collate_fn=collate_fn)
    devloader = DataLoader(dev,
                           batch_size=args.forward_batch_size,
                           shuffle=True,
                           collate_fn=collate_fn)

    if args.tuned_checkpoint_T is not None:
        state_dict_T = torch.load(args.tuned_checkpoint_T, map_location='cpu')
        model_T.load_state_dict(state_dict_T)
        model_T.eval()
    else:
        assert args.do_predict is True

    if args.load_model_type == 'bert':
        assert args.init_checkpoint_S is not None
        # bert前三层初始化
        state_dict_S = torch.load(args.init_checkpoint_S, map_location='cpu')
        missing_keys, unexpected_keys = model_S.load_state_dict(state_dict_S, strict=False)
        logger.info(f'missing keys:{missing_keys}')
        logger.info(f'number unexpected keys:{len(unexpected_keys)}')


    model_T.to(args.device)
    model_S.to(args.device)

    num_train_steps = int(len(word2idx) / args.train_batch_size) * args.num_train_epochs

    if args.do_train:
        # parameters
        params = list(model_S.named_parameters())
        all_trainable_params = divide_parameters(params, lr=args.learning_rate)

        # logger.info("all_trainable_params: ", all_trainable_params)
        logger.info('Length of all_trainable_params: %d', len(all_trainable_params))

        optimizer = BERTAdam(all_trainable_params,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_steps,
                             schedule=args.schedule,
                             s_opt1=args.s_opt1,
                             s_opt2=args.s_opt2,
                             s_opt3=args.s_opt3)

        train_config = TrainingConfig(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            ckpt_frequency=args.ckpt_frequency,
            log_dir=args.output_dir,
            output_dir=args.output_dir,
            device=args.device)

        from matches import matches
        intermediate_matches = None
        if isinstance(args.matches, (list, tuple)):
            intermediate_matches = []
            for match in args.matches:
                intermediate_matches += matches[match]
        logger.info(f'intermediate_matches, {intermediate_matches}')

        distill_config = DistillationConfig(
            temperature=args.temperature,
            intermediate_matches=intermediate_matches,
            hard_label_weight=1,
            kd_loss_weight=1)

        distiller = GeneralDistiller(train_config=train_config,
                                     distill_config=distill_config,
                                     model_T=model_T,
                                     model_S=model_S,
                                     adaptor_S=simple_adaptor,
                                     adaptor_T=simple_adaptor)

        callback_func = partial(evaluate,
                                devloader=devloader,
                                args=args, )

        with distiller:
            distiller.train(optimizer,
                            scheduler=None,
                            dataloader=trainloader,
                            num_epochs=args.num_train_epochs,
                            callback=callback_func)


def load_model(model, pretrain_model_path):
    global logger
    checkpoint = torch.load(pretrain_model_path,
                            map_location=torch.device('cpu'))
    # 模型刚开始训练的时候, 需要载入预训练的BERT

    checkpoint = {
        k[5:]: v
        for k, v in checkpoint.items() if k[:4] == 'bert' and 'pooler' not in k
    }
    model.load_state_dict(checkpoint, strict=False)
    torch.cuda.empty_cache()
    logger.info('{} loaded!'.format(pretrain_model_path))


if __name__ == '__main__':
    logger = create_logger(distill_log_path)
    main()
