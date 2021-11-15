#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
FilePath: /generative/predict.py
Desciption: Predict using BERT seq2seq model.
'''
import sys
import os
import time
import torch

sys.path.append('..')
from config import is_cuda, root_path
from generative.bert_model import BertConfig
from generative.seq2seq import Seq2SeqModel
from generative.tokenizer import load_chinese_base_vocab


class bertSeq2Seq(object):
    def __init__(self, model_path, is_cuda, is_distill=False):
        self.word2idx = load_chinese_base_vocab()
        if is_distill:
            self.config = BertConfig(len(self.word2idx),num_hidden_layers=3)
        else:
            self.config = BertConfig(len(self.word2idx))
        self.bert_seq2seq = Seq2SeqModel(self.config)
        self.is_cuda = is_cuda
        if is_cuda:
            device = torch.device('cuda')
            self.bert_seq2seq.load_state_dict(torch.load(model_path))
            self.bert_seq2seq.to(device)
        else:
            checkpoint = torch.load(model_path,
                                    map_location=torch.device('cpu'))
            self.bert_seq2seq.load_state_dict(checkpoint)
        # 加载state dict参数
        self.bert_seq2seq.eval()

    def generate(self, text, k=20):
        result = self.bert_seq2seq.generate(text,
                                            beam_size=k,
                                            is_cuda=self.is_cuda)
        return result


if __name__ == '__main__':
    text = '我要找人工'
    k=20
    print('number beams', k)
    print('用户：',text)
    t1 = time.time()
    bs = bertSeq2Seq(os.path.join(root_path, 'model/generative/bert.model.epoch.29'), is_cuda)
    print('BERT12',bs.generate(text, k=k),f'time used {time.time()-t1:2f}s')
    t1 = time.time()
    distilled = bertSeq2Seq(os.path.join(root_path, 'model/generative/gs283492.pkl'), is_cuda, is_distill=True)
    print('BERT3',distilled.generate(text, k=k),f'time used {time.time()-t1:2f}s')
