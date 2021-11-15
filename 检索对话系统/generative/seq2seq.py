#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
FilePath: /generative/seq2seq.py
Desciption: Using the BERT model for seq2seq task.
'''
import sys

sys.path.append('..')
import torch
import torch.nn as nn

from config import max_length

from .bert_model import BertConfig, BertLMPredictionHead, BertModel
from .tokenizer import Tokenizer, load_chinese_base_vocab


class Seq2SeqModel(nn.Module):
    """
    """

    def __init__(self, config: BertConfig):
        super(Seq2SeqModel, self).__init__()
        self.bert = BertModel(config)
        self.hidden_dim = config.hidden_size
        self.vocab_size = config.vocab_size
        self.decoder = BertLMPredictionHead(
            config, self.bert.embeddings.word_embeddings.weight)
        # 加载字典和分词器
        self.word2ix = load_chinese_base_vocab()
        self.tokenizer = Tokenizer(self.word2ix)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def compute_loss(self, predictions, labels, target_mask):
        """
        target_mask : 句子a部分和pad部分全为0， 而句子b部分为1
        """
        predictions = predictions.view(-1, self.vocab_size)
        labels = labels.view(-1)
        target_mask = target_mask.view(-1).float()
        loss = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
        return (loss(predictions, labels) * target_mask).sum() / target_mask.sum()

    def forward(self,
                input_tensor,
                token_type_id,
                labels=None,
                position_enc=None,
                output_attentions=True,
                is_cuda=True):

        input_tensor = input_tensor.to(self.device)
        token_type_id = token_type_id.to(self.device)
        if position_enc is not None:
            position_enc = position_enc.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
        input_shape = input_tensor.shape
        batch_size = input_shape[0]
        seq_len = input_shape[1]

        ## 构建特殊的mask
        ones = torch.ones((1, 1, seq_len, seq_len), dtype=torch.float32, device=self.device)
        a_mask = ones.tril()  # triangle lower
        s_ex12 = token_type_id.unsqueeze(1).unsqueeze(2).float()  # [batch size, 1, 1, len_seq]
        s_ex13 = token_type_id.unsqueeze(1).unsqueeze(3).float()  # [bs, 1, len seq, 1]
        a_mask = (1.0 - s_ex12) * (1.0 - s_ex13) + s_ex13 * a_mask
        # first seq self attention + ones.tril() 去掉左上部分的 1

        enc_layers, _, all_attention_matrices, attention_mask = self.bert(input_tensor,
                                                          position_ids=position_enc,
                                                          token_type_ids=token_type_id,
                                                          attention_mask=a_mask,
                                                          output_all_encoded_layers=True,
                                                          output_attentions=output_attentions)
        squence_out = enc_layers[-1]  ## 取出来最后一层输出 (batch, seq_len, 768)

        predictions = self.decoder(squence_out)

        if labels is not None:
            predictions = predictions[:, :-1].contiguous()
            target_mask = token_type_id[:, 1:].contiguous()
            loss = self.compute_loss(predictions, labels, target_mask)
            return predictions, loss, enc_layers, all_attention_matrices, attention_mask
        else:
            return predictions

    def generate(self, text, out_max_length=50, beam_size=1, is_cuda=False):
        # 对 一个 句子生成相应的结果
        # 通过输出最大长度得到输入的最大长度，这里问题不大，如果超过最大长度会进行截断

        self.out_max_length = out_max_length
        input_max_length = max_length - out_max_length
        # print(text)
        try:
            token_ids, token_type_ids = self.tokenizer.encode(text, max_length=input_max_length)
        except:
            # 可能是transformer的tokenizer
            tokenizer_out = self.tokenizer.encode_plus(text, max_length=input_max_length, truncation=True)
            token_ids = tokenizer_out['input_ids']
            token_type_ids = tokenizer_out['token_type_ids']
        token_ids = torch.tensor(token_ids, device=self.device).view(1, -1)
        token_type_ids = torch.tensor(token_type_ids, device=self.device).view(1, -1)
        print('start beam search')
        out_puts_ids = self.beam_search(token_ids, token_type_ids, self.word2ix, beam_size=beam_size,
                                        device=self.device)
        # 解码 得到相应输出
        out_puts_ids = out_puts_ids.detach().cpu().numpy().tolist()
        return self.tokenizer.decode(out_puts_ids)

    def beam_search(self,
                    token_ids,
                    token_type_ids,
                    word2ix,
                    beam_size=1,
                    device='cpu',
                    alpha=0.5):
        """
        beam-search操作
        """
        sep_id = word2ix['[SEP]']

        # 用来保存输出序列
        output_ids = torch.empty(1, 0, device=device, dtype=torch.long)
        # 用来保存累计得分

        with torch.no_grad():
            output_scores = torch.zeros(token_ids.shape[0], device=device)
            for step in range(self.out_max_length):
                if step == 0:
                    scores = self.forward(token_ids, token_type_ids)
                    # 重复beam-size次 输入ids
                    token_ids = token_ids.view(1, -1).repeat(beam_size, 1)
                    token_type_ids = token_type_ids.view(1, -1).repeat(beam_size, 1)
                else:
                    scores = self.forward(new_input_ids, new_token_type_ids)

                logit_score = torch.log_softmax(scores[:, -1], dim=-1)

                logit_score = output_scores.view(-1, 1) + logit_score  # 累计得分
                ## 取topk的时候我们是展平了然后再去调用topk函数
                # 展平
                logit_score = logit_score.view(-1)
                hype_score, hype_pos = torch.topk(logit_score, beam_size)
                indice1 = (hype_pos // scores.shape[-1])  # 行索引
                indice2 = (hype_pos % scores.shape[-1]).long().reshape(-1, 1)  # 列索引

                # 更新得分
                output_scores = hype_score
                output_ids = torch.cat([output_ids[indice1], indice2], dim=1).long()
                new_input_ids = torch.cat([token_ids, output_ids], dim=1)
                new_token_type_ids = torch.cat([token_type_ids, torch.ones_like(output_ids)], dim=1)

                end_counts = (output_ids == sep_id).sum(1)  # 统计出现的end标记
                best_one = output_scores.argmax()
                if end_counts[best_one] == 1:
                    # 说明出现终止了～
                    return output_ids[best_one][:-1]
                else:
                    # 保留未完成部分
                    flag = (end_counts < 1)  # 标记未完成序列
                    if not flag.all():  # 如果有已完成的
                        token_ids = token_ids[flag]
                        token_type_ids = token_type_ids[flag]
                        new_input_ids = new_input_ids[flag]
                        new_token_type_ids = new_token_type_ids[flag]
                        output_ids = output_ids[flag]  # 扔掉已完成序列
                        output_scores = output_scores[flag]  # 扔掉已完成序列
                        end_counts = end_counts[flag]  # 扔掉已完成end计数
                        beam_size = flag.sum()  # topk相应变化
        return output_ids[output_scores.argmax().item()]
