#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FilePath: /ranking/bm25.py
Desciption: Train a bm25 model.
"""
import jieba
import jieba.posseg as pseg
import numpy as np
import pandas as pd
import joblib
import sys

sys.path.append('..')
from config import root_path
import math
from collections import Counter
import os
import csv


class BM25(object):
    def __init__(self,
                 do_train=True,
                 save_path=os.path.join(root_path, 'model/ranking/')):
        if do_train:
            self.data = pd.read_csv(os.path.join(root_path,
                                                 'data/ranking/train.tsv'),
                                    sep='\t',
                                    header=None,
                                    quoting=csv.QUOTE_NONE,
                                    names=['question1', 'question2', 'target'])
            self.idf, self.avgdl = self.get_idf()
            self.saver(save_path)
        else:
            self.stopwords = self.load_stop_word()
            self.load(save_path)

    def load_stop_word(self):
        """
        Returns:
            list[string]: list of stopwords.
        """
        stop_words = os.path.join(root_path, 'data/stopwords.txt')
        stopwords = open(stop_words, 'r', encoding='utf8').readlines()
        stopwords = [w.strip() for w in stopwords]
        return stopwords

    def tf(self, word, count):
        """
        term frequency
        Args:
            word (str): word to compute tf score.
            count (dict): words to number of words count.
        Returns:
            float: term frequency - tf score.
        """
        return count[word] / sum(count.values())

    def n_containing(self, word, count_list):
        """
        Args:
            word (string): a word in string.
            count_list (list[string]): list of document strings.
        Returns:
            int: The number of documents include the word.
        """
        return sum(1 for count in count_list if word in count)

    def cal_idf(self, word, count_list):
        """
        Args:
            word (string): a word in string.
            count_list (list[string]): The corpus, list of document strings.
        Returns:
            float: The idf score of the word related to the corpus.
        """
        return math.log(
            len(count_list)) / (1 + self.n_containing(word, count_list))

    def get_idf(self):
        """
        Returns:
            Tuple(Counter, float): the counter maps word to idf score, and
            the average document length.
        """
        self.data['question2'] = self.data['question2'].apply(
            lambda x: ' '.join(jieba.cut(x)))
        idf = Counter(
            [y for x in self.data['question2'].tolist() for y in x.split()])
        idf = {
            k: self.cal_idf(k, self.data['question2'].tolist())
            for k, v in idf.items()
        }
        avgdl = np.array(
            [len(x.split()) for x in self.data['question2'].tolist()]).mean()
        return idf, avgdl

    def saver(self, save_path):
        """
        Save bm25 model
        Args:
            save_path (str): folder path to save files.
        """
        joblib.dump(self.idf, save_path + 'bm25_idf.bin')
        joblib.dump(self.avgdl, save_path + 'bm25_avgdl.bin')

    def load(self, save_path):
        """
        load bm25 model
        Args:
            save_path (str): folder path to load files.
        """
        self.idf = joblib.load(save_path + 'bm25_idf.bin')
        self.avgdl = joblib.load(save_path + 'bm25_avgdl.bin')

    def bm_25(self, q, d, k1=1.2, k2=200, b=0.75):
        """
        Compute bm25 score.
        Args:
            q (str): query text.
            d (str): document content.
            k1 (float, optional):
                control the importance of frequency in the query.
            k2 (float, optional):
                control the importance of frequency in the document.
            b (float, optional):
                factor to control relative document length.
        """
        stop_flag = ['x', 'c', 'u', 'd', 'p', 't', 'uj', 'm', 'f', 'r']
        words = pseg.cut(q)  # 切分查询式
        fi = {}
        qfi = {}
        for word, flag in words:
            if flag not in stop_flag and word not in self.stopwords:
                fi[word] = d.count(word)
                qfi[word] = q.count(word)
        # 计算K值
        K = k1 * (1 - b + b * (len(d) / self.avgdl))
        ri = {}
        for key in fi:
            ri[key] = fi[key] * (k1 + 1) * qfi[key] * (k2 + 1) / (
                (fi[key] + K) * (qfi[key] + k2))  # 计算R

        score = 0
        for key in ri:
            score += self.idf.get(key, 20.0) * ri[key]
        return score


if __name__ == '__main__':
    bm25 = BM25(do_train=True)
