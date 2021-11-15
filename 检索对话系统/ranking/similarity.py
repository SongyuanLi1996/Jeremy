#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
FilePath: ranking/similarity.py
Desciption: Definition of manual features.
'''

import logging
import sys
import os

import jieba.posseg as pseg
import numpy as np
from gensim import corpora, models

sys.path.append('..')

from config import root_path
from retrieval.hnsw_faiss import wam
from ranking.bm25 import BM25

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


class TextSimilarity(object):
    def __init__(self):
        logging.info('load dictionary')
        self.dictionary = corpora.Dictionary.load(
            os.path.join(root_path, 'model/ranking/ranking.dict'))
        logging.info('load corpus')
        self.corpus = corpora.MmCorpus(
            os.path.join(root_path, 'model/ranking/ranking.mm'))
        logging.info('load tfidf')
        self.tfidf = models.TfidfModel.load(
            os.path.join(root_path, 'model/ranking/tfidf'))
        logging.info('load bm25')
        self.bm25 = BM25(do_train=False)
        logging.info('load word2vec')
        self.w2v_model = models.KeyedVectors.load(
            os.path.join(root_path, 'model/ranking/w2v'))
        logging.info('load fasttext')
        self.fasttext = models.FastText.load(
            os.path.join(root_path, 'model/ranking/fast'))
        self.stopwords = [x.strip() for x in open(os.path.join(root_path,
                                           'data/stopwords.txt')).readlines()]
        print('stopwords: ',self.stopwords[:50])

    def lcs(self, str_a, str_b):
        """Longest common substring

        Returns:
            float : The length of LCS divided by the length of
                the shorter one among two input strings.
        """
        lengths = [[0 for j in range(len(str_b) + 1)]
                   for i in range(len(str_a) + 1)]

        for i, x in enumerate(str_a):
            for j, y in enumerate(str_b):
                if x == y:
                    lengths[i + 1][j + 1] = lengths[i][j] + 1
                else:
                    lengths[i + 1][j + 1] = max(lengths[i + 1][j],
                                                lengths[i][j + 1])

        longestdist = lengths[len(str_a)][len(str_b)]
        ratio = longestdist / min(len(str_a), len(str_b))
        return ratio

    def editDistance(self, str1, str2):
        """Edit distance (add, delete, replace)
        Returns:
            ratio: Minimum edit distance divided by the length sum
                of two input strings.
        """
        m = len(str1)
        n = len(str2)
        lensum = float(m + n)
        d = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            d[i][0] = i
        for j in range(n + 1):
            d[0][j] = j

        for j in range(1, n + 1):
            for i in range(1, m + 1):
                if str1[i - 1] == str2[j - 1]:
                    d[i][j] = d[i - 1][j - 1]
                else:
                    d[i][j] = min(d[i - 1][j], d[i][j - 1],
                                  d[i - 1][j - 1]) + 1

        dist = d[-1][-1]
        ratio = (lensum - dist) / lensum
        return ratio

    def tokenize(self, str_a, drop_stop=False):
        """
        接受一个字符串作为参数，返回分词后的结果字符串(空格隔开)和集合类型
        """
        wordsa = pseg.cut(str_a)
        cuta = ''
        seta = set()
        for key in wordsa:
            if drop_stop and key.word in self.stopwords:
                continue
            else:
                cuta += key.word + ' '
                seta.add(key.word)

        return [cuta, seta]

    def JaccardSim(self, str_a, str_b):
        """
        Jaccard相似性系数
        计算sa和sb的相似度 len（sa & sb）/ len（sa | sb）
        """
        seta = self.tokenize(str_a)[1]
        setb = self.tokenize(str_b)[1]
        jaccard_sim = 1.0 * len(seta & setb) / len(seta | setb)
        return jaccard_sim

    @staticmethod
    def cos_sim(a, b):
        a = np.array(a)
        b = np.array(b)
        cos_sim = np.sum(a * b) / (np.sqrt(np.sum(a ** 2)) * np.sqrt(np.sum(b ** 2)))
        return cos_sim

    @staticmethod
    def eucl_sim(a, b):
        a = np.array(a)
        b = np.array(b)
        eucl_sim = 1 / (1 + np.sqrt((np.sum(a - b) ** 2)))
        return eucl_sim

    @staticmethod
    def pearson_sim(a, b):
        a = np.array(a)
        b = np.array(b)

        a = a - np.average(a)
        b = b - np.average(b)
        pearson_sim = np.sum(a * b) / (np.sqrt(np.sum(a ** 2)) * np.sqrt(np.sum(b ** 2)))
        return pearson_sim

    def tokenSimilarity(self, str_a, str_b, method='w2v', sim='cos'):
        """
        基于分词求相似度，默认使用cos_sim 余弦相似度,默认使用前20个最频繁词项进行计算
        method: w2v, tfidf, fasttext
        sim: cos, pearson, eucl
        """

        str_a = self.tokenize(str_a, drop_stop=True)[0]
        str_b = self.tokenize(str_b, drop_stop=True)[0]
        vec_a, vec_b, model = None, None, None
        if method == 'w2v':
            vec_a = wam(str_a, self.w2v_model)
            vec_b = wam(str_b, self.w2v_model)
            model = self.w2v_model
        elif method == 'fasttest':
            vec_a = wam(str_a, self.fasttext)
            vec_b = wam(str_b, self.fasttext)
            model = self.fasttext
        elif method == 'tfidf':
            vec_a = np.array(self.tfidf[self.dictionary.doc2bow(
                str_a.split())]).mean()
            vec_b = np.array(self.tfidf[self.dictionary.doc2bow(
                str_b.split())]).mean()
        else:
            raise NotImplementedError(f'method {method} do not implemented.')
        result = None
        if (vec_a is not None) and (vec_b is not None):
            if sim == 'cos':
                result = self.cos_sim(vec_a, vec_b)
            elif sim == 'eucl':
                result = self.eucl_sim(vec_a, vec_b)
            elif sim == 'pearson':
                result = self.pearson_sim(vec_a, vec_b)
            elif sim == 'wmd' and model:
                result = model.wmdistance(str_a, str_b)
            else:
                raise NotImplementedError(f'similarity {sim} do not implemented.')
        return result

    def generate_all(self, str1, str2):
        return {
            'lcs':
                self.lcs(str1, str2),
            'edit_dist':
                self.editDistance(str1, str2),
            'jaccard':
                self.JaccardSim(str1, str2),
            'bm25':
                self.bm25.bm_25(str1, str2),
            'w2v_cos':
                self.tokenSimilarity(str1, str2, method='w2v', sim='cos'),
            'w2v_eucl':
                self.tokenSimilarity(str1, str2, method='w2v', sim='eucl'),
            'w2v_pearson':
                self.tokenSimilarity(str1, str2, method='w2v', sim='pearson'),
            'w2v_wmd':
                self.tokenSimilarity(str1, str2, method='w2v', sim='wmd'),
            'fast_cos':
                self.tokenSimilarity(str1, str2, method='fasttest', sim='cos'),
            'fast_eucl':
                self.tokenSimilarity(str1, str2, method='fasttest', sim='eucl'),
            'fast_pearson':
                self.tokenSimilarity(str1, str2, method='fasttest', sim='pearson'),
            'fast_wmd':
                self.tokenSimilarity(str1, str2, method='fasttest', sim='wmd'),
            'tfidf_cos':
                self.tokenSimilarity(str1, str2, method='tfidf', sim='cos'),
            'tfidf_eucl':
                self.tokenSimilarity(str1, str2, method='tfidf', sim='eucl'),
            'tfidf_pearson':
                self.tokenSimilarity(str1, str2, method='tfidf', sim='pearson')
        }
