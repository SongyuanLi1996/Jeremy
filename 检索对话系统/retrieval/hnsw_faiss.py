#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
FilePath: /retrieval/hnsw_faiss.py
Desciption: 使用Faiss训练hnsw模型。
'''

import logging
import sys
import os
import time

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
import faiss

sys.path.append('..')

from utils.preprocessor import clean
import config

logging.basicConfig(format='%(levelname)s - %(asctime)s: %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)


def wam(sentence, w2v_model):
    """
    @description: 通过word average model 生成句向量
    @param {type}
    sentence: 以空格分割的句子
    w2v_model: word2vec模型
    @return: The sentence vector.
    """
    arr = []
    for s in clean(sentence).split():
        if s not in w2v_model.wv.vocab.keys():
            pass
        else:
            arr.append(w2v_model.wv.get_vector(s))
    return np.mean(np.array(arr), axis=0).reshape(1, -1)


class HNSW(object):
    def __init__(
        self,
        w2v_path,
        ef=config.ef_construction,
        M=config.M,
        model_path=None,
        data_path=config.train_path,
    ):
        """
        Args:
            w2v_path (str): saved gensim word2vec model path.
            ef (int, optional): maximum size of dynamic list for HNSW, higher is more
                accurate and slower to construct.
            M (int, optional): maximum number of outgoing connections in the HNSW graph.
            model_path (str optional): HNSW model path to load or save.
            data_path (str optional): raw training data path.
        """
        self.w2v_model = KeyedVectors.load(w2v_path)
        self.data = self.load_data(data_path)
        if model_path and os.path.exists(model_path):
            self.index = self.load_hnsw(model_path)
        elif data_path:
            self.index = self.build_hnsw(model_path, ef=ef, m=M)
        else:
            logging.error('No existing model and no building data provided.')

    def load_data(self, data_path):
        """
        @description: 读取数据，并生成句向量
        @param {type}
        data_path：问答pair数据所在路径
        @return: 包含句向量的dataframe
        """
        data = pd.read_csv(data_path)
        data['custom_vec'] = data['custom'].apply(
            lambda x: wam(x, self.w2v_model))
        data['custom_vec'] = data['custom_vec'].apply(
            lambda x: x[0][0] if x.shape[1] != 300 else x)
        data = data.dropna()
        return data

    def evaluate(self, vecs, ground_truth):
        """
        @description: evaluate HNSW based on recall@1, missing rate and query time.
        @param {np.array} vecs: The vectors to evaluate. (num_samples, dim)
        @param {np.array} vecs: The ground_truth of evaluated vectors. (num_samples,)

        @return {type} None
        """

        logging.info('Evaluating.')
        nq, d = vecs.shape
        t0 = time.time()
        p, i = self.index.search(vecs, 1)
        t1 = time.time()
        missing_rate = (i == -1).sum() / float(nq)
        recall_at_1 = (i == ground_truth).sum() / float(nq)
        print(f'\t {(t1 - t0) * 1000 / nq:.3f} ms per query, '
              f'Recall@1 {recall_at_1:.4f}, '
              f'missing_rate {missing_rate:.4f}')

    def build_hnsw(self, to_file, ef=2000, m=64):
        """
        @description: 训练hnsw模型
        @param {type}
        to_file： 模型保存目录
        @return:
        """
        logging.info('Building hnsw index.')
        vecs = np.stack(self.data['custom_vec'].values).reshape(-1, 300)
        vecs = vecs.astype('float32')  # vec (5120217, 300)

        dim = self.w2v_model.vector_size

        index = faiss.IndexHNSWFlat(dim, m)
        index.hnsw.efConstruction = ef
        index.verbose = True
        index.add(vecs)
        faiss.write_index(index, to_file)

        return index

    def load_hnsw(self, model_path):
        """
        @description: 加载训练好的hnsw模型
        @param {type}
        model_path： 模型保存的目录
        @return: hnsw 模型
        """
        logging.info(f'Loading hnsw index from {model_path}.')
        hnsw = faiss.read_index(model_path)
        return hnsw

    def search(self, text, k=5):
        '''
        @description: 通过hnsw 检索
        @param {type}
        text: 检索句子
        k: 检索返回的数量
        @return: DataFrame contianing the customer input, assistance response
                and the distance to the query.
        '''
        logging.info(f'Searching for {text}.')
        test_vec = wam(clean(text), self.w2v_model)
        D, i = self.index.search(test_vec, k)
        result = pd.concat(
            (self.data.iloc[i[0]]['custom'].reset_index(),
             self.data.iloc[i[0]]['assistance'].reset_index(drop=True),
             pd.DataFrame(D.reshape(-1, 1), columns=['q_distance'])),
            axis=1)
        return result


if __name__ == '__main__':
    hnsw = HNSW(config.w2v_path, config.ef_construction, config.M,
                config.hnsw_path, config.train_path)
    test = '我要转人工'
    print(hnsw.search(test, k=5))
    eval_vecs = np.stack(hnsw.data['custom_vec'].values).reshape(-1, 300)
    eval_vecs.astype('float32')
    hnsw.evaluate(eval_vecs[:10000], ground_truth=np.arange(10000))
