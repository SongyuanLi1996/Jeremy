#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
FilePath: /retrieval/hnsw_hnswlib.py
Desciption: 使用hnswlib训练hnsw模型。
'''
from gensim.models import KeyedVectors
import pandas as pd
import numpy as np
import hnswlib
import logging
import sys
import os
import time

sys.path.append('..')

import config
from utils.preprocessor import clean

logging.basicConfig(format='%(levelname)s - %(asctime)s: %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)


def wam(sentence, w2v_model):
    '''
    @description: 通过word average model 生成句向量
    @param {type}
    sentence: 以空格分割的句子
    w2v_model: word2vec模型
    @return: Sentence embeded vector.
    '''
    arr = []
    for s in clean(sentence).split():
        if s not in w2v_model.wv.vocab.keys():
            pass
        else:
            arr.append(w2v_model.wv.get_vector(s))
    return np.mean(np.array(arr), axis=0).reshape(1, -1)


class HNSW(object):
    def __init__(self,
                 w2v_path,
                 data_path=config.train_path,
                 ef=config.ef_construction,
                 M=config.M,
                 model_path=config.hnsw_path_hnswlib):
        self.w2v_model = KeyedVectors.load(w2v_path)
        self.data = self.data_load(data_path)
        self.dim = self.w2v_model.vector_size
        if model_path and os.path.exists(model_path):
            self.hnsw = self.load_hnsw(model_path)
        else:
            self.hnsw = self.build_hnsw(model_path, ef=ef, m=M)

    def data_load(self, data_path):
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

    def build_hnsw(self, to_file, ef=2000, m=64):
        """
        @description: 训练hnsw模型
        @param {type}
        to_file： 模型保存目录
        @return:
        """
        logging.info('Building hnsw index.')
        vecs = np.stack(self.data['custom_vec'].values).reshape(-1, self.dim)
        num_elements = vecs.shape[0]
        p = hnswlib.Index(space='l2', dim=self.dim)
        p.init_index(max_elements=num_elements, ef_construction=ef, M=m)

        p.set_ef(ef)
        p.add_items(vecs, num_threads=-1)
        labels, distances = p.knn_query(vecs[:10000], k=1)
        print('labels: ', labels)
        print('distances: ', distances)
        logging.info('Recall:{}'.format(
            np.mean(labels.reshape(-1) == np.arange(len(vecs[:10000])))))
        p.save_index(to_file)
        return p

    def load_hnsw(self, model_path):
        """
        @description: 加载训练好的hnsw模型
        @param {type}
        model_path： 模型保存的目录
        @return: hnsw 模型
        """
        logging.info(f'Loading hnsw index from {model_path}.')

        p = hnswlib.Index(space='l2', dim=self.dim)
        p.load_index(model_path)
        return p

    def search(self, text, k=5):
        """
        @description: 通过hnsw 检索
        @param {type}
        text: 检索句子
        k: 检索返回的数量
        @return:
        """
        logging.info(f'Searching for {text}.')

        text_vec = wam(clean(text), self.w2v_model)
        i, distances = self.hnsw.knn_query(text_vec, k=k)
        result = pd.concat(
            (self.data.iloc[i[0]]['custom'].reset_index(),
             self.data.iloc[i[0]]['assistance'].reset_index(drop=True),
             pd.DataFrame(distances.reshape(-1, 1), columns=['q_distance'])),
            axis=1)
        return result

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
        i, p = self.hnsw.knn_query(vecs, 1)
        t1 = time.time()
        missing_rate = (i == -1).sum() / float(nq)
        recall_at_1 = (i == ground_truth).sum() / float(nq)
        print(f'\t {(t1 - t0) * 1000 / nq:.3f} ms per query, '
              f'Recall@1 {recall_at_1:.4f}, '
              f'missing_rate {missing_rate:.4f}')


if __name__ == '__main__':
    hnsw = HNSW(
        w2v_path=config.w2v_path,
        data_path=config.train_path,
        ef=config.ef_construction,
        M=config.M,
        model_path=config.hnsw_path_hnswlib,
    )
    test = '我要转人工'
    print(hnsw.search(test, k=5))
    eval_vecs = np.stack(hnsw.data['custom_vec'].values).reshape(-1, 300)
    eval_vecs.astype('float32')
    hnsw.evaluate(eval_vecs[:10000], ground_truth=np.arange(10000))
