#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
FilePath: /ranking/ranker.py
Desciption: Generating features and train a LightGBM ranker.
'''

import sys
import os
import csv
import logging

sys.path.append('..')
from sklearn.model_selection import train_test_split
from ranking.similarity import TextSimilarity
from ranking.matchnn import MatchingNN
from config import root_path
import lightgbm as lgb
import pandas as pd
import joblib
from tqdm import tqdm


tqdm.pandas()

# Parameters for lightGBM
params = {'boosting_type': 'gbdt',
          'max_depth': 10,
          'objective': 'binary',
          'nthread': -1,
          'num_leaves': 64,
          'learning_rate': 0.01,
          'max_bin': 128,
          'subsample_for_bin': 200,
          'subsample': 0.5,
          'subsample_freq': 5,
          'colsample_bytree': 0.8,
          'reg_alpha': 5,
          'reg_lambda': 10,
          'min_split_gain': 0.5,
          'min_child_weight': 1,
          'min_child_samples': 5,
          'scale_pos_weight': 1,
          'max_position': 20,
          'group': 'name:groupId',
          'metric': 'auc'}


class RANK(object):
    def __init__(self,
                 do_train=True,
                 model_path=os.path.join(root_path, 'model/ranking/lightgbm')):
        self.ts = TextSimilarity()
        self.matchingNN = MatchingNN()
        self.stopwords = open(os.path.join(root_path,
                                           'data/stopwords.txt')).readlines()
        if do_train:
            logging.info('Training mode')
            self.train = pd.read_csv(
                os.path.join(root_path, 'data/ranking/train.tsv'),
                sep='\t',
                header=None,
                nrows=10000,
                quoting=csv.QUOTE_NONE,
                names=['question1', 'question2', 'label'])

            self.data = self.generate_feature(self.train)
            self.columns = [
                i for i in self.train.columns if 'question' not in i]

            self.trainer()
            self.save(model_path)
        else:
            logging.info('Predicting mode')
            self.test = pd.read_csv(
                os.path.join(root_path, 'data/ranking/test.tsv'),
                sep='\t',
                header=None,
                quoting=csv.QUOTE_NONE,
                names=['question1', 'question2', 'label'])
            self.testdata = self.generate_feature(self.test)
            self.gbm = joblib.load(model_path)
            self.predict(self.testdata)

    def generate_feature(self, data):
        """
        Args:
            data (pd.DataFrame): data with question1, question2 column
        Returns:
            data (pd.DataFrame): data with question1, question2 and features.
        """
        logging.info('Generating manual features.')
        data = pd.concat([data, pd.DataFrame.from_records(
            data.progress_apply(lambda row: self.ts.generate_all(
                row['question1'],
                row['question2']),
                axis=1))], axis=1)
        logging.info('Generating deep-matching features.')
        # 将bert模型的知识转移到lightbgm上。
        data['matching_score'] = data.progress_apply(
            lambda row: self.matchingNN.predict(
                row['question1'],
                row['question2'])[1], axis=1)
        return data

    def trainer(self):
        logging.info('Training lightgbm model.')
        self.gbm = lgb.LGBMRanker(metric='auc')
        columns = [i for i in self.data.columns if i not in [
            'question1',
            'question2',
            'label']]
        X_train, X_test, y_train, y_test = train_test_split(
            self.data[columns],
            self.data['label'],
            test_size=0.2,
            random_state=42)
        query_train = [X_train.shape[0]]
        query_val = [X_test.shape[0]]
        self.gbm.fit(X_train,
                     y_train,
                     group=query_train,
                     eval_set=[(X_test, y_test)],
                     eval_group=[query_val],
                     eval_at=[5, 10, 20],
                     early_stopping_rounds=500)

    def save(self, model_path):
        logging.info('Saving lightgbm model.')
        joblib.dump(self.gbm, model_path)

    def predict(self, data: pd.DataFrame):
        """Doing prediction.

        Args:
            data (pd.DataFrame): the output of self.generate_feature

        Returns:
            result[list]: The scores of all query-candidate pairs.
        """
        columns = [i for i in data.columns if i not in [
            'question1',
            'question2',
            'label']]

        result = self.gbm.predict(data[columns])
        return result


if __name__ == '__main__':
    rank = RANK(do_train=True)
