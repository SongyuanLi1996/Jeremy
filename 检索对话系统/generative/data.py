#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
FilePath: /generative/data.py
Desciption: Data processing for the generative module.
'''
import os
import sys

sys.path.append('..')

import config
from utils.preprocessor import split_dataset
import pandas as pd


def gen_generative_dataset(*args):
    columns = ['starter', 'response']
    df_total = pd.DataFrame(columns=['custom', 'assistance'])
    for path in args:
        print(f'add dataset {path}')
        df_temp = pd.read_csv(path,header=0)[['custom', 'assistance']]
        df_total = pd.concat([df_total, df_temp],
                             axis=0,
                             ignore_index=True)
    df_total.columns = columns
    df_train, df_dev, df_test = split_dataset(df_total)

    df_train.to_csv(config.generative_train,sep='\t',index=None,header=None)
    df_dev.to_csv(config.generative_dev,sep='\t',index=None,header=None)
    df_test.to_csv(config.generative_test,sep='\t',index=None,header=None)


if __name__ == '__main__':
    datasets_path = [config.train_path, config.dev_path, config.test_path]
    gen_generative_dataset(*datasets_path)
