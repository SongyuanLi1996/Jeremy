#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
FilePath: /utils/preprocessor.py
Desciption: 对数据进行预处理。 清洗， 转换为问答pair， 并保存。
'''

import logging
import re
import sys
import os

sys.path.append('..')
import random
from config import dev_raw, root_path, sep, test_raw, train_raw
import argparse
import pandas as pd
import config

logging.basicConfig(format='%(levelname)s - %(asctime)s: %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)


def filter_content(sentence):
    """
    特殊字段有：
    1. #E-s[数字x] #E-2[数字x] 等一系列数字—— 表情
    2. [ORDERID_10187709] —— 订单号
    3. [数字x] —— 数字
    4. https://item.jd.com/5898522.html —— 网址
    5. [地址x] —— 地址
    6. [链接x] —— 链接
    7. [金额x] —— 金额
    8. [日期x] —— 日期
    9. [时间x] —— 时间
    10. [站点x] —— 站点
    11. [组织机构x] ——组织机构
    12. [电话x] —— 电话
    13. [姓名x] —— 人名
    对于表情，做法是直接删除。其他用希腊符号替换。
    """
    sentence = re.sub(
        r'#E\-[\w]*(抱拳|傲慢|得意|蛋糕|呕吐|闭嘴|礼物|yaoping|柠檬|流泪|怒火|撇嘴|太阳|咒骂|糗|猪猪|足球|磕头|大兵|电话|灯泡|飞鸟|奋斗|高兴|击打|饥饿|咖啡|口罩|骷髅|可乐|疯狂|白眼|阴险|叹气|奸笑|发呆|害羞|飞吻|怒火|悲伤|胜利|生病|弱|可怜|咖啡|酷酷|眩晕|流泪|发抖|难过|右哼哼|惊恐|悲伤|犯困|愤怒|凋谢|哈欠|拥抱|抓狂|鄙视|时间|啤酒|勾引|左哼哼|月亮|偷笑|震惊|惊讶|跳跳|瞌睡|可爱|衰样|好|憨笑|水果|色色|黑线|微笑|流汗|握手|心碎|问号|大哭|亲亲|抠鼻|拜拜|鬼脸|香吻|米饭|花朵|尴尬|擦汗|安慰|委屈|调皮|爱心|我一定尽力为您解答的哦|很棒|鼓掌)+',
        'α', sep.join(sentence))
    sentence = re.sub(r'#E\-[\w]+\[数字x]', 'α', sentence)
    sentence = re.sub(r'\[ORDERID_[\d]+]', '[订单x]', sentence)
    sentence = re.sub(r'\[数字x]', 'γ', sentence)
    sentence = re.sub(r'\[链接x]', 'ε', sentence)
    sentence = re.sub(r'\[表情]', 'α', sentence)
    sentence = re.sub('<sep>', sep, sentence)
    sentence = re.sub('<SEP>', sep, sentence)
    sentence = re.sub(
        r'(http|ftp|https):\/\/[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?',
        'ε', sentence)
    sentence = re.sub(r'(http|ftp|https):\/\/ε', 'ε', sentence)
    sentence = re.sub(r'[\d]+.*[\d]+', 'γ', sentence)
    sentence = re.sub(r'【收到不支持的消息类型，暂无法显示】', ' ', sentence)

    sentence = re.sub(r'#E\-[s]*(ν|γ|π|ζ|ρ|α|ε)*', 'α', sentence)
    sentence = re.sub('α', ' ', sentence)
    sentence = re.sub('ε', '[链接x]', sentence)
    sentence = re.sub('γ', '[数字x]', sentence)

    return sentence


def read_file(path, is_train=False):
    """
    @description: 读取文件， 并将原始数据中多次输入合并为一句。
    @param {type}
    path: 数据文件所在目录
    is_train： 是否为训练数据集
    @return:list  包含session_id, role, content
    """
    chat = []

    with open(path, 'r') as f:
        if is_train:
            next(f)  # only chat.txt has header
        tmp = []
        sessions = set()
        session_id, custom_id, is_assistance, content = '', '', '', []
        for lines in f:
            if len(sessions
                   ) > config.num_train_samples:  # 50k sessions at most.
                break
            line = lines.strip().replace(' ', '').split('\t')
            if len(line) < 5:  # Filtering short samples.
                continue
            if is_train:
                session_id_in_doc, custom_id_in_doc, is_assistance_in_doc = \
                    line[0], line[1], line[2]
            else:
                session_id_in_doc, custom_id_in_doc, is_assistance_in_doc = \
                    line[2], line[1], line[3]
            sessions.add(session_id_in_doc)
            if session_id != session_id_in_doc and session_id != '':
                fc = filter_content(content)
                if fc != '':
                    tmp.append([
                        session_id, 'custom'
                        if str(is_assistance) == '0' else 'assistance', fc
                    ])
                    content = []
                chat.extend(tmp)
                tmp = []
                session_id, custom_id = session_id_in_doc, custom_id_in_doc
            else:
                if is_assistance != is_assistance_in_doc and \
                        is_assistance != '':
                    content = filter_content(content)
                    is_assistance = 'custom' if str(
                        is_assistance) == '0' else 'assistance'
                    if content != '':
                        tmp.append([session_id, is_assistance, content])
                    is_assistance = is_assistance_in_doc
                    content = [line[-1]]
                else:
                    content.append(line[-1])
                    is_assistance = is_assistance_in_doc
                    session_id, _ = session_id_in_doc, custom_id_in_doc
    if content != '':
        tmp.append([
            session_id,
            'custom' if str(is_assistance) == '0' else 'assistance',
            filter_content(content)
        ])
    chat.extend(tmp)
    return chat


def clean(sent, sep='['):
    """
    @description: 过滤无用符号， 并对[SEP] 等分割符号， 假如前后空格，避免影响分词结果
    @param {type}
    sent: 句子
    sep: 分隔符是以< or [ 开头
    @return: string 清洗后的句子
    """
    sent = re.sub(r"[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+",
                  '', sent)
    i = 0
    tmp = []
    while i < len(sent):
        if sent[i] != sep:
            tmp.append(sent[i])
            i += 1
        else:
            tmp.append(sent[i:i + 5])
            i += 5
    return ' '.join(tmp)


def generate_data(filepath, save=True, to_file=None, pair=False):
    """
    @description: 将read_file 的结果进行转化， 问答pair, 或者将一次会话内容合并一起
    @param {type}
    file_path， 原始数据路径
    save, 是否保存
    to_file， 保存的文件名， 会根据文件名判断是否为训练集
    pair: 是否生成问答pair的结果
    @return: 处理后可作为模型输入的csv数据集。
    """

    data = read_file(filepath, 'train' in to_file)
    data = pd.DataFrame(data, columns=['session_id', 'role', 'content'])
    if 'train' in to_file:
        data = data[(data['content'].str.len() <= 128)
                    & (data['content'].str.len() > 1)].reset_index(drop=True)
    data = data.reset_index()
    data['index'] = data['index'].apply(lambda x: x - 1 if x % 2 == 1 else x)
    data = data.pivot_table(index=['index', 'session_id'],
                            columns='role',
                            values='content',
                            aggfunc='first').reset_index()
    data = data[['session_id', 'custom',
                 'assistance']].dropna().reset_index(drop=True)

    if save:
        data.to_csv('{}.csv'.format(to_file), index=False)
    return data


def process_atec_dataset(file):
    df = pd.read_csv(
        file,
        sep='\t',
        header=None,
        index_col=0,
        names=['question1', 'question2', 'label'],
    )
    df['label'] = df['label'].apply(
        lambda x: int(x) if isinstance(x, str) and len(x) == 1 else x
    )
    return df


def process_a2_dataset(dataset_folder='../ranking_datasets'):
    datasets = [
        x for x in os.listdir(dataset_folder) if x[-4:] in ['.csv', '.txt']
    ]
    total_dataset = pd.DataFrame(columns=['question1', 'question2', 'label'])
    for dataset in datasets:
        dataset_path = os.path.join(dataset_folder, dataset)
        if dataset.startswith('atec'):
            df_atec = process_atec_dataset(dataset_path)
            total_dataset = pd.concat([total_dataset, df_atec],
                                      axis=0,
                                      ignore_index=True)

        elif dataset.startswith('task3'):
            df_task3 = pd.read_csv(
                dataset_path,
                sep='\t',
                header=None,
                names=['question1', 'question2', 'label'],
            )
            df_task3['label'] = df_task3['label'].apply(
                lambda x: int(x) if isinstance(x, str) and len(x) == 1 else x
            )
            total_dataset = pd.concat([total_dataset, df_task3],
                                      axis=0,
                                      ignore_index=True)
        else:
            raise FileExistsError(f'unexpected dataset {dataset}')
    total_dataset = total_dataset[total_dataset['label'].apply(
        lambda x: isinstance(x, int))]
    return total_dataset


def split_dataset(dataset, split_ratio=None):
    if split_ratio is None:
        split_ratio = [0.8, 0.1, 0.1]
    assert isinstance(split_ratio, list), 'split_ratio should be list'
    assert sum(split_ratio) == 1, 'sum of split ratio should be 1'
    num_samples = len(dataset)
    idx = [i for i in range(num_samples)]
    random.shuffle(idx)
    split_idx_1 = int(split_ratio[0] * num_samples)
    split_idx_2 = int((1 - split_ratio[-1]) * num_samples)
    train_idx = idx[:split_idx_1]
    valid_idx = idx[split_idx_1:split_idx_2]
    test_idx = idx[split_idx_2:]
    print('number of train sample:\t', len(train_idx))
    print('number of train sample:\t', len(valid_idx))
    print('number of test sample:\t', len(test_idx))
    return dataset.iloc[train_idx], dataset.iloc[valid_idx], dataset.iloc[test_idx]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='A2')
    args = parser.parse_args()
    if args.task == 'A1':
        # generate dataset for Assignment 1
        dev = generate_data(dev_raw,
                            save=True,
                            to_file=os.path.join(root_path, 'data/dev'),
                            pair=True)
        logging.info('Dev set created.')
        test = generate_data(test_raw,
                             save=True,
                             to_file=os.path.join(root_path, 'data/test'),
                             pair=True)
        logging.info('test set created.')
        data = generate_data(train_raw,
                             save=True,
                             to_file=os.path.join(root_path,
                                                  'data/train_no_blank'),
                             pair=True)
        logging.info('training set created.')
    elif args.task == 'A2':
        total_dataset = process_a2_dataset()
        train, dev, test = split_dataset(total_dataset)
        train.to_csv(config.ranking_train,
                     sep='\t',
                     header=None,
                     index=None)
        dev.to_csv(config.ranking_dev,
                   sep='\t',
                   header=None,
                   index=None)
        test.to_csv(config.ranking_test,
                    sep='\t',
                    header=None,
                    index=None)
    else:
        raise ValueError(f"task {args.task} not in ['A1','A2']")
