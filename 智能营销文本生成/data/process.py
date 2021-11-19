#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os

import json

sys.path.append('..')
from data_utils import write_samples, partition
import config

samples = set()

json_file = '../files/clothing_1w_1.json'


with open(json_file, 'r', encoding='utf8') as file:
    jsf = json.load(file)

for key, val in jsf.items():
    imgid = key  # Get image id.
    source = val['src'].replace('。', '').replace('，', '')  # Get source.
    cate = val['cate'] # Get category.
    targets = val['tgt'] # Get targets.

    # Create a sample for every target(reference).
    for target in targets:
        sample = source + '\t' + target + '\t' + cate + '\t' + imgid
        samples.add(sample)

write_path = config.sample_path

print('write_path: ', write_path)
write_samples(samples, write_path)
partition(samples)
#
# with open('../files/clothing_1w_1.json') as f:
#     data = json.load(f)
#     samples = set()
#     for key, value in data.items():
#         tgt = ''.join(value['tgt'])
#         sample = value['src']+'<sep>'+tgt
#         samples.add(sample)
# write_samples(list(samples), '../files/sample.txt')
# partition(samples)
