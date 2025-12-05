# coding: utf-8
#
# Copyright 2020 Yequan Wang
# Author: Yequan Wang (tshwangyequan@gmail.com)
#
# Basic data operation

# 다음 import를 통해서 파이썬 하위 버전에서 상위버전 문법을 사용 수 있음
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence #sequence들이 길이가 다르기 때문에 padding을 해서 길이를 맞추기 위한 import


def load_data_from_memory(lines, max_length_text, config_label):
    
    # 하나의 JSON 레코드 처리
    def tackle_one(input: dict): # parameter input:dict는 input이 dictionary 형태여야함
        # init label
        label = {k: 0 for k in config_label['detail'].keys()} # config_label 딕셔너리 안에 key 값마다 value를 0으로 설정하여 label에 dictionary 형태로 저장
        mask = np.zeros(config_label['n_task']) # 0으로 가득찬 array 생성 --> 아마도 task의 개수만큼 0으로 채워진 배열을 생성
        for k, v in input['label'].items(): # input으로부터 key값과 value값을 불러옴
            # k = int(k) # 수정 20250718
            if v is None: # value 값이 존재하지 않으면
                continue
            if k in label: # key값이 label 안에 있으면
                mask[int(k)] = 1 # 0이었던 원소를 1로 변경 -- 마스크는 일반적으로 loss 계산할 때 쓰는데, 마스크가 1인 경우에만 loss에 포함시킴
                label[k] = v # label에 key값에 해당하는 값을 value값으로 변경
            else:
                pass
        dict_inst = { # token, label, mask 형태로 
            'tokens': input['text'][:max_length_text], # max_length_text 길이까지만 텍스트를 자른다 
            'labels': label,
            'mask': mask
        }
        return dict_inst

    data = []
    for line in lines:
        try:
            line = json.loads(line.strip()) 
            # json 문자열을 python 객체로 변환 (딕셔너리 형태)
            # line.strip()은 문자열 양쪽의 공백을 제거한다.
        except json.decoder.JSONDecodeError:
            pass
        else:
            data_one = {key: tackle_one(value) for key, value in line.items()}
            data.append(data_one)
    return data

# 모델을 평가하기 위한 입력 데이터 전처리 함수
def build_data_for_eva(list_text, max_length_text, config_label, sampled_num: int = -1):
    data = load_data_from_memory(list_text, max_length_text, config_label)
    sampled_num = len(list_text) if sampled_num == -1 else sampled_num # 샘플 개수가 지정되었으면 지정된 값을 사용하고, 그렇지 않다면 전체 데이터 길이 만큼 사용
    data_with_label = [build_single_instance(tmp, config_label) for tmp in data]
    # data_with_label = {
    #     key: [build_single_instance_kernel(tmp) for tmp in value] for key, value in data.items()
    # }
    return data_with_label[:sampled_num] # 정해진 샘플 개수만큼 잘라서 return

#하나의 샘플을 PyTorch 모델에 바로 넣을 수 있도록 텐서 형태로 변환하는 함수
def build_single_instance_kernel(item):
    item_data = {
        'bert_text': item['tokens'],
        'len_sen': len(item['tokens']),
        'labels': {key: torch.tensor(value, dtype=torch.long) for key, value in item['labels'].items()},
        'mask': torch.tensor(item['mask'], dtype=torch.long),
    }
    return item_data

# 여러 개 샘플을 각각 텐서 포맷으로 변환하는 함수
def build_single_instance(input, config_label):
    data_with_label = {
        key: build_single_instance_kernel(value) for key, value in input.items()
    }
    return data_with_label


def text_collate_fn(data):
    """
    batched_data = {
        # 'sentences': pad_sequence([tmp['sentences'] for tmp in data], batch_first=True, padding_value=0),
        'bert_text': [tmp['bert_text'] for tmp in data],
        'len_sen': [tmp['len_sen'] for tmp in data],
        'labels': torch.stack([tmp['labels'] for tmp in data]),
        # 'labels': pad_sequence([tmp['labels'] for tmp in data], batch_first=True, padding_value=0),
    }
    """
    batched_data = dict() 
    assert len(data) > 0, 'Please check data_utils.py' # data의 길이가 0이하 즉 데이터가 비어있으면 에러메시지

    # 여러 개의 샘플을 하나의 배치로 묶는 함수
    def collate_fn_one(input):
        batched_data_one = dict()
        for name in input[0].keys():
            if name != 'labels':
                batched_data_one[name] = [tmp[name] for tmp in input]
                if name == 'mask':
                    batched_data_one[name] = pad_sequence(batched_data_one[name]) # mask의 길이가 다를 경우 길이를 맞춰주기 위해서 패딩을 적용함
            else:
                batched_data_one[name] = {
                    tmp: torch.stack([_tmp['labels'][tmp] for _tmp in input]) for tmp in input[0]['labels'].keys() # torch.stack 새로운 차원으로 주어진 텐서를 붙임
                }
        return batched_data_one

    dict_data = {k: [tmp[k] for tmp in data] for k, v in data[0].items()}

    for k, v in dict_data.items():
        batched_data[k] = collate_fn_one(v)
    return batched_data
