#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Data definitions, GAE computation for Gorge Chase PPO.
峡谷追猎 PPO 数据类定义与 GAE 计算。
"""

import numpy as np
from common_python.utils.common_func import create_cls, attached
from agent_ppo.conf.conf import Config


# ObsData: feature=1899D vector, legal_action=16D mask
ObsData = create_cls("ObsData", feature=None, legal_action=None)

# ActData: action(0~15), d_action, prob(18D 三头扁平概率), value(3D 三头价值)
# V5：prob 维度改为 18 = 2 (flash_gate) + 8 (move_dir) + 8 (flash_dir)
ActData = create_cls("ActData", action=None, d_action=None, prob=None, value=None)

# V5：动作概率总维度 = 2 + 8 + 8 = 18
ACTION_PROB_DIM = 2 + 8 + 8

# SampleData: single-frame sample with int dims
SampleData = create_cls(
    "SampleData",
    obs=Config.DIM_OF_OBSERVATION,
    legal_action=Config.ACTION_NUM,
    act=1,
    reward=Config.VALUE_NUM,         # V5: 3
    reward_sum=Config.VALUE_NUM,     # V5: 3
    done=1,
    value=Config.VALUE_NUM,          # V5: 3
    next_value=Config.VALUE_NUM,     # V5: 3
    advantage=Config.VALUE_NUM,      # V5: 3 (三路独立 GAE)
    prob=ACTION_PROB_DIM,            # V5: 18 (gate+move+flash_dir 三头扁平)
)


def sample_process(list_sample_data):
    """Fill next_value and compute GAE advantage (V5 三路独立 GAE).

    填充 next_value 并对三路 reward 各自跑一遍 GAE。
    """
    for i in range(len(list_sample_data) - 1):
        list_sample_data[i].next_value = list_sample_data[i + 1].value

    # 终局处理：truncated（活到 max_step 但 done 是 truncated）时用最后一帧 value 做 bootstrap，
    # 避免把"活到时间结束"当成 next_value=0 处理，从而低估长寿局。
    # terminated（被怪杀死）时 next_value 保持 0。
    last = list_sample_data[-1]
    _done_val = float(last.done) if hasattr(last.done, "__len__") else float(last.done)
    if _done_val < 0.5:
        # truncated 分支
        last.next_value = last.value

    _calc_gae(list_sample_data)
    return list_sample_data


def _calc_gae(list_sample_data):
    """Compute GAE for each of 3 value heads independently.

    对 survive / collect / explore 三路分别计算 GAE。
    reward/value/next_value/advantage/reward_sum 都是 (3,) 维 numpy 数组。
    """
    gamma = Config.GAMMA
    lamda = Config.LAMDA
    value_num = Config.VALUE_NUM

    gae = np.zeros(value_num, dtype=np.float32)
    for sample in reversed(list_sample_data):
        # 三路独立：向量化的 delta / gae 更新，每一路自己算各自的 advantage
        delta = -sample.value + sample.reward + gamma * sample.next_value
        gae = gae * gamma * lamda + delta
        sample.advantage = gae.copy()
        sample.reward_sum = gae + sample.value
