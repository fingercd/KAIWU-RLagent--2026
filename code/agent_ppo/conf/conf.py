#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Configuration for Gorge Chase PPO.
峡谷追猎 PPO 配置（精简版：只把 code 里被验证有用的设置移植回来）。
"""


class Config:

    # Feature dimensions / 特征维度（共 1899 维，V5：压力特征 + 闪现战略特征）
    FEATURES = [
        4,    # hero_self
        10,   # monsters
        12,   # treasures_top3
        8,    # buffs_top2
        1764, # multichannel_map (4ch×21×21)
        8,    # escape_depth
        8,    # monster_aware_escape
        3,    # local_topology
        8,    # monster_memory
        8,    # flash_through_wall
        10,   # dead_end_analysis
        16,   # legal_action
        3,    # progress
        12,   # exploration
        8,    # flash_to_safe
        2,    # trap_status
        8,    # flash_dist_gain
        2,    # anti_loop
        3,    # pressure_feat（保留作为输入特征喂给 critic，但不再乘到奖励上）
        2,    # flash_strategic_feat
    ]
    FEATURE_SPLIT_SHAPE = FEATURES
    FEATURE_LEN = sum(FEATURE_SPLIT_SHAPE)
    DIM_OF_OBSERVATION = FEATURE_LEN

    ACTION_NUM = 16
    VALUE_NUM = 3

    # ---- PPO 超参 ----
    GAMMA = 0.995
    LAMDA = 0.95
    INIT_LEARNING_RATE_START = 0.0002
    BETA_START = 0.003
    BETA_END = 0.0003
    BETA_DECAY_STEPS = 30000
    CLIP_PARAM = 0.2          # policy ratio clip
    VF_COEF = 1.0
    GRAD_CLIP_RANGE = 1.0     # 原 0.5 → 1.0，避免 Transformer 梯度被卡死

    # 三组 advantage 合成权重（先各自标准化再加权）
    ADV_WEIGHT_SURVIVE = 1.0
    ADV_WEIGHT_COLLECT = 1.0
    ADV_WEIGHT_EXPLORE = 1.0

    # ---- Reward clip：每组各自上下限 ----
    # 旧版统一 ±1 会把宝箱拾取等稀疏大奖压死，改成 ±5 让信号有空间
    REWARD_CLIP_SURVIVE = 5.0
    REWARD_CLIP_COLLECT = 5.0
    REWARD_CLIP_EXPLORE = 5.0

    # ---- Value clip：单独配置（不再共用 CLIP_PARAM=0.2，太小学不动）----
    VALUE_CLIP_RANGE = 2.0

    # ---- 加速期（与 train_env_conf.toml monster_speedup 对齐）----
    SPEEDUP_THRESHOLD = 300
    R_POST_SPEEDUP_ALIVE_BASE = 0.01   # 加速后每步小生存奖，告诉模型"活过 300 步是好事"

    # ---- 探索奖励强化（移植自 code 版，让模型更积极探索全图）----
    # 加速期 coverage 硬约束：coverage_rate < 阈值时按缺口线性扣分，强迫模型走出绕圈窝
    LATE_DRIFT_HARD_COEF = -0.40            # 系数（绝对值越大惩罚越重）
    LATE_DRIFT_COVERAGE_THRESHOLD = 0.20    # coverage_rate 阈值（200 步窗口内唯一格子比例）
    # 朝地图边界移动奖励（加速后 + coverage 低时生效，把模型从中心地推向边缘扩大探索面）
    BORDER_APPROACH_REWARD = 0.10
    BORDER_APPROACH_COV_THRESHOLD = 0.40
    # 首次到达新格的"好奇心"附加奖（与 EXPLORE_MOVE_REWARD 合并，单步最大 0.14）
    R_CURIOSITY = 0.10
