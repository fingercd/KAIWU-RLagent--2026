#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Feature preprocessor and reward design for Gorge Chase PPO.
峡谷追猎 PPO 特征预处理与奖励设计。

========== 总体设计（V5：三头架构）==========
本文件的唯一职责：把环境原始观测 env_obs 转换为三样东西：
  1. feature (1899维) — 喂给神经网络的特征向量（V5：新增压力+闪现战略特征）
  2. legal_action (16维) — 合法动作掩码
  3. reward (3维)      — 本步三组即时奖励 [survive, collect, explore]

========== 特征向量布局（1894 维）==========
分段 | 维度 | 内容 | 回答什么决策问题
-----+------+------+------------------
hero_feat           |    4 | 英雄位置xy + 闪现CD + buff剩余时间       | "我现在是什么状态"
monster_feats       |   10 | 2只怪物各5维(视野/位置/速度/距离)         | "危险从哪来"（实体分支）
treasure_top3_feat  |   12 | 最近3个宝箱各4维(存在/距离/方向)          | "哪个资源值得追"（实体分支）
buff_top2_feat      |    8 | 最近2个buff各4维(存在/距离/方向)          | "加速道具在哪"（实体分支）
multichannel_map    | 1764 | 4通道21×21地图 (Ch0=通行性 Ch1=怪物 Ch2=宝箱 Ch3=buff) | "地形+实体空间关系"（FPN CNN）
escape_depth        |    8 | 8个方向各能跑多远(归一化)                  | "哪个方向有退路"
monster_escape      |    8 | 8个方向的怪物感知安全分                    | "考虑怪物后哪个方向最安全"
topology_feat       |    3 | 开阔率 + 可走方向数 + 死角标记             | "这里是不是死角"
memory_feat         |    8 | 2只怪物的短期记忆(有效性/衰减/逃离方向)    | "怪物消失前在哪个方向"
flash_through       |    8 | 8个方向的闪现穿墙机会评分                  | "闪现能不能穿过这面墙"
dead_end_info       |   10 | 死胡同得分+8方向出口标记+被困度（BFS版）    | "我是不是在死胡同里"
legal_action        |   16 | 16个动作的合法掩码                         | "哪些动作可以用"
progress_feat       |    3 | 步数进度 + 是否加速 + 加速倒计时           | "游戏到了什么阶段"
exploration_feat    |   12 | 位移(1)+8方向陌生度(8)+资源安全分(2)+buff进度(1) | "哪里没去过、资源值不值得追"
flash_to_safe       |    8 | 8方向闪现落点安全分（>0=安全，0=落地即死） | "闪往哪里不会死"
trap_status         |    2 | is_trapped + can_flash_escape            | "被困了吗、能闪出去吗"
flash_dist_gain     |    8 | 8方向闪现净距离改善量[-1,1]                | "朝这方向闪值不值"
anti_loop           |    2 | is_new_cell + coverage_rate              | "当前格是新的吗、在绕圈吗"

========== 奖励设计总览（V5 三组压力加权版）==========
奖励分为三组 + 独立项。每组各自乘"压力加权系数"后累加成三路 reward，
返回 [r_survive_sum, r_collect_sum, r_explore_sum] 喂给三头 critic。

【survive 组】× w_survive (= 1.0 + 1.5 * pressure，范围 1.0~2.5)
  r1 基础生存 / r2 怪物距离 / r7 逃跑深度 / r8 安全分 / r9 死角 / r10 安全方向 /
  r_memory 记忆逃离 / r_dead_end 死胡同 / r_flash_wall 穿墙闪 / r_flash_effect 闪现效果 /
  r_corridor 开阔度 / r_encircle 包夹 / r_speedup_buffer 加速缓冲 / r_second_monster 第二怪

【collect 组】× w_collect (= 1.0 - 0.7 * pressure，范围 0.3~1.0)
  r3 靠近宝箱 / r4 拾取宝箱 / r5 拾取buff / r_buff_approach 靠近buff / r_easy_treasure

【explore 组】× w_explore (= 1.0 - 0.6 * pressure，范围 0.4~1.0)
  r6 原地 / r_explore 探索 / r_wall_collision 撞墙 / r_repeat 重复 /
  r_anti_repeat 预反重复 / r_drift 漂移率

【独立项】不受任何权重调制，直接累加到 survive 组（强信号保留）
  r_flash_escape_trap 被困闪现脱险 / r_flash_cross_monster 跨怪闪现

压力分数 pressure = clip(0.4*shrink_rate + 0.3*sandwich + 0.2*danger_level + 0.1*escape_loss, 0, 1)
"""

from collections import deque

import numpy as np

from agent_ppo.conf.conf import Config

# ============================================================
#                      全局常量
# ============================================================

# --- 地图与单位尺度 ---
MAP_SIZE = 128.0            # 地图尺寸 128×128 格
MAX_MONSTER_SPEED = 5.0     # 怪物最大速度（用于归一化）
MAX_DIST_BUCKET = 5.0       # 距离分桶最大值（环境协议用）
MAX_FLASH_CD = 2000.0       # 闪现最大冷却步数（用于归一化）
MAX_BUFF_DURATION = 50.0    # 加速buff最大持续步数（用于归一化）

# --- 物件类型常量（与环境数据协议一致）---
ORGAN_SUB_TREASURE = 1      # 物件类型：宝箱
ORGAN_SUB_BUFF = 2          # 物件类型：加速buff
ORGAN_STATUS_AVAILABLE = 1  # 物件状态：可拾取

# --- 动作空间 ---
ACTION_DIM = 16             # 8个移动方向(0~7) + 8个闪现方向(8~15)

# --- 8 个移动方向在栅格上的增量 ---
# (行增量, 列增量)：行对应z轴(下为正)，列对应x轴(右为正)
# 方向编号：0=右, 1=右上, 2=上, 3=左上, 4=左, 5=左下, 6=下, 7=右下
DIRECTION_DELTAS = [
    (0, 1), (-1, 1), (-1, 0), (-1, -1),
    (0, -1), (1, -1), (1, 0), (1, 1),
]

# --- 世界坐标系下的单位方向向量 (dx, dz) ---
# 用于将动作方向与怪物方向做点积，判断是否在朝/背离怪物移动
WORLD_DIR_VECS = [
    (1.0, 0.0), (1.0, -1.0), (0.0, -1.0), (-1.0, -1.0),
    (-1.0, 0.0), (-1.0, 1.0), (0.0, 1.0), (1.0, 1.0),
]

# ============================================================
#               奖励系数（每个系数控制一项奖励的强度）
# ============================================================

# --- 基础系数 ---
ORGAN_LOCAL_DIST_MAX = 20.0           # 物件距离归一化上限（20格）
TREASURE_PROXIMITY_BOOST = 2.5        # 宝箱越近，靠近奖励的放大系数（1+2.5/(dist+1)）
BUFF_APPROACH_COEF = 0.12             # 靠近buff的奖励系数
BUFF_PROXIMITY_BOOST = 2.5            # buff越近，靠近奖励的放大系数

# --- r1: 基础生存奖励 ---
# 每活一步就给 +0.002，保证模型在没有显式收益时也能收到"先活着"的信号
SURVIVAL_REWARD = 0.002

# --- r2: 怪物距离shaping系数 ---
# 远离怪物=正奖励，靠近怪物=负奖励；系数越大，模型越重视与怪物保持距离
DIST_SHAPING_COEF = 0.12

# --- r3: 靠近宝箱的奖励系数 ---
# 每步朝宝箱移动时获得正奖励，系数×距离变化×proximity_boost
TREASURE_APPROACH_COEF = 0.16

# --- r4: 实际拾取宝箱的奖励（稀疏，吃到才给）---
# 旧 1.0 偏小，吃 10 个宝箱总收益不到 survive 一条路径的几十分之一。
# 提到 4.0 让"吃宝箱"成为明确目标，但又不至于像 code 里 10.0 那样炸 collect critic。
TREASURE_PICKUP_REWARD = 4.0

# --- r5: 实际拾取buff的奖励（稀疏，吃到才给）---
# 旧 0.35 太低，加速期前模型基本忽略 buff。提到 1.0。
BUFF_PICKUP_REWARD = 1.0

# --- r6: 原地不动的惩罚 ---
# 站着不动每步扣 0.03，无怪时×3=0.09（逼模型动起来）
STATIONARY_PENALTY = -0.03

# --- r7: 逃跑深度变化系数 ---
# 移到8方向平均可跑距离更大的位置=正（鼓励去开阔区域）
ESCAPE_DEPTH_DELTA_COEF = 0.15

# --- r8: 怪物感知安全分变化系数 ---
# 移到"考虑怪物方向后"安全分更高的位置=正
MONSTER_ESCAPE_DELTA_COEF = 0.12

# --- r9: 死角惩罚 ---
# 站在只有≤1个畅通方向的位置，每步扣 0.08
DEAD_CORNER_PENALTY = -0.08

# --- r10: 危险时方向选择系数 ---
# 怪物近时：朝安全方向跑的奖励系数
RETREAT_DIR_COEF = 0.06
# 怪物近时：朝怪物方向跑的惩罚系数
ALIGN_TOWARD_MONSTER_PENALTY = 0.05
# r10 触发阈值：怪物归一化距离 < 0.18 时才触发
MONSTER_CLOSE_DIST_NORM = 0.18

# --- 逃跑探针最大深度 ---
ESCAPE_PROBE_MAX = 10  # 每个方向最多探测10格

# --- 闪现穿墙相关 ---
# 8个方向的闪现距离：正交方向10格，斜向8格
FLASH_RANGES = [10, 8, 10, 8, 10, 8, 10, 8]
# r_flash_wall: 成功闪现穿过墙壁的奖励
FLASH_WALL_REWARD = 0.10

# --- 怪物短期记忆 ---
MEMORY_DECAY_STEPS = 80     # 怪物记忆保持80步后失效
MEMORY_FLEE_COEF = 0.05     # r_memory: 无怪时朝远离记忆中怪物方向移动的奖励系数
MEMORY_DANGER_WINDOW = 15   # 怪物消失15步内仍视为危险（允许闪现）

# --- 无怪探索（移植 code：奖励翻倍，鼓励积极探索）---
EXPLORE_MOVE_REWARD = 0.04         # 首次到达新格基础奖（旧 0.02 → 0.04）
EXPLORE_TOWARD_TREASURE = 0.05     # 朝最近宝箱方向额外奖（旧 0.026 → 0.05）
NO_MONSTER_TREASURE_BOOST = 2.5    # 无怪时 r3 靠近宝箱的倍率（保持 2.5，不降）

# --- 死胡同 ---
DEAD_END_EXIT_REWARD = 0.15         # r_dead_end: 在死胡同中朝出口方向走的奖励
DEAD_END_WRONG_DIR_PENALTY = -0.10  # r_dead_end: 在死胡同中走错方向（非出口）的惩罚
DEAD_END_STAY_PENALTY = -0.05       # r_dead_end: 停留在死胡同中的持续惩罚
DEAD_END_DEEPER_PENALTY = -0.05     # r_dead_end: 越走越深入死胡同的惩罚
DEAD_END_APPROACH_PENALTY = -0.06   # r_dead_end: 当前行动方向只剩<2格就到头的惩罚
BIRTH_ESCAPE_MAX_STEPS = 30         # 出生逃生模式最多持续30步

# --- 大佬补充的高级奖励系数 ---
# r_corridor: 位置开阔度正向奖励（鼓励去开阔区域拉扯）
CORRIDOR_REWARD_COEF = 0.02
# r_encircle: 被两怪夹击的惩罚（两怪分布在英雄两侧，夹角>120度）
ENCIRCLEMENT_PENALTY = -0.06
# r_flash_effect: 闪现真的脱险（怪物距离明显拉开）的奖励
FLASH_ESCAPE_REWARD = 0.12
# r_flash_effect: 闪现浪费（闪现后距离没改善）的惩罚
FLASH_WASTE_PENALTY = -0.08
# r_speedup_buffer: 怪物即将加速时还贴太近的惩罚系数
PRE_SPEEDUP_BUFFER_COEF = 0.04
# r_repeat: 最近100步内频繁回到同一位置的惩罚（V3记忆版：从-0.02加重到-0.06）
REPEAT_EXPLORE_PENALTY = -0.06
# r_wall_collision: 选了移动方向但撞墙没有位移的惩罚
WALL_COLLISION_PENALTY = -0.04
# r_second_monster: 第二只怪物距离过近时的额外惩罚
SECOND_MONSTER_CLOSE_PENALTY = -0.03


# ============================================================
#                    工具函数
# ============================================================

def _norm(v, v_max, v_min=0.0):
    """将数值 v 线性归一化到 [0, 1] 区间。超出范围会被裁剪。"""
    v = float(np.clip(v, v_min, v_max))
    return (v - v_min) / (v_max - v_min) if (v_max - v_min) > 1e-6 else 0.0


def _sym_norm_to_01(v):
    """将约 [-1,1] 范围的数值映射到 [0,1]。用于方向向量分量的归一化。"""
    return float(np.clip((float(v) + 1.0) * 0.5, 0.0, 1.0))


def _unit_vec(dx, dz):
    """将 (dx, dz) 归一化为单位向量。长度为0时返回 (0,0)。"""
    length = float(np.hypot(dx, dz))
    if length < 1e-6:
        return 0.0, 0.0
    return dx / length, dz / length


# ============================================================
#               视野穿透修复：射线遮挡检测
# ============================================================
# 环境返回的宝箱/buff列表包含21×21视野内的所有物件，
# 即使中间有墙挡着也会出现在列表里。
# 这里用DDA射线从英雄到物件画一条线，
# 如果中间任何一格是墙（map_info==0），就判定为被遮挡。

def _ray_blocked_by_wall(hero_pos, organ_pos, map_info):
    """射线遮挡检测：从英雄位置到物件位置之间是否有墙体阻断。

    原理：在 map_info 局部栅格上做 DDA（数字差分分析）射线，
    如果射线经过的任何中间格子是障碍（值==0），返回 True（被遮挡）。

    参数:
        hero_pos: 英雄位置 {"x": int, "z": int}
        organ_pos: 物件位置 {"x": int, "z": int}
        map_info: 21×21 局部地图（以英雄为中心），1=可走，0=障碍

    返回:
        True = 被墙挡住（不可见），False = 通视
    """
    if map_info is None or len(map_info) < 9:
        return False  # 地图数据不可用时保守返回可见
    n = len(map_info)
    crow = n // 2  # 英雄在地图中心
    ccol = n // 2

    # 将物件的世界坐标转换为 map_info 的局部栅格坐标
    hx = int(hero_pos["x"])
    hz = int(hero_pos["z"])
    ox = int(organ_pos["x"])
    oz = int(organ_pos["z"])
    target_r = oz - hz + crow  # z→行
    target_c = ox - hx + ccol  # x→列

    # 超出地图范围视为不可见
    if target_r < 0 or target_r >= n or target_c < 0 or target_c >= n:
        return True

    # DDA：从中心 (crow,ccol) 到 (target_r,target_c) 逐格检查
    dr = target_r - crow
    dc = target_c - ccol
    steps = max(abs(dr), abs(dc))
    if steps == 0:
        return False  # 物件就在脚下

    for t in range(1, steps):  # 不检查起点和终点，只检查中间格
        r = int(round(crow + dr * t / steps))
        c = int(round(ccol + dc * t / steps))
        if 0 <= r < n and 0 <= c < len(map_info[0]):
            if map_info[r][c] == 0:  # 碰到墙
                return True
        else:
            return True  # 超出边界视为被挡
    return False


def _encode_top_k_organs(hero_pos, organs, sub_type, k, map_info=None):
    """编码最近 k 个同类物件的特征。

    每个物件 4 维：[存在标记, 距离归一化, 相对方向dx归一化, 相对方向dz归一化]
    不存在的槽位填 [0, 0, 0, 0]。

    开启 map_info 时会先用射线检测过滤掉被墙遮挡的物件，
    避免模型"透视"看到墙后面的宝箱。

    参数:
        hero_pos: 英雄位置
        organs: 环境返回的全部物件列表
        sub_type: 要筛选的物件类型 (1=宝箱, 2=buff)
        k: 取最近的 k 个
        map_info: 可选，传入则启用视野遮挡过滤

    返回:
        (k*4,) 的 float32 数组
    """
    hx = float(hero_pos["x"])
    hz = float(hero_pos["z"])
    candidates = []
    for o in organs or []:
        try:
            if int(o.get("sub_type", 0)) != sub_type:
                continue
            if int(o.get("status", 0)) != ORGAN_STATUS_AVAILABLE:
                continue
        except (TypeError, ValueError):
            continue
        # 视野遮挡过滤：被墙挡住的物件直接跳过
        if map_info is not None and _ray_blocked_by_wall(hero_pos, o["pos"], map_info):
            continue
        p = o["pos"]
        ox, oz = float(p["x"]), float(p["z"])
        d = float(np.hypot(ox - hx, oz - hz))
        candidates.append((d, ox, oz))
    candidates.sort(key=lambda x: x[0])  # 按距离排序

    out = []
    for i in range(k):
        if i < len(candidates):
            d, ox, oz = candidates[i]
            dist_norm = _norm(d, MAP_SIZE * 1.414)  # 对角线最大距离约181
            inv = 1.0 / (d + 1e-6)
            dx = (ox - hx) * inv  # 归一化方向 x 分量 ∈ [-1, 1]
            dz = (oz - hz) * inv  # 归一化方向 z 分量 ∈ [-1, 1]
            out.extend([1.0, dist_norm, _sym_norm_to_01(dx), _sym_norm_to_01(dz)])
        else:
            out.extend([0.0, 0.0, 0.0, 0.0])  # 不存在时填零
    return np.array(out, dtype=np.float32)


# ============================================================
#               地图特征计算
# ============================================================

def _compute_expanded_map(map_info):
    """提取以英雄为中心的 9×9 局部通行地图。

    从 21×21 的 map_info 中裁出中心 9×9 区域，
    每格值为 1.0(可走) 或 0.0(障碍)。
    这 81 维会送入模型的 CNN 分支处理。

    返回: (81,) float32 数组
    """
    if map_info is None or len(map_info) < 9:
        return np.zeros(81, dtype=np.float32)
    n = len(map_info)
    crow = n // 2
    ccol = n // 2
    feat = np.zeros(81, dtype=np.float32)
    idx = 0
    for dr in range(-4, 5):      # -4到+4，共9行
        for dc in range(-4, 5):  # -4到+4，共9列
            r, c = crow + dr, ccol + dc
            v = 0.0
            if 0 <= r < n and 0 <= c < len(map_info[0]):
                v = float(map_info[r][c] != 0)  # 非0=可走→1.0，0=墙→0.0
            feat[idx] = v
            idx += 1
    return feat


def _compute_multichannel_map(map_info, hero_pos, monsters, organs):
    """生成 4 通道 21×21 地图特征（共 1764 维 flat）。方案 C 核心特征。

    以英雄为中心的 21×21 视野（环境原生尺寸，中心对应英雄位置）：
      Channel 0: 通行性    (1=路, 0=墙) — 直接取 map_info 21×21
      Channel 1: 怪物位置  (可见怪物所在格=1，其余=0)
      Channel 2: 宝箱位置  (可拾取宝箱格=1)
      Channel 3: buff 位置 (可拾取 buff 格=1)

    实体坐标转换：
      世界坐标差值 (ox-hx, oz-hz) → 中心偏移 (dc, dr) → 网格索引 (crow+dr, ccol+dc)
      超出 [0, n-1] 范围的实体不渲染。

    返回: (4*21*21,) = (1764,) float32，Channel-first 展平（C0 连续→C1→C2→C3）
    """
    n_ch = 4
    n_row = 21
    n_col = 21
    result = np.zeros((n_ch, n_row, n_col), dtype=np.float32)

    if map_info is None or len(map_info) < n_row:
        return result.flatten()

    n = len(map_info)
    w = len(map_info[0]) if n > 0 else 0
    crow = n // 2
    ccol = w // 2

    # Channel 0：通行性 — 直接读取 map_info 中心 21×21
    for r in range(n_row):
        for c in range(n_col):
            gr = crow - n_row // 2 + r
            gc = ccol - n_col // 2 + c
            if 0 <= gr < n and 0 <= gc < w:
                result[0, r, c] = float(map_info[gr][gc] != 0)

    hx = int(hero_pos["x"])
    hz = int(hero_pos["z"])

    # Channel 1：可见怪物位置
    for m in (monsters or [])[:2]:
        if int(m.get("is_in_view", 0)):
            mp = m["pos"]
            dc = int(mp["x"]) - hx    # x 偏移 → 列偏移
            dr = int(mp["z"]) - hz    # z 偏移 → 行偏移
            r = crow + dr
            c = ccol + dc
            # 转换到 0~20 的网格 index（crow=10, ccol=10）
            ri = crow + dr - (crow - n_row // 2)   # = dr + n_row // 2
            ci = ccol + dc - (ccol - n_col // 2)   # = dc + n_col // 2
            if 0 <= ri < n_row and 0 <= ci < n_col:
                result[1, ri, ci] = 1.0

    # Channel 2：可拾取宝箱；Channel 3：可拾取 buff
    for o in (organs or []):
        try:
            sub = int(o.get("sub_type", 0))
            sta = int(o.get("status", 0))
        except (TypeError, ValueError):
            continue
        if sta != ORGAN_STATUS_AVAILABLE:
            continue
        ch_idx = None
        if sub == ORGAN_SUB_TREASURE:
            ch_idx = 2
        elif sub == ORGAN_SUB_BUFF:
            ch_idx = 3
        if ch_idx is None:
            continue
        p = o["pos"]
        dc = int(p["x"]) - hx
        dr = int(p["z"]) - hz
        ri = dr + n_row // 2   # n_row//2 = 10，与 crow 偏移等价
        ci = dc + n_col // 2
        if 0 <= ri < n_row and 0 <= ci < n_col:
            result[ch_idx, ri, ci] = 1.0

    return result.flatten()


def _compute_escape_depth(map_info):
    """计算 8 个方向各能连续走多远（归一化到 [0,1]）。

    从英雄位置出发，沿每个方向逐格探测，碰到墙或边界就停。
    值越大说明该方向退路越深（越不容易被堵死）。

    返回: (8,) float32 数组
    """
    if map_info is None or len(map_info) < 9:
        return np.zeros(8, dtype=np.float32)
    n = len(map_info)
    crow = n // 2
    ccol = n // 2
    out = np.zeros(8, dtype=np.float32)
    for k, (drow, dcol) in enumerate(DIRECTION_DELTAS):
        depth = 0
        for t in range(1, ESCAPE_PROBE_MAX + 1):
            r = crow + drow * t
            c = ccol + dcol * t
            if r < 0 or r >= n or c < 0 or c >= len(map_info[0]):
                break
            if map_info[r][c] == 0:
                break
            depth += 1
        out[k] = depth / float(ESCAPE_PROBE_MAX)  # 0~1
    return out


def _compute_monster_aware_escape(escape_depth, hero_pos, monsters):
    """将逃跑深度与怪物方向结合，计算 8 方向的"感知安全分"。

    对每个方向，用该方向与"远离怪物方向"的点积来加权逃跑深度。
    方向既远离怪物、又通路深的，安全分最高。
    无怪物可见时，直接返回原始 escape_depth。

    返回: (8,) float32 数组
    """
    hx = float(hero_pos["x"])
    hz = float(hero_pos["z"])
    monster_flee_vecs = []
    for m in monsters[:2]:
        if not int(m.get("is_in_view", 0)):
            continue
        mp = m["pos"]
        mx, mz = float(mp["x"]), float(mp["z"])
        vx, vz = _unit_vec(hx - mx, hz - mz)  # 远离怪物的单位向量
        monster_flee_vecs.append((vx, vz))
    if not monster_flee_vecs:
        return escape_depth.astype(np.float32).copy()

    out = np.zeros(8, dtype=np.float32)
    for k in range(8):
        ux, uz = _unit_vec(*WORLD_DIR_VECS[k])
        per_m = []
        for vx, vz in monster_flee_vecs:
            away = max(0.0, ux * vx + uz * vz)  # 与远离方向的一致性 [0,1]
            per_m.append(float(escape_depth[k]) * away)
        out[k] = float(min(per_m)) if per_m else float(escape_depth[k])
    return out


def _compute_local_topology(map_info):
    """计算局部地形拓扑特征（3 维）。

    返回:
        [0] open_ratio: 以英雄为中心 5×5 区域的开阔率 (0~1)，越大越开阔
        [1] open_dir_count_norm: 4个正交方向中，连续3格畅通的方向数 / 4
        [2] dead_corner_flag: 是否为死角（畅通方向≤1个时为1.0）
    """
    if map_info is None or len(map_info) < 9:
        return np.array([0.5, 0.25, 0.0], dtype=np.float32)
    n = len(map_info)
    crow = n // 2
    ccol = n // 2

    # 5×5 区域的开阔率
    open_cells = 0
    total = 0
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            r, c = crow + dr, ccol + dc
            total += 1
            if 0 <= r < n and 0 <= c < len(map_info[0]) and map_info[r][c] != 0:
                open_cells += 1
    open_ratio = open_cells / float(total)

    # 4个正交方向的通行检测（连续3格无障碍才算畅通）
    cardinals = [(0, 1), (-1, 0), (0, -1), (1, 0)]
    open_dirs = 0
    for drow, dcol in cardinals:
        ok = True
        for t in range(1, 4):
            r = crow + drow * t
            c = ccol + dcol * t
            if r < 0 or r >= n or c < 0 or c >= len(map_info[0]) or map_info[r][c] == 0:
                ok = False
                break
        if ok:
            open_dirs += 1
    open_dir_count_norm = open_dirs / 4.0
    dead_corner_flag = 1.0 if open_dirs <= 1 else 0.0
    return np.array([open_ratio, open_dir_count_norm, dead_corner_flag], dtype=np.float32)


def _compute_nearest_monster_align(hero_pos, monsters):
    """计算 8 方向各自与"指向最近可见怪物"方向的对齐度。

    返回值 ∈ [-1, 1]：+1 表示该方向正朝怪物走，-1 表示背离。
    仅用于奖励 r10 的计算（不进入特征向量）。
    """
    hx = float(hero_pos["x"])
    hz = float(hero_pos["z"])
    best = None
    best_d = 1e9
    for m in monsters[:2]:
        if not int(m.get("is_in_view", 0)):
            continue
        mp = m["pos"]
        mx, mz = float(mp["x"]), float(mp["z"])
        d = float(np.hypot(hx - mx, hz - mz))
        if d < best_d:
            best_d = d
            best = (mx, mz)
    out = np.zeros(8, dtype=np.float32)
    if best is None or best_d < 1e-3:
        return out
    mx, mz = best
    tx, tz = _unit_vec(mx - hx, mz - hz)  # 指向怪物的单位向量
    for k, wd in enumerate(WORLD_DIR_VECS):
        ux, uz = _unit_vec(*wd)
        out[k] = float(ux * tx + uz * tz)  # 点积 = cos(夹角)
    return out


def _min_organ_dist(organs, hero_pos, sub_type, map_info=None):
    """找到最近同类物件的距离。

    返回: (原始欧式距离, 归一化距离)
    无可见物件时返回 (999.0, 1.0)。
    开启 map_info 时自动过滤被墙遮挡的物件。
    """
    hx = float(hero_pos["x"])
    hz = float(hero_pos["z"])
    best_raw = 999.0
    for o in organs or []:
        try:
            if int(o.get("sub_type", 0)) != sub_type:
                continue
            if int(o.get("status", 0)) != ORGAN_STATUS_AVAILABLE:
                continue
        except (TypeError, ValueError):
            continue
        if map_info is not None and _ray_blocked_by_wall(hero_pos, o["pos"], map_info):
            continue
        p = o["pos"]
        d = float(np.hypot(float(p["x"]) - hx, float(p["z"]) - hz))
        if d < best_raw:
            best_raw = d
    if best_raw > 998.0:
        return 999.0, 1.0
    return best_raw, _norm(best_raw, ORGAN_LOCAL_DIST_MAX)


# ============================================================
#               深度死胡同分析
# ============================================================

def _compute_dead_end_analysis(map_info):
    """分析当前位置是否在死胡同中（10 维）。

    V4 升级：改用 BFS 从英雄出发统计 21×21 视野内可达格子数，
    比原来的"8方向直线深度扫描"更准确（能识别 L 型/弯曲长通道）。

    判定策略：
    1. BFS 统计以英雄为中心的完整可达区域（格子数 n_reach）
    2. n_reach < 25  → 强死胡同 (1.0)：可达区域很小
    3. n_reach < 50  → 中等     (0.7)：较窄通道
    4. n_reach < 80  → 弱       (0.3)：中等开阔
    5. n_reach >= 80 → 不是死胡同 (0.0)：开阔区
    6. 出口方向：沿用 8 方向直线深度扫描，取最深方向标记

    返回:
        [0]    dead_end_score  : 0~1，越高越像死胡同
        [1:9]  exit_dir_0~7   : 出口方向标记（深度最大的方向=1）
        [9]    trapped_score   : 被困程度（0=出口近，1=最深处）
    """
    if map_info is None or len(map_info) < 9:
        return np.zeros(10, dtype=np.float32)

    n = len(map_info)
    w = len(map_info[0])
    crow = n // 2
    ccol = n // 2
    out = np.zeros(10, dtype=np.float32)

    # --- BFS 统计可达格子数 ---
    # 8 邻居搜索（斜向需要两条正交边至少一条通透，简化处理仅用 4 邻居即可近似）
    # 使用 4 邻居 BFS，对"斜向需要通透边"的约束做保守估计（斜向不单独连通）
    visited = np.zeros((n, w), dtype=bool)
    stack = [(crow, ccol)]
    visited[crow][ccol] = True
    n_reach = 1
    neighbors4 = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    while stack:
        r, c = stack.pop()
        for dr, dc in neighbors4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < w and not visited[nr][nc]:
                if map_info[nr][nc] != 0:  # 可走
                    visited[nr][nc] = True
                    n_reach += 1
                    stack.append((nr, nc))

    # --- 根据可达格子数判定死胡同等级 ---
    if n_reach < 25:
        out[0] = 1.0
    elif n_reach < 50:
        out[0] = 0.7
    elif n_reach < 80:
        out[0] = 0.3
    else:
        out[0] = 0.0

    # --- 8 方向直线深度扫描（用于出口标记和 trapped_score）---
    dir_depths = np.zeros(8, dtype=np.float32)
    for k, (drow, dcol) in enumerate(DIRECTION_DELTAS):
        depth = 0
        for t in range(1, 11):
            r = crow + drow * t
            c = ccol + dcol * t
            if r < 0 or r >= n or c < 0 or c >= w:
                break
            if map_info[r][c] == 0:
                break
            depth += 1
        dir_depths[k] = depth

    max_depth = float(np.max(dir_depths))

    # 标记出口方向 + 被困程度
    if out[0] > 0.3 and max_depth > 0:
        for k in range(8):
            if dir_depths[k] >= max_depth * 0.8:
                out[1 + k] = 1.0  # 标记最深方向为出口
        out[9] = max(0.0, 1.0 - max_depth / 10.0)  # 最深方向越浅=越被困

    return out


# ============================================================
#               闪现穿墙检测
# ============================================================

def _compute_flash_through_wall(map_info):
    """检测 8 个方向的闪现穿墙机会。

    模拟闪现的实际机制：从闪现最远端往近端扫描，找到第一个可走格作为落点。
    如果落点与英雄之间存在墙体，说明闪现可以穿墙到达该位置。
    这在逃出死胡同和被包夹时是免费的逃生手段。

    返回: (8,) float32 数组，每个方向的穿墙落点距离归一化（0=无穿墙机会）
    """
    if map_info is None or len(map_info) < 9:
        return np.zeros(8, dtype=np.float32)
    n = len(map_info)
    crow = n // 2
    ccol = n // 2
    out = np.zeros(8, dtype=np.float32)

    for k, (drow, dcol) in enumerate(DIRECTION_DELTAS):
        flash_range = FLASH_RANGES[k]  # 正交10格，斜向8格

        # 从最远端向近端扫描，找闪现落点（模拟环境机制）
        best_landing = -1
        for t in range(flash_range, 0, -1):
            r = crow + drow * t
            c = ccol + dcol * t
            if r < 0 or r >= n or c < 0 or c >= len(map_info[0]):
                continue
            if map_info[r][c] != 0:  # 找到可走的格子
                best_landing = t
                break

        if best_landing <= 0:
            continue

        # 检查英雄到落点之间是否有墙
        has_wall_between = False
        for t in range(1, best_landing):
            r = crow + drow * t
            c = ccol + dcol * t
            if r < 0 or r >= n or c < 0 or c >= len(map_info[0]):
                has_wall_between = True
                break
            if map_info[r][c] == 0:
                has_wall_between = True
                break

        if has_wall_between:
            # 中间有墙但落点可达 = 穿墙机会！评分 = 落点距离/闪现范围
            out[k] = float(best_landing) / float(flash_range)

    return out


def _compute_flash_to_safe(map_info, hero_pos, monsters):
    """计算 8 个闪现方向的"落点安全分"。

    对每个方向：
    1. 从最远端向近端扫，找到闪现实际落点（第一个可走格）
    2. 计算落点与每只可见怪物的切比雪夫距离（=max(|Δx|,|Δz|)）
    3. 落点与最近怪物切比雪夫距离 > 1（不在 3×3 贴身范围）→ 安全
       安全分 = landing / flash_range（落点越远越好）
    4. 落点在怪物 3×3 内（切比雪夫 ≤ 1）→ 落点即死，记 0

    返回: (8,) float32，值 0 = 不安全/无落点，>0 = 安全且越大越好
    """
    if map_info is None or len(map_info) < 9:
        return np.zeros(8, dtype=np.float32)
    n = len(map_info)
    crow = n // 2
    ccol = n // 2
    out = np.zeros(8, dtype=np.float32)

    hx = int(hero_pos["x"])
    hz = int(hero_pos["z"])

    # 收集可见怪物的世界坐标
    visible_monster_world = []
    for m in monsters[:2]:
        if int(m.get("is_in_view", 0)):
            mp = m["pos"]
            visible_monster_world.append((float(mp["x"]), float(mp["z"])))

    for k, (drow, dcol) in enumerate(DIRECTION_DELTAS):
        flash_range = FLASH_RANGES[k]

        # 找闪现落点（从远到近，第一个可走格）
        best_landing = -1
        for t in range(flash_range, 0, -1):
            r = crow + drow * t
            c = ccol + dcol * t
            if r < 0 or r >= n or c < 0 or c >= len(map_info[0]):
                continue
            if map_info[r][c] != 0:
                best_landing = t
                break

        if best_landing <= 0:
            continue  # 该方向没有可走落点

        # 落点的世界坐标
        land_wx = float(hx + dcol * best_landing)
        land_wz = float(hz + drow * best_landing)

        # 检查落点与所有可见怪物的切比雪夫距离
        if visible_monster_world:
            min_cheby = min(
                max(abs(land_wx - mx), abs(land_wz - mz))
                for mx, mz in visible_monster_world
            )
            if min_cheby <= 1.0:
                # 落点在怪物 3×3 贴身范围内 → 落地即死，安全分=0
                out[k] = 0.0
            else:
                # 安全：按落点距离归一化，越远脱险效果越好
                out[k] = float(best_landing) / float(flash_range)
        else:
            # 无可见怪物：只要有落点都算安全（用于死胡同等场景）
            out[k] = float(best_landing) / float(flash_range)

    return out


def _compute_flash_dist_gain(map_info, hero_pos, monsters):
    """计算 8 个闪现方向的"净距离改善量"（8 维）。

    问题 A 核心特征：提供客观的"闪完离怪物更远多少"信号，
    不关心方向是否朝向怪物，只看数学结果。

    关键场景：鲁班在死胡同底，怪物堵在开口；朝怪闪 10 格 → 越过怪 → 落在怪后方，
    净距离改善很大（d_before=2 → d_after=8，gain=+0.6）。
    模型能客观识别"朝怪闪现值不值得"。

    返回: (8,) float32，归一化到约 [-1, 1]
          > 0 = 闪后离最近怪物更远（拉开距离）
          < 0 = 闪后更近（反而贴近怪物）
          = 0 = 无怪/无落点/距离持平
    """
    if map_info is None or len(map_info) < 9:
        return np.zeros(8, dtype=np.float32)
    n = len(map_info)
    w = len(map_info[0])
    crow = n // 2
    ccol = n // 2
    out = np.zeros(8, dtype=np.float32)

    # 找最近可见怪物
    hx = float(hero_pos["x"])
    hz = float(hero_pos["z"])
    best_mx, best_mz = None, None
    best_d = 1e9
    for m in monsters[:2]:
        if int(m.get("is_in_view", 0)):
            mp = m["pos"]
            mx, mz = float(mp["x"]), float(mp["z"])
            d = float(np.hypot(hx - mx, hz - mz))
            if d < best_d:
                best_d = d
                best_mx, best_mz = mx, mz
    if best_mx is None:
        return out  # 无可见怪物，全部 0

    d_before = best_d

    for k, (drow, dcol) in enumerate(DIRECTION_DELTAS):
        flash_range = FLASH_RANGES[k]
        # 找闪现落点（从远到近第一个可走格）
        best_landing = -1
        for t in range(flash_range, 0, -1):
            r = crow + drow * t
            c = ccol + dcol * t
            if r < 0 or r >= n or c < 0 or c >= w:
                continue
            if map_info[r][c] != 0:
                best_landing = t
                break
        if best_landing <= 0:
            continue  # 该方向无落点

        # 落点世界坐标
        land_wx = hx + dcol * best_landing
        land_wz = hz + drow * best_landing
        d_after = float(np.hypot(land_wx - best_mx, land_wz - best_mz))

        # 归一化：按 10 格做分母，裁剪到 [-1, 1]
        gain = (d_after - d_before) / 10.0
        out[k] = float(np.clip(gain, -1.0, 1.0))

    return out


def _compute_trap_status(dead_end_score, escape_depth, nearest_align, flash_to_safe,
                          cur_min_dist_norm, any_monster_visible):
    """判断英雄是否处于"被困"状态（2 维）。V4 放宽版。

    返回:
        [0] is_trapped      : 0 或 1
        [1] can_flash_escape: is_trapped=1 且至少 1 个闪现方向能安全落地

    V4 放宽后的判定逻辑：
    - 可见怪物
    - 死胡同（dead_end_score > 0.3，不再要求强死胡同 0.5）
    - 存在至少一个可走方向朝怪（而非所有方向）
    - 怪物较近（cur_min_dist_norm < 0.35）

    目的：原来的 all_toward_monster 条件过严，用户提到"死胡同底+怪堵开口"
    场景下往往鲁班还能侧向或反向小幅移动，触发不了 is_trapped。
    放宽后能覆盖更多真实被困场景。
    """
    out = np.zeros(2, dtype=np.float32)

    if not any_monster_visible:
        return out  # 无可见怪物不算被困
    if float(dead_end_score) <= 0.3:
        return out  # 不在死胡同
    if float(cur_min_dist_norm) >= 0.35:
        return out  # 怪物还远（约 64 格以外）不算紧急

    passable_dirs = [k for k in range(8) if float(escape_depth[k]) > 0.15]
    if not passable_dirs:
        # 完全堵死（0 可走方向）也算被困
        out[0] = 1.0
    else:
        # 至少一个可走方向朝怪（any 而非 all）
        has_toward = any(float(nearest_align[k]) > 0.3 for k in passable_dirs)
        if has_toward:
            out[0] = 1.0

    if out[0] > 0.5:
        # 检查是否有安全落地的闪现方向
        if any(float(flash_to_safe[k]) > 0.0 for k in range(8)):
            out[1] = 1.0

    return out


# ============================================================
#          V5 新增：局势压力特征（shrink / sandwich / escape_loss）
# ============================================================

def _compute_pressure_feat(cur_reach, reach_history, cur_mean_escape, escape_history,
                            hero_pos, monsters):
    """计算局势压力特征（3 维）。

    回答的问题：
      - shrink_rate_20   ：活动区域在不在收缩（近 20 步 BFS 可达格数下降比例）
      - sandwich_score   ：是不是被双怪夹击（双怪连线到鲁班的垂距 + 张角指示）
      - escape_loss_rate ：退路在不在变差（近 20 步 8 方向平均逃跑深度下降率）

    参数:
        cur_reach: 当前 BFS 可达格数
        reach_history: deque，最近 20 步的 reach 历史
        cur_mean_escape: 当前 escape_depth 的平均值
        escape_history: deque，最近 20 步的 mean_escape 历史
        hero_pos: 英雄位置 {"x","z"}
        monsters: 原始怪物列表

    返回: (3,) float32 数组，全部在 [0, 1]
    """
    out = np.zeros(3, dtype=np.float32)

    # shrink_rate_20：可达格数下降比例
    if len(reach_history) >= 5:
        past_reach = float(reach_history[0])  # 最早那一帧
        if past_reach > 1e-3:
            shrink = (past_reach - float(cur_reach)) / past_reach
            out[0] = float(np.clip(shrink, 0.0, 1.0))

    # sandwich_score：两只怪都可见时，双怪连线到鲁班的垂距归一化 + 张角过大指示
    visible = []
    for m in (monsters or [])[:2]:
        if int(m.get("is_in_view", 0)):
            mp = m["pos"]
            visible.append((float(mp["x"]), float(mp["z"])))
    if len(visible) == 2:
        hx = float(hero_pos["x"])
        hz = float(hero_pos["z"])
        m1x, m1z = visible[0]
        m2x, m2z = visible[1]
        # 双怪连线向量
        lx, lz = m2x - m1x, m2z - m1z
        line_len = float(np.hypot(lx, lz))
        if line_len > 1e-3:
            # 英雄到线段的垂距（对连线向量做叉乘）
            nx, nz = -lz / line_len, lx / line_len  # 法向量
            px = hx - m1x
            pz = hz - m1z
            perp_dist = abs(px * nx + pz * nz)
            # 垂距越小越危险；归一化：10 格以内都算危险
            perp_score = 1.0 - float(np.clip(perp_dist / 10.0, 0.0, 1.0))
            # 张角：两怪相对英雄的张角
            v1x, v1z = m1x - hx, m1z - hz
            v2x, v2z = m2x - hx, m2z - hz
            len1 = float(np.hypot(v1x, v1z))
            len2 = float(np.hypot(v2x, v2z))
            angle_score = 0.0
            if len1 > 1e-3 and len2 > 1e-3:
                cos_a = float(np.clip((v1x * v2x + v1z * v2z) / (len1 * len2), -1.0, 1.0))
                # cos < 0 → 张角 > 90 度，即英雄在双怪之间
                if cos_a < 0.0:
                    angle_score = min(1.0, (-cos_a))  # 张角越大分数越高
            out[1] = float(np.clip(0.6 * perp_score + 0.4 * angle_score, 0.0, 1.0))

    # escape_loss_rate_20：8 方向平均逃跑深度下降率
    if len(escape_history) >= 5:
        past_escape = float(escape_history[0])
        if past_escape > 1e-3:
            loss = (past_escape - float(cur_mean_escape)) / past_escape
            out[2] = float(np.clip(loss, 0.0, 1.0))

    return out


def _compute_flash_strategic_feat(flash_cd_norm, flash_dist_gain, dead_end_score, pressure):
    """计算闪现战略价值特征（2 维）。

    回答的问题：
      - flash_strategic_value  ：闪现如果用掉，对脱困有多大价值（压力高 + 死胡同 + 有拉开距离）
      - flash_opportunity_cost ：低压力时闪现作为战略储备的价值（越低越不应用）

    参数:
        flash_cd_norm: 闪现冷却归一化（0=可用，>0=冷却中）
        flash_dist_gain: 8 方向闪现净距离改善（已归一化到 [-1,1]）
        dead_end_score: 死胡同得分 [0,1]
        pressure: 当前压力分数 [0,1]

    返回: (2,) float32 数组，全部在 [0, 1]
    """
    out = np.zeros(2, dtype=np.float32)
    flash_ready = flash_cd_norm <= 1e-3  # 闪现可用

    if flash_ready:
        max_gain = float(np.max(flash_dist_gain)) if len(flash_dist_gain) > 0 else 0.0
        max_gain = max(0.0, max_gain)  # 只关心正向收益
        # strategic_value：闪现可用 + 压力大 + 死胡同 + 有拉开距离
        out[0] = float(np.clip(max_gain * (1.0 + float(dead_end_score)) * float(pressure), 0.0, 1.0))
        # opportunity_cost：低压力时用闪现的机会成本高
        if pressure < 0.3:
            out[1] = float(np.clip(max_gain * (0.3 - float(pressure)) / 0.3, 0.0, 1.0))

    return out


def _compute_reach_count(map_info):
    """简化 BFS 统计 21×21 视野内从英雄出发的可达格子数（与 dead_end_analysis 逻辑一致）。

    单独抽出来是为了给 pressure_feat 用（不想重复修改 dead_end_analysis）。
    """
    if map_info is None or len(map_info) < 9:
        return 0
    n = len(map_info)
    w = len(map_info[0]) if n > 0 else 0
    crow = n // 2
    ccol = w // 2
    visited = np.zeros((n, w), dtype=bool)
    stack = [(crow, ccol)]
    visited[crow][ccol] = True
    n_reach = 1
    neighbors4 = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    while stack:
        r, c = stack.pop()
        for dr, dc in neighbors4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < w and not visited[nr][nc]:
                if map_info[nr][nc] != 0:
                    visited[nr][nc] = True
                    n_reach += 1
                    stack.append((nr, nc))
    return n_reach


# ============================================================
#                    Preprocessor 主类
# ============================================================

class Preprocessor:
    """特征预处理器：将环境原始观测转换为特征向量 + 合法动作掩码 + 即时奖励。

    每局开始时调用 reset() 清空所有跨步状态，
    每步调用 feature_process() 返回 (feature, legal_action, reward)。
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """每局开始时重置所有跨步状态。"""
        self.step_no = 0
        self.max_step = 1000

        # --- 上一步的状态快照（用于计算奖励中的"变化量"）---
        self.last_min_monster_dist_norm = 0.5   # 上步最近怪物归一化距离
        self.last_treasure_count = 0            # 上步已收集宝箱数
        self.last_buff_remaining_time = 0       # 上步buff剩余时间
        self.last_hero_pos = None               # 上步英雄位置 (x, z)
        self.stuck_step_count = 0               # 连续原地不动的步数
        self.last_local_open_ratio = 0.5        # 上步局部开阔率
        self.last_dead_corner_flag = 0.0        # 上步死角标记
        self.last_escape_score = 0.5            # 上步平均安全分
        self.last_min_treasure_dist_norm = 1.0  # 上步最近宝箱归一化距离
        self.last_min_buff_dist_norm = 1.0      # 上步最近buff归一化距离
        self.last_mean_escape_depth = 0.5       # 上步平均逃跑深度
        self.last_buff_collected = 0            # 上步已收集buff数
        self.last_treasure_collected = 0        # 上步已收集宝箱数

        # --- 怪物短期记忆（怪物消失后仍记住它最后的位置和方向）---
        self.memory_monster_positions = [None, None]      # 两只怪最后已知位置 (x,z)
        self.memory_monster_last_seen = [0, 0]            # 最后看到的step编号
        self.memory_monster_flee_dir = [(0.0, 0.0), (0.0, 0.0)]  # 远离怪物的单位方向

        # --- 出生逃生模式（开局前30步如果在死胡同则双倍惩罚）---
        self.birth_escape_mode = True
        self.birth_escape_steps = 0

        # --- 位置历史（最近100步的位置，用于检测重复探索；V3记忆版从50扩到100）---
        self.position_history = deque(maxlen=100)

        # --- 探索特征所需状态（对应 exploration_feat 12 维）---
        self.visit_count_map = {}           # 每个位置被访问的次数（用于陌生度）
        self.episode_start_pos = None       # 本局起点位置（用于位移计算）

        # --- r_easy_treasure 改 delta 型所需：记录上一步"在宝箱 4 格内"时的距离 ---
        # 初始为圈边界 4.0，这样进入圈内时 delta 从 0 开始累积，避免冲击奖励
        self._last_near_treasure_dist = 4.0

        # --- V4 反绕圈所需状态 ---
        # visited_cells: 本局所有访问过的格子集合（r_explore 只奖首次访问时查询）
        # long_position_history: 200 步滑动窗口位置历史（用于计算 coverage_rate）
        self.visited_cells = set()
        self.long_position_history = deque(maxlen=200)

        # --- V5 新增：局势压力特征所需历史（20 步滑动窗口）---
        self.reach_history = deque(maxlen=20)          # BFS 可达格数历史
        self.mean_escape_history = deque(maxlen=20)    # 8 方向平均逃跑深度历史

        # --- V5 新增：reward breakdown（供监控上报各子项均值）---
        # 每步写入，Agent 通过 remain_info 带出给 train_workflow 聚合
        self.last_reward_breakdown = {}

        # --- V5 新增：闪现质量统计（单局累积，供 monitor 聚合）---
        self.flash_use_count = 0            # 闪现动作总次数
        self.flash_wall_count = 0           # 闪现穿墙次数
        self.flash_waste_count = 0          # 闪现浪费次数（r_flash_effect<0）
        self.flash_escape_trap_count = 0    # 被困闪现脱困次数
        self.flash_cd_bucket = [0, 0, 0]    # [cd=0 用闪, 0<cd<0.2 用闪, cd>=0.2 用闪]
        self.flash_post_dist_change_sum = 0.0  # 闪现后瞬时怪距变化累积

        # --- V5 新增：前后期统计（单局累积）---
        # 前后期分界使用 Config.SPEEDUP_THRESHOLD（默认 300，与 monster_speedup 对齐）：
        #   early = step < SPEEDUP_THRESHOLD（加速前）
        #   late  = step >= SPEEDUP_THRESHOLD（加速后）
        self.early_reward_sum = 0.0
        self.late_reward_sum = 0.0
        self.early_treasure = 0
        self.late_treasure = 0

        # --- V5 新增：闪现前 CD 快照（供 flash_cd_bucket 分桶）---
        self._prev_flash_cd_norm = 0.0

        # --- 视野切换跟踪（用于把 r2/r3 在切换帧归零，避免 ±0.x 假信号）---
        self.last_any_monster_visible = False
        self.last_treasure_visible = False

        # --- r_border_approach 用：上一步到最近边界的距离 ---
        self._last_min_border_dist = 999.0

    def feature_process(self, env_obs, last_action):
        """核心方法：把环境观测转换为 (特征, 合法动作, 奖励)。

        参数:
            env_obs: 环境返回的原始观测字典
            last_action: 上一步执行的动作 (0~15)，第一步传 None 或 -1

        返回:
            feature: (191,) float32 数组 — 喂给神经网络
            legal_action: list[int] — 16 维合法动作掩码 (0/1)
            reward: list[float] — 长度 1 的列表，本步即时奖励
        """
        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info = observation["env_info"]
        map_info = observation["map_info"]       # 21×21 局部地图
        legal_act_raw = observation.get("legal_action")
        if legal_act_raw is None:
            legal_act_raw = observation.get("legal_act", [])

        self.step_no = int(observation["step_no"])
        self.max_step = int(env_info.get("max_step", 1000))

        # ================================================================
        #                      特征提取（191 维）
        # ================================================================

        # ------ 英雄特征 (4 维) ------
        # 回答"我现在是什么状态"
        hero = frame_state["heroes"]
        hero_pos = hero["pos"]
        hero_x_norm = _norm(float(hero_pos["x"]), MAP_SIZE)          # 位置 x 归一化 [0,1]
        hero_z_norm = _norm(float(hero_pos["z"]), MAP_SIZE)          # 位置 z 归一化 [0,1]
        flash_cd_norm = _norm(float(hero.get("flash_cooldown", 0)), MAX_FLASH_CD)   # 闪现CD [0,1]，0=可用
        buff_remain_norm = _norm(float(hero.get("buff_remaining_time", 0)), MAX_BUFF_DURATION)  # buff剩余 [0,1]
        hero_feat = np.array([hero_x_norm, hero_z_norm, flash_cd_norm, buff_remain_norm], dtype=np.float32)

        # ------ 怪物特征 (10 维 = 2只 × 5维) ------
        # 回答"危险从哪来、有多快、有多近"
        # 每只怪: [是否在视野(0/1), 位置x, 位置z, 速度, 距离]
        monsters = frame_state.get("monsters", []) or []
        monster_feats = []
        for i in range(2):
            if i < len(monsters):
                m = monsters[i]
                is_in_view = float(int(m.get("is_in_view", 0)))
                m_pos = m["pos"]
                if is_in_view > 0:
                    m_x_norm = _norm(float(m_pos["x"]), MAP_SIZE)
                    m_z_norm = _norm(float(m_pos["z"]), MAP_SIZE)
                    m_speed_norm = _norm(float(m.get("speed", 1)), MAX_MONSTER_SPEED)
                    raw_dist = float(np.hypot(
                        float(hero_pos["x"]) - float(m_pos["x"]),
                        float(hero_pos["z"]) - float(m_pos["z"]),
                    ))
                    dist_norm = _norm(raw_dist, MAP_SIZE * 1.414)
                else:
                    m_x_norm = m_z_norm = m_speed_norm = 0.0
                    dist_norm = 1.0  # 不在视野=最远
                monster_feats.append(
                    np.array([is_in_view, m_x_norm, m_z_norm, m_speed_norm, dist_norm], dtype=np.float32)
                )
            else:
                monster_feats.append(np.zeros(5, dtype=np.float32))

        any_monster_visible = any(int(m.get("is_in_view", 0)) for m in monsters[:2])

        # ------ 共享变量：最近怪物距离 + 危险等级（提前，供 trap_status / mask / 奖励段共用）------
        cur_min_dist_norm = 1.0
        for m_feat in monster_feats:
            if m_feat[0] > 0:
                cur_min_dist_norm = min(cur_min_dist_norm, float(m_feat[4]))
        danger_level = 0.0
        if any_monster_visible:
            danger_level = max(0.0, 1.0 - cur_min_dist_norm / MONSTER_CLOSE_DIST_NORM)

        # ------ 怪物短期记忆：更新 + 生成特征 (8 维) ------
        # 回答"怪物消失前最后在哪个方向"
        hx_f = float(hero_pos["x"])
        hz_f = float(hero_pos["z"])

        # 更新记忆：怪物可见时刷新其位置和远离方向
        for i in range(min(2, len(monsters))):
            m = monsters[i]
            if int(m.get("is_in_view", 0)):
                mp = m["pos"]
                mx_f, mz_f = float(mp["x"]), float(mp["z"])
                self.memory_monster_positions[i] = (mx_f, mz_f)
                self.memory_monster_last_seen[i] = self.step_no
                fx, fz = _unit_vec(hx_f - mx_f, hz_f - mz_f)
                self.memory_monster_flee_dir[i] = (fx, fz)

        # 过期清理：超过 MEMORY_DECAY_STEPS 步没看到就清除
        for i in range(2):
            if self.memory_monster_positions[i] is not None:
                if self.step_no - self.memory_monster_last_seen[i] > MEMORY_DECAY_STEPS:
                    self.memory_monster_positions[i] = None

        # 生成记忆特征：每只怪 4 维 [记忆有效, 衰减因子, 逃离方向dx, 逃离方向dz]
        memory_feat = np.zeros(8, dtype=np.float32)
        for i in range(2):
            base = i * 4
            if self.memory_monster_positions[i] is not None:
                age = self.step_no - self.memory_monster_last_seen[i]
                if age <= MEMORY_DECAY_STEPS:
                    decay = 1.0 - float(age) / float(MEMORY_DECAY_STEPS)  # 1.0→0.0 线性衰减
                    memory_feat[base + 0] = 1.0       # 记忆有效标记
                    memory_feat[base + 1] = decay      # 衰减因子
                    fx, fz = self.memory_monster_flee_dir[i]
                    memory_feat[base + 2] = _sym_norm_to_01(fx)  # 逃离方向 x
                    memory_feat[base + 3] = _sym_norm_to_01(fz)  # 逃离方向 z

        # ------ 物件特征（宝箱 12维 + buff 8维）------
        # 回答"哪个资源值得追、加速道具在哪"
        # 已启用视野遮挡过滤：被墙挡住的物件会被忽略
        organs = frame_state.get("organs", []) or []
        treasure_top3_feat = _encode_top_k_organs(hero_pos, organs, ORGAN_SUB_TREASURE, 3, map_info)
        buff_top2_feat = _encode_top_k_organs(hero_pos, organs, ORGAN_SUB_BUFF, 2, map_info)

        # ------ 地图特征（方案C：4 通道 21×21 多通道地图 + 派生标量特征）------
        # multichannel_map: 1764 维 (4×21×21)，各通道渲染通行性/怪物/宝箱/buff 空间分布
        multichannel_map = _compute_multichannel_map(map_info, hero_pos, monsters, organs)
        escape_depth = _compute_escape_depth(map_info)        # 8方向逃跑深度
        monster_escape = _compute_monster_aware_escape(escape_depth, hero_pos, monsters)  # 8方向怪物安全分
        topology_feat = _compute_local_topology(map_info)     # 拓扑：开阔率 + 方向数 + 死角

        # ------ 死胡同分析 (10 维) ------
        dead_end_info = _compute_dead_end_analysis(map_info)

        # ------ 闪现穿墙检测 (8 维) ------
        flash_through = _compute_flash_through_wall(map_info)

        # ------ 闪现落点安全分 (8) + 净距离改善 (8) + 被困状态 (2) ------
        # flash_to_safe:    8 方向闪现落点是否在怪物 3×3 以外（0=不安全）
        # flash_dist_gain:  8 方向闪现"净距离改善量"（归一化[-1,1]），客观评估朝怪闪是否值得
        # nearest_align:    8 方向与"指向最近怪物"的对齐度
        # trap_status:      [is_trapped, can_flash_escape] （V4 放宽版）
        flash_to_safe = _compute_flash_to_safe(map_info, hero_pos, monsters)
        flash_dist_gain = _compute_flash_dist_gain(map_info, hero_pos, monsters)
        nearest_align = _compute_nearest_monster_align(hero_pos, monsters)
        trap_status = _compute_trap_status(
            dead_end_info[0], escape_depth, nearest_align, flash_to_safe,
            cur_min_dist_norm, any_monster_visible,
        )

        # ------ V5 新增：局势压力特征 (3 维) ------
        # shrink_rate_20 / sandwich_score / escape_loss_rate_20
        cur_reach = _compute_reach_count(map_info)
        cur_mean_escape = float(np.mean(escape_depth))
        pressure_feat = _compute_pressure_feat(
            cur_reach, self.reach_history, cur_mean_escape, self.mean_escape_history,
            hero_pos, monsters,
        )

        # 计算压力分数（供 reward 加权 + flash_strategic_feat 使用）
        # pressure = 0.4*shrink + 0.3*sandwich + 0.2*danger_level + 0.1*escape_loss
        pressure_score = float(np.clip(
            0.4 * float(pressure_feat[0])
            + 0.3 * float(pressure_feat[1])
            + 0.2 * float(danger_level)
            + 0.1 * float(pressure_feat[2]),
            0.0, 1.0,
        ))

        # ------ V5 新增：闪现战略价值特征 (2 维) ------
        flash_strategic_feat = _compute_flash_strategic_feat(
            flash_cd_norm, flash_dist_gain, dead_end_info[0], pressure_score,
        )

        # ------ 合法动作掩码 (16 维) ------
        action_dim = min(ACTION_DIM, int(Config.ACTION_NUM))
        legal_action = [1] * action_dim
        if isinstance(legal_act_raw, list) and legal_act_raw:
            if isinstance(legal_act_raw[0], bool):
                for j in range(min(action_dim, len(legal_act_raw))):
                    legal_action[j] = int(legal_act_raw[j])
            else:
                valid_set = {int(a) for a in legal_act_raw if int(a) < action_dim}
                legal_action = [1 if j in valid_set else 0 for j in range(action_dim)]

        # 闪现控制：无怪时默认禁用闪现，除非记忆危险+有穿墙机会，或在死胡同中
        # V5 放宽：dead_end 阈值从 0.5 降到 0.3，配合 flash_strategic_feat 让模型更自由决策
        if not any_monster_visible:
            memory_danger = any(
                self.memory_monster_positions[i] is not None
                and (self.step_no - self.memory_monster_last_seen[i]) <= MEMORY_DANGER_WINDOW
                for i in range(2)
            )
            in_dead_end = dead_end_info[0] > 0.3
            for j in range(8, action_dim):
                dir_idx = j - 8
                allow = flash_through[dir_idx] > 0.3 and (memory_danger or in_dead_end)
                if not allow:
                    legal_action[j] = 0

        # 被困场景（死胡同+所有移动方向朝怪）：屏蔽"落地即死"的闪现方向（温和策略）
        # 保留 flash_to_safe > 0 的方向，只屏蔽绝对落点死亡的方向
        if trap_status[0] > 0.5:
            for j in range(8, action_dim):
                dir_idx = j - 8
                if float(flash_to_safe[dir_idx]) <= 0.0:
                    legal_action[j] = 0

        if sum(legal_action) == 0:
            legal_action = [1] * action_dim  # 安全兜底

        # ------ 进度特征 (3 维) ------
        # 回答"游戏到了什么阶段、怪物加速了没"
        step_norm = _norm(float(self.step_no), float(self.max_step))
        is_speedup = 0.0
        for m in monsters[:2]:
            if int(m.get("is_in_view", 0)) and float(m.get("speed", 1)) > 1.0:
                is_speedup = 1.0
                break
        # 默认值用 Config.SPEEDUP_THRESHOLD（300），与 train_env_conf.toml 的 monster_speedup 对齐。
        # 旧版默认 500 会让"离加速还有多久"算错，间接影响 r_speedup_buffer 触发时机。
        _default_ms = int(getattr(Config, "SPEEDUP_THRESHOLD", 300))
        try:
            ms = int(env_info.get("monster_speed", _default_ms))
        except (TypeError, ValueError):
            ms = _default_ms
        rem_speedup = max(0, ms - self.step_no)
        speedup_countdown_norm = _norm(float(rem_speedup), float(max(ms, 1)))  # 离加速还有多久
        progress_feat = np.array([step_norm, is_speedup, speedup_countdown_norm], dtype=np.float32)

        # ================================================================
        #   共享变量（供特征 + 奖励共用）
        #   cur_min_dist_norm/danger_level 已在 monster_feats 之后提前计算。
        #   此处只补算资源距离（依赖 organs + map_info）。
        # ================================================================
        # 最近宝箱距离（启用墙体遮挡过滤）
        raw_treasure_dist, cur_treasure_dn = _min_organ_dist(organs, hero_pos, ORGAN_SUB_TREASURE, map_info)
        # 最近 buff 距离
        raw_buff_dist, cur_buff_dn = _min_organ_dist(organs, hero_pos, ORGAN_SUB_BUFF, map_info)

        # ------ 探索特征 (12 维) ------
        # [0]     displacement_norm : 相对起点位移归一化
        # [1:9]   novelty_8dir      : 8 方向相邻格的陌生度（1/(1+访问次数)）
        # [9]     treasure_safety   : 宝箱"值得追"的综合指标（近且安全）
        # [10]    buff_safety       : buff 同上
        # [11]    buff_progress     : 当前 buff 剩余时间归一化
        hx_cur = int(hero_pos["x"])
        hz_cur = int(hero_pos["z"])
        if self.episode_start_pos is None:
            self.episode_start_pos = (hx_cur, hz_cur)
        disp_raw = float(np.hypot(
            hx_cur - self.episode_start_pos[0],
            hz_cur - self.episode_start_pos[1],
        ))
        displacement_norm = _norm(disp_raw, MAP_SIZE)

        novelty_8dir = np.zeros(8, dtype=np.float32)
        for k, (drow, dcol) in enumerate(DIRECTION_DELTAS):
            probe_pos = (hx_cur + dcol, hz_cur + drow)
            cnt = self.visit_count_map.get(probe_pos, 0)
            novelty_8dir[k] = 1.0 / (1.0 + float(cnt))

        # 资源安全分 = 资源距离得分 × 安全因子（有怪时用 cur_min_dist_norm 作为安全代理，越远越安全）
        safety_factor = float(cur_min_dist_norm) if any_monster_visible else 1.0
        treasure_safety = max(0.0, 1.0 - float(cur_treasure_dn)) * safety_factor
        buff_safety = max(0.0, 1.0 - float(cur_buff_dn)) * safety_factor

        exploration_feat = np.array(
            [displacement_norm]
            + list(novelty_8dir)
            + [treasure_safety, buff_safety, float(buff_remain_norm)],
            dtype=np.float32,
        )

        # ------ 反绕圈 2 维特征 ------
        # is_new_cell:  当前位置是否是本局首次访问（1=首次，0=已访问过）
        # coverage_rate: 最近 200 步窗口内去过的唯一格子数 / 窗口实际长度
        #                低 coverage = 在小范围绕圈
        cur_pos_tuple = (hx_cur, hz_cur)
        is_new_cell = 0.0 if cur_pos_tuple in self.visited_cells else 1.0
        if len(self.long_position_history) >= 20:
            coverage_rate = float(
                len(set(self.long_position_history)) / float(len(self.long_position_history))
            )
        else:
            coverage_rate = 1.0  # 开局默认 1.0，不给压力
        anti_loop_feat = np.array([is_new_cell, coverage_rate], dtype=np.float32)

        # ------ 拼接总特征 (1899 维) ------
        # V5：新增 pressure_feat(3) + flash_strategic_feat(2)，追加在末尾
        feature = np.concatenate([
            hero_feat,                                      #    4  英雄状态
            monster_feats[0],                               #    5  怪物1
            monster_feats[1],                               #    5  怪物2
            treasure_top3_feat,                             #   12  最近3宝箱
            buff_top2_feat,                                 #    8  最近2buff
            multichannel_map,                               # 1764  4通道21×21地图 → FPN CNN
            escape_depth,                                   #    8  8方向逃跑深度
            monster_escape,                                 #    8  8方向怪物安全分
            topology_feat,                                  #    3  局部拓扑
            memory_feat,                                    #    8  怪物记忆
            flash_through,                                  #    8  闪现穿墙机会
            dead_end_info,                                  #   10  死胡同分析（V4 BFS 版）
            np.array(legal_action, dtype=np.float32),       #   16  合法动作掩码
            progress_feat,                                  #    3  进度
            exploration_feat,                               #   12  探索：位移+陌生度+资源安全+buff进度
            flash_to_safe,                                  #    8  8方向闪现落点安全分（>0=安全，0=落地即死）
            trap_status,                                    #    2  被困状态：[is_trapped, can_flash_escape]
            flash_dist_gain,                                #    8  8方向闪现"净距离改善量"（归一化[-1,1]）
            anti_loop_feat,                                 #    2  反绕圈：[is_new_cell, coverage_rate]
            pressure_feat,                                  #    3  V5 压力特征：shrink/sandwich/escape_loss
            flash_strategic_feat,                           #    2  V5 闪现战略：strategic_value/opportunity_cost
        ])                                                  # 合计 1899

        # ================================================================
        #                      奖励计算（V5 三组压力加权版）
        #   - 删除 phase_survival_weight / phase_treasure_weight / no_monster_boost /
        #     survival_danger_scale / treasure_danger_scale 五个旧加权
        #   - 改用统一压力分数 pressure_score（在特征段已计算）对三组乘权重
        #   - 返回 3 维 reward: [r_survive_sum, r_collect_sum, r_explore_sum]
        # ================================================================

        # ---- r1: 基础生存奖励 ----
        # 每活一步固定给 +0.002，让模型在没有其他信号时也知道"活着=好"
        r1 = SURVIVAL_REWARD

        # ---- r2: 怪物距离 shaping ----
        # 修复：怪物进入/离开视野那一帧，cur_min_dist_norm 会从 1.0 突降/突升到实际值，
        # 这是传感器切换造成的虚假 ±0.x 信号，不代表英雄真的在远离/靠近怪物，直接归零。
        if any_monster_visible != self.last_any_monster_visible:
            r2 = 0.0
        else:
            r2 = DIST_SHAPING_COEF * (cur_min_dist_norm - self.last_min_monster_dist_norm)

        # ---- r3: 靠近宝箱奖励 ----
        # 修复：宝箱被墙挡住/重新可见那一帧，cur_treasure_dn 会突跳到 1.0 或从 1.0 突降，
        # 同样会产生 r3 的假惩罚/假奖励，让模型误以为绕路是错。切换帧直接归零。
        treasure_visible_now = raw_treasure_dist < 998.0
        proximity_boost = 1.0 + TREASURE_PROXIMITY_BOOST / (raw_treasure_dist + 1.0)
        proximity_boost = min(proximity_boost, 5.0)  # 上限保护，防止贴脸数值爆炸
        if treasure_visible_now != self.last_treasure_visible:
            r3 = 0.0
        else:
            r3 = TREASURE_APPROACH_COEF * (self.last_min_treasure_dist_norm - cur_treasure_dn) * proximity_boost

        # ---- r4: 拾取宝箱奖励（稀疏）----
        # 只在实际吃到宝箱时触发，每个宝箱 +1.0
        try:
            collected = int(hero.get("treasure_collected_count", 0))
        except (TypeError, ValueError):
            collected = 0
        pickup_delta = max(0, collected - self.last_treasure_collected)
        r4 = TREASURE_PICKUP_REWARD * float(pickup_delta)

        # ---- r5: 拾取 buff 奖励（按加速点分段加强）----
        # 只在实际吃到 buff 时触发，每个 buff 给奖。基础 BUFF_PICKUP_REWARD=1.0。
        # 设计思路（用户要求）：
        #   step <  speedup-100  (默认 <200) → ×1   保持基础，避免开局乱抢 buff 让出位置给怪
        #   speedup-100..speedup (200..300)  → ×2   加速前 100 步，鼓励抢 buff 抢加速差距
        #   step >= speedup       (>=300)    → ×3.5 加速后吃到 buff = 救命级大奖（与穿墙闪×4 接近）
        try:
            buff_collected = int(env_info.get("collected_buff", 0))
        except (TypeError, ValueError):
            buff_collected = 0
        buff_delta = max(0, buff_collected - self.last_buff_collected)
        if buff_delta > 0:
            _speedup_thr = int(getattr(Config, "SPEEDUP_THRESHOLD", 300))
            _pre_speedup_window = 100
            if self.step_no >= _speedup_thr:
                _buff_stage_mul = 3.5
            elif self.step_no >= _speedup_thr - _pre_speedup_window:
                _buff_stage_mul = 2.0
            else:
                _buff_stage_mul = 1.0
            r5 = BUFF_PICKUP_REWARD * float(buff_delta) * _buff_stage_mul
        else:
            r5 = 0.0

        # ---- r_buff_approach: 靠近 buff 奖励 ----
        # raw_buff_dist / cur_buff_dn 已在 feature 段算好
        buff_prox_boost = 1.0 + BUFF_PROXIMITY_BOOST / (raw_buff_dist + 1.0)
        r_buff_approach = BUFF_APPROACH_COEF * (self.last_min_buff_dist_norm - cur_buff_dn) * buff_prox_boost

        # ---- r6: 原地不动惩罚 ----
        # 位置没变就扣分，无怪时扣更多（×3）逼模型去探索
        cur_pos = (int(hero_pos["x"]), int(hero_pos["z"]))
        r6 = 0.0
        if self.last_hero_pos is not None and cur_pos == self.last_hero_pos:
            r6 = float(STATIONARY_PENALTY) * (3.0 if not any_monster_visible else 1.0)

        # ---- r7: 逃跑深度变化 ----
        # 移到8方向平均可跑距离更大的位置=正，鼓励去开阔区域
        mean_ed = float(np.mean(escape_depth))
        r7 = ESCAPE_DEPTH_DELTA_COEF * (mean_ed - self.last_mean_escape_depth)

        # ---- r8: 怪物安全分变化 ----
        # 移到"考虑怪物后"平均安全分更高的位置=正
        mean_me = float(np.mean(monster_escape))
        r8 = MONSTER_ESCAPE_DELTA_COEF * (mean_me - self.last_escape_score)

        # ---- r9: 死角惩罚 ----
        # 站在只有≤1个出口的位置就扣 0.08
        # V4 修复：在死胡同 (dead_end_score > 0.3) 时豁免，避免和 r_dead_end 重复扣分
        # 原因：死胡同通常伴随 dead_corner_flag=1，双重惩罚最高可达 -0.18/步，
        #       可能盖过 r_flash_escape_trap (+0.35) 的正向信号
        dead_corner_flag = float(topology_feat[2])
        if float(dead_end_info[0]) > 0.3:
            r9 = 0.0  # 死胡同场景交给 r_dead_end 专门处理
        else:
            r9 = DEAD_CORNER_PENALTY * dead_corner_flag

        # ---- r10: 危险时方向选择奖励/惩罚 ----
        # 怪物距离<阈值时：朝安全方向跑=加分，朝怪物方向跑=扣分
        r10 = 0.0
        # nearest_align 已在特征段提前计算，此处直接复用
        la = int(last_action) if last_action is not None else -1
        if la >= 0 and cur_min_dist_norm < MONSTER_CLOSE_DIST_NORM:
            dir_idx = la % 8
            r10 = RETREAT_DIR_COEF * float(monster_escape[dir_idx])
            # V4：朝怪方向惩罚仅对移动动作生效（la<8）。闪现跳 10 格可能越过怪物，
            # 不应该被"方向朝怪"一刀切惩罚，否则会压制正确的朝怪闪现脱险行为。
            if la < 8:
                r10 -= ALIGN_TOWARD_MONSTER_PENALTY * max(0.0, float(nearest_align[dir_idx]))

        # ---- r_memory: 记忆逃离奖励 ----
        # 怪物不在视野但记忆有效时，朝远离记忆中怪物方向移动=正奖励
        # 衰减因子让越新鲜的记忆影响越大
        r_memory = 0.0
        if not any_monster_visible and la >= 0:
            dir_idx = la % 8
            ux, uz = _unit_vec(*WORLD_DIR_VECS[dir_idx])
            for i in range(2):
                if self.memory_monster_positions[i] is not None:
                    age = self.step_no - self.memory_monster_last_seen[i]
                    if age <= MEMORY_DECAY_STEPS:
                        decay = 1.0 - float(age) / float(MEMORY_DECAY_STEPS)
                        fx, fz = self.memory_monster_flee_dir[i]
                        align = max(0.0, ux * fx + uz * fz)  # 行动方向与逃离方向的一致性
                        r_memory += MEMORY_FLEE_COEF * decay * align

        # ---- r_explore: 无怪探索奖励（移植 code：合并 r_curiosity，新格大奖）----
        # 设计：首次到达新格 = EXPLORE_MOVE_REWARD(0.04) + R_CURIOSITY(0.10) = 0.14（大奖）
        # 朝宝箱方向的额外奖（即使不是新格也给，因为方向正确就该奖）
        # 目的：让模型主动去走没去过的格子，而不是在原地附近绕圈刷靠近奖
        r_explore = 0.0
        if not any_monster_visible and self.last_hero_pos is not None:
            if cur_pos != self.last_hero_pos:
                if cur_pos not in self.visited_cells:
                    r_explore = EXPLORE_MOVE_REWARD + float(getattr(Config, "R_CURIOSITY", 0.10))
                if raw_treasure_dist < 998.0 and treasure_top3_feat[0] > 0:
                    t_dx = treasure_top3_feat[2] * 2.0 - 1.0
                    t_dz = treasure_top3_feat[3] * 2.0 - 1.0
                    t_ux, t_uz = _unit_vec(t_dx, t_dz)
                    move_dx = float(cur_pos[0] - self.last_hero_pos[0])
                    move_dz = float(cur_pos[1] - self.last_hero_pos[1])
                    move_ux, move_uz = _unit_vec(move_dx, move_dz)
                    toward_treasure = max(0.0, move_ux * t_ux + move_uz * t_uz)
                    r_explore += EXPLORE_TOWARD_TREASURE * toward_treasure

        # ---- r_dead_end: 死胡同引导奖励/惩罚 ----
        # 在死胡同中：朝出口走=正，走错方向=负，停着不走=负，越走越深=负
        # 开局前30步如果在死胡同，惩罚翻倍（逼模型快速逃生）
        dead_end_score = float(dead_end_info[0])
        exit_dirs = dead_end_info[1:9]
        r_dead_end = 0.0
        if dead_end_score > 0.3:
            r_dead_end += DEAD_END_STAY_PENALTY * dead_end_score     # 停留惩罚
            toward_exit = False
            if la >= 0:
                dir_idx = la % 8
                if exit_dirs[dir_idx] > 0.5:
                    r_dead_end += DEAD_END_EXIT_REWARD * dead_end_score   # 朝出口=奖
                    toward_exit = True
                else:
                    r_dead_end += DEAD_END_WRONG_DIR_PENALTY * dead_end_score  # 走错=罚
                if escape_depth[dir_idx] < 0.2:
                    r_dead_end += DEAD_END_APPROACH_PENALTY  # 该方向快到头了=罚
            # V4 修复：朝出口走的时候 mean_ed 可能因"前方是开阔区但当前格还在死胡同内"
            # 而下降，此时不应罚"越走越深"，否则会误伤正确行为
            if mean_ed < self.last_mean_escape_depth and not toward_exit:
                r_dead_end += DEAD_END_DEEPER_PENALTY  # 越走越深=罚

        if self.birth_escape_mode:
            self.birth_escape_steps += 1
            if dead_end_score > 0.3:
                r_dead_end += -0.05  # V5：翻倍改加常量，避免被压力加权再放大
            else:
                self.birth_escape_mode = False  # 已离开死胡同，关闭逃生模式
            if self.birth_escape_steps >= BIRTH_ESCAPE_MAX_STEPS:
                self.birth_escape_mode = False

        # ---- r_flash_wall: 闪现穿墙奖励（按加速点 SPEEDUP_THRESHOLD 分段）----
        # 上一步使用了闪现且穿过了墙壁 → 基础 +0.10
        # 阈值从加速点(默认 300)反推：加速前 100 步窗口×2，加速后×4，更早保持基础。
        #   step <  speedup-100  (默认 <200)  → ×1（保持稳，避免过早乱闪）
        #   speedup-100..speedup (200..300)   → ×2（加速前 100 步，提前抢闪现穿墙路径）
        #   step >= speedup      (>=300)      → ×4（加速后穿墙已是主要逃生/抄近路手段）
        r_flash_wall = 0.0
        if la >= 8:
            flash_dir = la - 8
            if flash_through[flash_dir] > 0.3:
                _speedup_thr = int(getattr(Config, "SPEEDUP_THRESHOLD", 300))
                _pre_speedup_window = 100  # 加速前的"准备窗口"长度
                if self.step_no >= _speedup_thr:
                    stage_multiplier = 4.0
                elif self.step_no >= _speedup_thr - _pre_speedup_window:
                    stage_multiplier = 2.0
                else:
                    stage_multiplier = 1.0
                r_flash_wall = FLASH_WALL_REWARD * stage_multiplier

        # ---- r_flash_escape_trap: 被困闪现脱险奖励 ----
        # 场景：死胡同被怪物堵（is_trapped=1）+ 该方向闪现落点安全 + 模型实际选了该方向
        # 强度大（+0.35）且独立于生存组/拿分组加权，确保极端危险时仍有强正向信号
        r_flash_escape_trap = 0.0
        if la >= 8:
            flash_dir = la - 8
            if trap_status[0] > 0.5 and float(flash_to_safe[flash_dir]) > 0.3:
                r_flash_escape_trap = 0.35

        # ---- r_flash_cross_monster: 跨怪物闪现客观收益奖励（V4 新增）----
        # 客观指标：闪现后实际距离怪物拉开了多少格，不管方向是否"朝向怪物"
        # 场景：鲁班贴脸怪物 → 朝怪闪 10 格 → 越过怪物 → 落到怪后方远处
        # gain = (d_after - d_before) / 10.0，归一化到 [-1,1]
        # gain > 0.3（拉开 >3 格）才给奖励，强度随 gain 线性放大
        # 此项不受任何组权重压缩，对抗 r10 的朝怪惩罚和 r_flash_effect 的贴脸门槛
        r_flash_cross_monster = 0.0
        if la >= 8 and any_monster_visible:
            flash_dir = la - 8
            gain = float(flash_dist_gain[flash_dir])
            if gain > 0.3:
                r_flash_cross_monster = 0.4 * min(gain, 1.0)

        # ---- r_flash_effect: 闪现效果奖励/惩罚（方案A：收紧触发条件）----
        # 规则：
        #   闪现前贴脸 (last_min_dist<0.25) + 闪现后拉开 >0.05  → +0.12 脱险奖励
        #   闪现前不危险 (last_min_dist>=0.25) 却闪            → -0.08 浪费惩罚（核心收紧点）
        #   闪现前贴脸但距离没改善 + 怪物仍在视野              → -0.08 浪费惩罚
        # 目的：只有"真正贴脸时用闪现拉开"才有正奖励，避免 CD 到就用
        # V4 修复：r_flash_cross_monster > 0 时不罚浪费。
        #         cross_monster 已用客观"净距离改善"判定为有效闪现（如跨怪物闪+抄近路），
        #         不应再被"闪前不贴脸"的条件误罚。
        r_flash_effect = 0.0
        if la >= 8:
            dist_improvement = cur_min_dist_norm - self.last_min_monster_dist_norm
            was_dangerous = self.last_min_monster_dist_norm < 0.25  # 闪现前已贴脸
            if was_dangerous and dist_improvement > 0.05:
                r_flash_effect = FLASH_ESCAPE_REWARD           # 贴脸 + 成功拉开 = 脱险
            elif r_flash_cross_monster > 0.0:
                r_flash_effect = 0.0                            # 客观有效闪现，不罚浪费
            elif not was_dangerous:
                r_flash_effect = FLASH_WASTE_PENALTY           # 不危险却闪 = 浪费
            elif dist_improvement < 0.01 and any_monster_visible:
                r_flash_effect = FLASH_WASTE_PENALTY           # 贴脸但没拉开距离 = 浪费

        # ---- r_corridor: 走廊/开阔度正向奖励 ----
        # 当前位置 5×5 区域开阔率>0.6 时给正奖励
        # 鼓励模型主动选择开阔区域拉扯，而不只是被动逃跑
        r_corridor = 0.0
        open_ratio = float(topology_feat[0])
        if open_ratio > 0.6:
            r_corridor = CORRIDOR_REWARD_COEF * (open_ratio - 0.6) / 0.4

        # ---- r_encircle: 两怪包夹惩罚 ----
        # 两只怪物分布在英雄两侧（夹角>120度=2.094弧度）时触发
        # 距离越近惩罚越重，鼓励模型在被合围前及时突破
        r_encircle = 0.0
        visible_monster_pos = []
        for m in monsters[:2]:
            if int(m.get("is_in_view", 0)):
                mp = m["pos"]
                visible_monster_pos.append((float(mp["x"]), float(mp["z"])))
        if len(visible_monster_pos) == 2:
            m1x, m1z = visible_monster_pos[0]
            m2x, m2z = visible_monster_pos[1]
            v1x, v1z = m1x - hx_f, m1z - hz_f  # 英雄→怪1向量
            v2x, v2z = m2x - hx_f, m2z - hz_f  # 英雄→怪2向量
            len1 = float(np.hypot(v1x, v1z))
            len2 = float(np.hypot(v2x, v2z))
            if len1 > 1e-3 and len2 > 1e-3:
                cos_angle = float(np.clip((v1x * v2x + v1z * v2z) / (len1 * len2), -1.0, 1.0))
                angle = float(np.arccos(cos_angle))
                if angle > 2.094:  # 120度 = 两怪在英雄两侧
                    avg_dist = (len1 + len2) / 2.0
                    proximity_factor = max(0.0, 1.0 - avg_dist / 20.0)  # 越近越严重
                    r_encircle = ENCIRCLEMENT_PENALTY * proximity_factor

        # ---- r_speedup_buffer: 临近加速缓冲惩罚 ----
        # 怪物马上要加速了(countdown<15%)但模型还贴着怪物 → 惩罚
        # 逼模型在高压阶段来临前提前拉开安全距离
        r_speedup_buffer = 0.0
        if is_speedup < 0.5 and 0 < speedup_countdown_norm < 0.15:
            if cur_min_dist_norm < 0.5:
                urgency = (0.15 - speedup_countdown_norm) / 0.15  # 越临近越紧急
                closeness = (0.5 - cur_min_dist_norm) / 0.5       # 越近越严重
                r_speedup_buffer = -PRE_SPEEDUP_BUFFER_COEF * urgency * closeness

        # ---- r_wall_collision: 撞墙惩罚 ----
        # 上一步选了移动方向（0~7）但位置没变 = 撞墙，动作无效
        # 比原地惩罚更具体，专门针对"选错方向"的行为
        r_wall_collision = 0.0
        if self.last_hero_pos is not None and cur_pos == self.last_hero_pos and 0 <= la < 8:
            r_wall_collision = WALL_COLLISION_PENALTY

        # ---- r_repeat: 重复探索惩罚 ----
        # V3记忆版：最近100步内访问超过1次就触发（原为>2次），归一化系数/8（原/10）
        # 效果：对"去过 2 次以上"的位置更早发出警告，迫使模型尽量走新路径
        # V4：死胡同豁免——死胡同里必须原路返回，不能因"回头"被罚，否则与逃生目标冲突
        r_repeat = 0.0
        if len(self.position_history) > 5 and float(dead_end_info[0]) < 0.3:
            repeat_count = self.position_history.count(cur_pos)
            if repeat_count > 1:
                r_repeat = REPEAT_EXPLORE_PENALTY * min(1.0, repeat_count / 8.0)

        # ---- r_drift: 漂移率惩罚（移植 code：系数从 -0.06 加重到 -0.12）----
        # 最近 200 步窗口里走过的唯一格子数 / 窗口长度 = coverage_rate
        # coverage 低 = 在小范围绕圈，无怪且非死胡同时按缺口扣分
        r_drift = 0.0
        if (not any_monster_visible
                and float(dead_end_info[0]) < 0.3
                and len(self.long_position_history) >= 100):
            if coverage_rate < 0.15:
                r_drift = -0.12 * (0.15 - coverage_rate) / 0.15

        # ---- r_late_drift_hard: 加速期 coverage 硬约束（移植 code）----
        # 加速点(默认 300)之后，如果 coverage_rate 跌到阈值(0.20)以下还在绕圈，
        # 按"缺口/阈值"线性扣分。系数 -0.40 是软 r_drift 的 3 倍多，强迫模型走出绕圈窝。
        r_late_drift_hard = 0.0
        _sp_thresh = int(getattr(Config, "SPEEDUP_THRESHOLD", 300))
        _cov_thresh = float(getattr(Config, "LATE_DRIFT_COVERAGE_THRESHOLD", 0.20))
        if (self.step_no >= _sp_thresh
                and float(dead_end_info[0]) < 0.3
                and len(self.long_position_history) >= 80
                and coverage_rate < _cov_thresh):
            r_late_drift_hard = float(getattr(Config, "LATE_DRIFT_HARD_COEF", -0.40)) * (
                _cov_thresh - coverage_rate
            ) / _cov_thresh

        # ---- r_border_approach: 朝边界探索奖（移植 code）----
        # 加速期 + coverage 低时生效：每次靠近最近边界 0.5 格以上 = 给 0.10 奖。
        # 把模型从中心地推向边缘，扩大全图覆盖。
        r_border_approach = 0.0
        if (self.step_no >= _sp_thresh
                and coverage_rate < float(getattr(Config, "BORDER_APPROACH_COV_THRESHOLD", 0.40))):
            dist_right  = MAP_SIZE - float(hero_pos["x"])
            dist_left   = float(hero_pos["x"])
            dist_bottom = MAP_SIZE - float(hero_pos["z"])
            dist_top    = float(hero_pos["z"])
            min_border_dist = min(dist_right, dist_left, dist_bottom, dist_top)
            if self._last_min_border_dist < 999.0:
                if min_border_dist < self._last_min_border_dist - 0.5:
                    r_border_approach = float(getattr(Config, "BORDER_APPROACH_REWARD", 0.10))
            self._last_min_border_dist = min_border_dist
        else:
            self._last_min_border_dist = 999.0

        # ---- r_second_monster: 第二只怪压力惩罚 ----
        # 只盯最近那只怪是不够的——第二只怪如果也贴近，活动空间被严重压缩
        r_second_monster = 0.0
        if len(monster_feats) >= 2 and monster_feats[1][0] > 0:
            second_dist = float(monster_feats[1][4])
            if second_dist < 0.4:
                r_second_monster = SECOND_MONSTER_CLOSE_PENALTY * (0.4 - second_dist) / 0.4

        # ---- r_anti_repeat: 朝高访问次数方向行动的额外惩罚 ----
        # V3记忆版新增：利用已有的 novelty_8dir 特征（1/(1+访问次数)），
        # 当模型选择的移动方向的相邻格访问次数高时（陌生度低）额外惩罚。
        # 与 r_repeat 互补：r_repeat 惩罚"到达已去过的地方"，
        # r_anti_repeat 惩罚"选择了走向已去过地方的动作方向"（更早一步干预）。
        # 只在无怪/移动动作时触发，避免干扰危险逃跑时的方向选择。
        # V4：同样死胡同豁免（朝已去过方向走本来就是死胡同回头唯一出路）
        r_anti_repeat = 0.0
        if not any_monster_visible and 0 <= la < 8 and float(dead_end_info[0]) < 0.3:
            dir_novelty = float(novelty_8dir[la])  # 1.0=从未去过, 0=去过很多次
            # 移植 code：系数从 -0.04 加重到 -0.08，配合 r_drift 双重反绕圈
            if dir_novelty < 0.5:
                r_anti_repeat = -0.08 * (0.5 - dir_novelty) / 0.5

        # ---- r_easy_treasure: 近距离宝箱激励（delta 型）----
        # 宝箱在 4 格以内且怪物不贴脸时，只在"正在靠近"时给信号，
        # 避免模型学到站在宝箱旁边蹲着刷奖励的懒行为。
        # 此项不受 danger_scale/treasure_weight 压缩，确保模型在安全距离内敢吃宝箱
        r_easy_treasure = 0.0
        if raw_treasure_dist < 4.0 and cur_min_dist_norm > 0.08:
            if raw_treasure_dist < self._last_near_treasure_dist:
                # 正在靠近宝箱：按距离减少量给奖励（距离每缩短1格≈0.02）
                r_easy_treasure = 0.08 * (self._last_near_treasure_dist - raw_treasure_dist) / 4.0
            self._last_near_treasure_dist = raw_treasure_dist
        else:
            # 离开 4 格圈或怪物贴脸：重置到圈边界 4.0，下次进入圈内 delta 从 0 开始累积
            self._last_near_treasure_dist = 4.0

        # ================================================================
        #              三组奖励合成（取消压力加权，让三目标平等竞争）
        #   - 旧版用 w_survive(0~2.5)/w_collect(0.3~1)/w_explore(0.4~1) 乘到三组上，
        #     一旦压力上来 collect/explore 就被压死，模型学成"躲怪不拿分"。
        #   - 现在压力分数(pressure_score)只作为输入特征喂给 critic，不再影响奖励。
        #   - 加 r_post_speedup_alive: 加速点(默认 300 步)之后每步给少量正奖，告诉模型
        #     "活过加速点是好事"，避免后期 critic 学不到分。
        # ================================================================
        # 加速期生存信号（300 步后每步 +0.01）
        if self.step_no >= int(getattr(Config, "SPEEDUP_THRESHOLD", 300)):
            r_post_speedup_alive = float(getattr(Config, "R_POST_SPEEDUP_ALIVE_BASE", 0.01))
        else:
            r_post_speedup_alive = 0.0

        # survive 组（独立大项 r_flash_escape_trap / r_flash_cross_monster 不再额外加权，与其他项平等累加）
        survive_sum = (
            r1 + r2 + r7 + r8 + r9 + r10
            + r_memory + r_dead_end + r_flash_wall + r_flash_effect
            + r_corridor + r_encircle + r_speedup_buffer + r_second_monster
            + r_flash_escape_trap + r_flash_cross_monster
            + r_post_speedup_alive
        )

        # collect 组
        collect_sum = r3 + r4 + r5 + r_buff_approach + r_easy_treasure

        # explore 组（新增 r_late_drift_hard / r_border_approach）
        explore_sum = (
            r6 + r_explore + r_wall_collision + r_repeat + r_anti_repeat
            + r_drift + r_late_drift_hard + r_border_approach
        )

        # 各自 clip（默认 ±5，比旧版 ±1 宽 5 倍，让稀疏大奖能传到 critic）
        clip_s = float(getattr(Config, "REWARD_CLIP_SURVIVE", 5.0))
        clip_c = float(getattr(Config, "REWARD_CLIP_COLLECT", 5.0))
        clip_e = float(getattr(Config, "REWARD_CLIP_EXPLORE", 5.0))
        r_survive = float(np.clip(survive_sum, -clip_s, clip_s))
        r_collect = float(np.clip(collect_sum, -clip_c, clip_c))
        r_explore_w = float(np.clip(explore_sum, -clip_e, clip_e))
        reward = [r_survive, r_collect, r_explore_w]

        # ---- 监控：reward breakdown（供 train_workflow 聚合上报）----
        self.last_reward_breakdown = {
            "r_survive": r_survive,
            "r_collect": r_collect,
            "r_explore": r_explore_w,
            "r_flash_wall": float(r_flash_wall),
            "r_flash_effect": float(r_flash_effect),
            "r_flash_escape_trap": float(r_flash_escape_trap),
            "r_flash_cross_monster": float(r_flash_cross_monster),
            "r_dead_end": float(r_dead_end),
            "r_treasure_pickup": float(r4),
            "r_buff_pickup": float(r5),
            "r_post_speedup_alive": float(r_post_speedup_alive),
            # 探索类监控（关键：看模型是不是真的在涨新格率）
            "r_explore_raw": float(r_explore),
            "r_drift": float(r_drift),
            "r_late_drift_hard": float(r_late_drift_hard),
            "r_border_approach": float(r_border_approach),
            "is_new_cell": float(is_new_cell),
            "coverage_rate": float(coverage_rate),
            "pressure": float(pressure_score),
        }

        # ---- 监控：闪现质量统计 ----
        if la >= 8:
            self.flash_use_count += 1
            flash_dir = la - 8
            if flash_through[flash_dir] > 0.3:
                self.flash_wall_count += 1
            if r_flash_effect < 0:
                self.flash_waste_count += 1
            if r_flash_escape_trap > 0:
                self.flash_escape_trap_count += 1
            # flash_cd 分桶（用 self.last_flash_cd_norm 表示闪前的 cd；本步 flash_cd_norm 闪后会重置）
            prev_cd = getattr(self, "_prev_flash_cd_norm", 0.0)
            if prev_cd <= 1e-3:
                self.flash_cd_bucket[0] += 1
            elif prev_cd < 0.2:
                self.flash_cd_bucket[1] += 1
            else:
                self.flash_cd_bucket[2] += 1
            # 闪现后距离变化
            self.flash_post_dist_change_sum += float(cur_min_dist_norm - self.last_min_monster_dist_norm)

        # ---- 监控：前后期统计（以加速点 SPEEDUP_THRESHOLD 为分界）----
        # 旧版 500/1000 阈值在 max_step=1000 时永远进不到 elif 分支，导致 late_* 永远是 0。
        step_total_reward = r_survive + r_collect + r_explore_w
        speedup_thr = int(getattr(Config, "SPEEDUP_THRESHOLD", 300))
        if self.step_no < speedup_thr:
            self.early_reward_sum += step_total_reward
            self.early_treasure += pickup_delta
        else:
            self.late_reward_sum += step_total_reward
            self.late_treasure += pickup_delta

        # ================================================================
        #                    更新跨步状态（供下一步用）
        # ================================================================
        self.last_min_monster_dist_norm = cur_min_dist_norm
        self.last_min_treasure_dist_norm = cur_treasure_dn
        self.last_min_buff_dist_norm = cur_buff_dn
        self.last_treasure_collected = collected
        self.last_buff_collected = buff_collected
        self.last_mean_escape_depth = mean_ed
        self.last_escape_score = mean_me
        self.last_dead_corner_flag = dead_corner_flag
        self.last_local_open_ratio = float(topology_feat[0])
        if self.last_hero_pos is not None and cur_pos == self.last_hero_pos:
            self.stuck_step_count += 1
        else:
            self.stuck_step_count = 0
        self.last_hero_pos = cur_pos
        self.position_history.append(cur_pos)  # 记录位置历史（用于重复探索检测）
        # 更新访问次数表，供下一步计算 novelty_8dir
        self.visit_count_map[cur_pos] = self.visit_count_map.get(cur_pos, 0) + 1
        # V4 反绕圈状态更新
        self.visited_cells.add(cur_pos)                 # 本局所有访问过的格子
        self.long_position_history.append(cur_pos)      # 200 步大窗口位置历史
        try:
            self.last_buff_remaining_time = int(hero.get("buff_remaining_time", 0))
        except (TypeError, ValueError):
            self.last_buff_remaining_time = 0
        self.last_treasure_count = collected

        # V5：压力历史 + 闪现前 cd 缓存
        self.reach_history.append(cur_reach)
        self.mean_escape_history.append(cur_mean_escape)
        self._prev_flash_cd_norm = float(flash_cd_norm)

        # 视野切换跟踪
        self.last_any_monster_visible = bool(any_monster_visible)
        self.last_treasure_visible = bool(treasure_visible_now)

        return feature, legal_action, reward
