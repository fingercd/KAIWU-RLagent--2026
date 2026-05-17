#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Neural network model for Gorge Chase PPO (V5 三头架构).
峡谷追猎 PPO 神经网络模型（V5 三头架构）。

========== 核心设计思想（V5：三头 Actor + 三头 Critic）==========
按信息语义拆成 8 个独立 token（与 V4 相同）：
  0. self / 1. monster / 2. treasure / 3. buff / 4. map /
  5. escape / 6. flash / 7. explore

Actor 三头（动作分层，方案 A：每头加 32 维中间层）：
  - head_flash_gate: Linear(64,32)→ReLU→Linear(32,2)  —— 要不要用闪现
  - head_move_dir  : Linear(64,32)→ReLU→Linear(32,8)  —— 移动方向 logits
  - head_flash_dir : Linear(64,32)→ReLU→Linear(32,8)  —— 闪现方向 logits

Critic 三头（目标分层，方案 A：每头加 32 维中间层）：
  - critic_survive: Linear(64,32)→ReLU→Linear(32,1)（最后一层 bias=+1.0）
  - critic_collect: Linear(64,32)→ReLU→Linear(32,1)
  - critic_explore: Linear(64,32)→ReLU→Linear(32,1)

========== 特征向量总览（1899 维，与 preprocessor.py 拼接顺序一一对应）==========
索引范围         维度    内容                            归属 token
[0:4]              4     英雄状态(位置/闪现CD/buff)       → self (0)
[4:14]            10     怪物1+怪物2 各5维               → monster (1)
[14:26]           12     最近3个宝箱 各4维               → treasure (2)
[26:34]            8     最近2个buff 各4维               → buff (3)
[34:1798]       1764     4通道21×21地图（方案C FPN）      → map (4)
[1798:1806]        8     8方向逃跑深度                   → escape (5)
[1806:1814]        8     8方向怪物感知安全分              → escape (5)
[1814:1817]        3     局部拓扑                        → escape (5)
[1817:1825]        8     怪物记忆(2只×4维)               → monster (1)
[1825:1833]        8     8方向闪现穿墙机会               → flash (6)
[1833:1843]       10     死胡同分析(BFS版)               → escape (5)
[1843:1859]       16     合法动作掩码                    → self (0)
[1859:1862]        3     进度(步数/加速/倒计时)           → self (0)
[1862:1874]       12     探索特征(位移+陌生度+资源安全)   → explore (7)
[1874:1882]        8     flash_to_safe: 落点安全分       → flash (6)
[1882:1884]        2     trap_status: 被困状态           → flash (6)
[1884:1892]        8     flash_dist_gain: 净距离改善      → flash (6)
[1892:1894]        2     anti_loop: 新格/覆盖率           → explore (7)
[1894:1897]        3     V5 压力特征(shrink/sandwich/loss) → escape (5)
[1897:1899]        2     V5 闪现战略(value/opp_cost)      → flash (6)
"""

import torch
import torch.nn as nn

from agent_ppo.conf.conf import Config


def make_fc_layer(in_features, out_features):
    """创建一个全连接层，使用正交初始化（有助于训练初期梯度稳定）。"""
    fc = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(fc.weight.data)
    nn.init.zeros_(fc.bias.data)
    return fc


def make_conv_layer(in_ch, out_ch, kernel_size, **kwargs):
    """创建一个卷积层，使用正交初始化。"""
    conv = nn.Conv2d(in_ch, out_ch, kernel_size, **kwargs)
    nn.init.orthogonal_(conv.weight.data)
    nn.init.zeros_(conv.bias.data)
    return conv


def _mk_mlp(in_dim):
    """创建两层 MLP：in_dim → 64 → 128，用于 token 投影。"""
    return nn.Sequential(
        make_fc_layer(in_dim, 64),
        nn.ReLU(),
        make_fc_layer(64, 128),
        nn.ReLU(),
    )


class TransformerBlock(nn.Module):
    """标准 Transformer Encoder Block：MHA + FFN + LayerNorm + 残差。

    Pre-LN 结构：LayerNorm 在 MHA/FFN 之前（更稳定）。
    实际用 Post-LN（LN 在残差之后），与原始 Transformer 一致。
    """

    def __init__(self, embed_dim, num_heads, ffn_dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            make_fc_layer(embed_dim, ffn_dim),
            nn.ReLU(),
            make_fc_layer(ffn_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """x: (B, num_tokens, embed_dim)"""
        # MHA 子层 + 残差 + LN
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        # FFN 子层 + 残差 + LN
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class Model(nn.Module):
    """V5 三头架构网络：8 语义 Token + 2 层 Transformer Encoder + 三头 Actor + 三头 Critic。

    整体流程：
        obs(1899维) ──切片重组──┬── self(23)     ──→ self_net     ──→ 128 ── token0
                               ├── monster(18)  ──→ monster_net  ──→ 128 ── token1
                               ├── treasure(12) ──→ treasure_net ──→ 128 ── token2
                               ├── buff(8)      ──→ buff_net     ──→ 128 ── token3
                               ├── map(1764)    ──→ FPN CNN      ──→ 128 ── token4
                               ├── escape(32)   ──→ escape_net   ──→ 128 ── token5  (V5: +3 pressure)
                               ├── flash(28)    ──→ flash_net    ──→ 128 ── token6  (V5: +2 strategic)
                               └── explore(14)  ──→ explore_net  ──→ 128 ── token7
                                                           │
                                          Transformer × 2 (MHA 8头 + FFN 256)
                                                           │
                                             fusion 1024 → 256 → 64 (hidden)
                                                           │
                              ┌─── 三头 Actor (64→32→N) ────┬──────────────┬──────────────┐
                              │  head_flash_gate(2)  head_move_dir(8)  head_flash_dir(8)
                              │
                              └─── 三头 Critic (64→32→1) ───┬──────────────┬──────────────┐
                                 critic_survive(bias=+1.0)  critic_collect  critic_explore
    """

    # 地图通道和尺寸常量（方案 C 保持）
    _MAP_CHANNELS = 4   # Channel 0=通行性 1=怪物 2=宝箱 3=buff
    _MAP_SIDE = 21      # 21×21 完整视野

    def __init__(self, device=None):
        super().__init__()
        self.model_name = "gorge_chase_v5"
        self.device = device

        action_num = Config.ACTION_NUM   # 16（8移动+8闪现）
        # V5：value_num 从 Config.VALUE_NUM=3 取，但三头 critic 分别输出 1 维再 cat

        # ====== 从 Config.FEATURE_SPLIT_SHAPE 自动计算 slice 索引 ======
        # V5 FEATURES = [4,10,12,8,1764, 8,8,3,8,8,10,16,3,12, 8,2,8,2, 3,2] → 合计 1899
        splits = Config.FEATURE_SPLIT_SHAPE
        cs = [0]
        for s in splits:
            cs.append(cs[-1] + s)
        # cs 最后两个值: 1894, 1897, 1899

        # 各特征段的 slice（用于 forward 中切片）
        self.idx_hero         = slice(cs[0],  cs[1])    # hero: [0:4]
        self.idx_monster      = slice(cs[1],  cs[2])    # monster_feats: [4:14]
        self.idx_treasure     = slice(cs[2],  cs[3])    # treasure: [14:26]
        self.idx_buff         = slice(cs[3],  cs[4])    # buff: [26:34]
        self.idx_map          = slice(cs[4],  cs[5])    # multichannel_map: [34:1798]
        self.idx_escape_dep   = slice(cs[5],  cs[6])    # escape_depth: [1798:1806]
        self.idx_monster_esc  = slice(cs[6],  cs[7])    # monster_escape: [1806:1814]
        self.idx_topology     = slice(cs[7],  cs[8])    # topology: [1814:1817]
        self.idx_memory       = slice(cs[8],  cs[9])    # memory_feat: [1817:1825]
        self.idx_flash_thru   = slice(cs[9],  cs[10])   # flash_through: [1825:1833]
        self.idx_deadend      = slice(cs[10], cs[11])   # dead_end_info: [1833:1843]
        self.idx_legal        = slice(cs[11], cs[12])   # legal_action: [1843:1859]
        self.idx_progress     = slice(cs[12], cs[13])   # progress_feat: [1859:1862]
        self.idx_explore      = slice(cs[13], cs[14])   # exploration_feat: [1862:1874]
        self.idx_flash_safe   = slice(cs[14], cs[15])   # flash_to_safe: [1874:1882]
        self.idx_trap         = slice(cs[15], cs[16])   # trap_status: [1882:1884]
        self.idx_flash_gain   = slice(cs[16], cs[17])   # flash_dist_gain: [1884:1892]
        self.idx_antiloop     = slice(cs[17], cs[18])   # anti_loop: [1892:1894]
        self.idx_pressure     = slice(cs[18], cs[19])   # V5 pressure_feat: [1894:1897]
        self.idx_flash_strat  = slice(cs[19], cs[20])   # V5 flash_strategic: [1897:1899]

        # ====== 8 个 token 投影分支 ======

        # token0 self: hero(4) + progress(3) + legal(16) = 23
        self.self_net = _mk_mlp(23)

        # token1 monster: monster_feats(10) + memory(8) = 18
        self.monster_net = _mk_mlp(18)

        # token2 treasure: treasure_top3(12)
        self.treasure_net = _mk_mlp(12)

        # token3 buff: buff_top2(8)
        self.buff_net = _mk_mlp(8)

        # token4 map: 4 通道 21×21 FPN CNN（保持方案 C 结构）
        self.map_conv1 = nn.Sequential(
            make_conv_layer(self._MAP_CHANNELS, 32, 3, padding=1),   # (32, 21, 21)
            nn.ReLU(),
        )
        self.map_conv2 = nn.Sequential(
            make_conv_layer(32, 32, 3, stride=2, padding=1),          # (32, 11, 11)
            nn.ReLU(),
        )
        self.map_conv3 = nn.Sequential(
            make_conv_layer(32, 32, 3, stride=2, padding=1),          # (32, 6, 6)
            nn.ReLU(),
        )
        self.map_pool = nn.AdaptiveAvgPool2d(1)   # → (32, 1, 1)
        self.map_fc = nn.Sequential(
            make_fc_layer(32 * 3, 128),            # 96 → 128
            nn.ReLU(),
        )

        # token5 escape: escape_depth(8) + monster_escape(8) + topology(3) + dead_end(10) + pressure(3) = 32
        self.escape_net = _mk_mlp(32)

        # token6 flash: flash_through(8) + flash_to_safe(8) + flash_dist_gain(8) + trap(2) + strategic(2) = 28
        self.flash_net = _mk_mlp(28)

        # token7 explore: exploration_feat(12) + anti_loop(2) = 14
        self.explore_net = _mk_mlp(14)

        # ====== Token Type Embedding ======
        self._num_tokens = 8
        self._embed_dim = 128
        self.token_type_embed = nn.Embedding(self._num_tokens, self._embed_dim)
        nn.init.normal_(self.token_type_embed.weight, mean=0.0, std=0.02)

        # ====== 2 层 Transformer Encoder ======
        self.encoder = nn.ModuleList([
            TransformerBlock(embed_dim=128, num_heads=8, ffn_dim=256)
            for _ in range(2)
        ])

        # ====== Fusion MLP ======
        self.fusion = nn.Sequential(
            make_fc_layer(self._num_tokens * self._embed_dim, 256),  # 1024 → 256
            nn.ReLU(),
            make_fc_layer(256, 64),                                   # 256 → 64
            nn.ReLU(),
        )

        # ====== V5 三头 Actor（动作分层）— 方案 A：head-specific hidden ======
        # 每个 head 加一个 32 维中间层，让三头学各自的差异化表达
        # 新增参数共约 1.2 万，训练速度影响 <2%
        self.head_flash_gate = nn.Sequential(
            make_fc_layer(64, 32),
            nn.ReLU(),
            make_fc_layer(32, 2),
        )
        self.head_move_dir = nn.Sequential(
            make_fc_layer(64, 32),
            nn.ReLU(),
            make_fc_layer(32, 8),
        )
        self.head_flash_dir = nn.Sequential(
            make_fc_layer(64, 32),
            nn.ReLU(),
            make_fc_layer(32, 8),
        )

        # ====== V5 三头 Critic（目标分层）— 方案 A：head-specific hidden ======
        # survive 偏置 +1.0：先天认为活着有价值，帮助前期训练启动
        # Critic 加深对价值函数学习帮助最大（尤其 collect/explore 抽象价值）
        self.critic_survive = nn.Sequential(
            make_fc_layer(64, 32),
            nn.ReLU(),
            make_fc_layer(32, 1),
        )
        # bias +1.0 加在最后一层 Linear（Sequential 的索引 [-1]）
        nn.init.constant_(self.critic_survive[-1].bias, 1.0)

        self.critic_collect = nn.Sequential(
            make_fc_layer(64, 32),
            nn.ReLU(),
            make_fc_layer(32, 1),
        )
        self.critic_explore = nn.Sequential(
            make_fc_layer(64, 32),
            nn.ReLU(),
            make_fc_layer(32, 1),
        )

    def forward(self, obs, inference=False):
        """前向推理（V5 三头架构）。

        参数:
            obs: (batch, 1899) 的特征张量
            inference: 评估模式标记（当前未使用，保留接口）

        返回:
            logits: (batch, 18) 三头扁平 logits，切片约定：
                    [:, 0:2]   = flash_gate（2 维）
                    [:, 2:10]  = move_dir（8 维）
                    [:, 10:18] = flash_dir（8 维）
            value:  (batch, 3) 三头价值 [v_survive, v_collect, v_explore]
        """
        # ====== 第一步：从 obs 切片出各语义段 ======
        hero         = obs[:, self.idx_hero]           # (B, 4)
        monster      = obs[:, self.idx_monster]        # (B, 10)
        treasure     = obs[:, self.idx_treasure]       # (B, 12)
        buff         = obs[:, self.idx_buff]           # (B, 8)
        map_flat     = obs[:, self.idx_map]            # (B, 1764)
        escape_dep   = obs[:, self.idx_escape_dep]     # (B, 8)
        monster_esc  = obs[:, self.idx_monster_esc]    # (B, 8)
        topology     = obs[:, self.idx_topology]       # (B, 3)
        memory       = obs[:, self.idx_memory]         # (B, 8)
        flash_thru   = obs[:, self.idx_flash_thru]     # (B, 8)
        deadend      = obs[:, self.idx_deadend]        # (B, 10)
        legal        = obs[:, self.idx_legal]          # (B, 16)
        progress     = obs[:, self.idx_progress]       # (B, 3)
        explore      = obs[:, self.idx_explore]        # (B, 12)
        flash_safe   = obs[:, self.idx_flash_safe]     # (B, 8)
        trap         = obs[:, self.idx_trap]           # (B, 2)
        flash_gain   = obs[:, self.idx_flash_gain]     # (B, 8)
        antiloop     = obs[:, self.idx_antiloop]       # (B, 2)
        pressure     = obs[:, self.idx_pressure]       # (B, 3)  V5
        flash_strat  = obs[:, self.idx_flash_strat]    # (B, 2)  V5

        # ====== 第二步：拼接成 8 个 token 的输入 ======
        t_self     = torch.cat([hero, progress, legal], dim=1)                              # (B, 23)
        t_monster  = torch.cat([monster, memory], dim=1)                                    # (B, 18)
        t_treasure = treasure                                                                # (B, 12)
        t_buff     = buff                                                                    # (B, 8)
        # map 单独处理
        t_escape   = torch.cat([escape_dep, monster_esc, topology, deadend, pressure], dim=1)  # (B, 32)
        t_flash    = torch.cat([flash_thru, flash_safe, flash_gain, trap, flash_strat], dim=1) # (B, 28)
        t_explore  = torch.cat([explore, antiloop], dim=1)                                   # (B, 14)

        # ====== 第三步：各分支投影到 128 维 ======
        e_self     = self.self_net(t_self)         # (B, 128)
        e_monster  = self.monster_net(t_monster)   # (B, 128)
        e_treasure = self.treasure_net(t_treasure) # (B, 128)
        e_buff     = self.buff_net(t_buff)         # (B, 128)

        # map FPN
        map_4d = map_flat.view(-1, self._MAP_CHANNELS, self._MAP_SIDE, self._MAP_SIDE)
        f1 = self.map_conv1(map_4d)                # (B, 32, 21, 21)
        f2 = self.map_conv2(f1)                    # (B, 32, 11, 11)
        f3 = self.map_conv3(f2)                    # (B, 32, 6, 6)
        e1 = self.map_pool(f1).flatten(1)          # (B, 32)
        e2 = self.map_pool(f2).flatten(1)          # (B, 32)
        e3 = self.map_pool(f3).flatten(1)          # (B, 32)
        e_map     = self.map_fc(torch.cat([e1, e2, e3], dim=1))  # (B, 128)

        e_escape  = self.escape_net(t_escape)      # (B, 128)
        e_flash   = self.flash_net(t_flash)        # (B, 128)
        e_explore = self.explore_net(t_explore)    # (B, 128)

        # ====== 第四步：堆成 (B, 8, 128) + token type embedding ======
        tokens = torch.stack([
            e_self, e_monster, e_treasure, e_buff,
            e_map, e_escape, e_flash, e_explore,
        ], dim=1)  # (B, 8, 128)
        tokens = tokens + self.token_type_embed.weight.unsqueeze(0)

        # ====== 第五步：2 层 Transformer Encoder ======
        for block in self.encoder:
            tokens = block(tokens)   # (B, 8, 128)

        # ====== 第六步：Fusion → V5 三头 Actor + 三头 Critic ======
        fused = tokens.reshape(tokens.size(0), -1)  # (B, 1024)
        hidden = self.fusion(fused)                  # (B, 64)

        # 三头 Actor：flash_gate(2) + move_dir(8) + flash_dir(8)，按约定顺序拼成 18 维扁平 logits
        gate_logits      = self.head_flash_gate(hidden)   # (B, 2)
        move_logits      = self.head_move_dir(hidden)     # (B, 8)
        flash_dir_logits = self.head_flash_dir(hidden)    # (B, 8)
        logits = torch.cat([gate_logits, move_logits, flash_dir_logits], dim=1)  # (B, 18)

        # 三头 Critic：cat 成 (B, 3)，顺序 [survive, collect, explore]
        v_s = self.critic_survive(hidden)
        v_c = self.critic_collect(hidden)
        v_e = self.critic_explore(hidden)
        value = torch.cat([v_s, v_c, v_e], dim=1)         # (B, 3)

        return logits, value

    def set_train_mode(self):
        """切换到训练模式。"""
        self.train()

    def set_eval_mode(self):
        """切换到评估模式。"""
        self.eval()
