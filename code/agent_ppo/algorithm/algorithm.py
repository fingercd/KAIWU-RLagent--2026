#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

PPO algorithm implementation for Gorge Chase PPO (V5 三头架构).
峡谷追猎 PPO 算法实现（V5 三头架构）。

V5 损失组成：
  total_loss = vf_coef * (vloss_s + vloss_c + vloss_e)
             + policy_loss(adv = α*adv_s + β*adv_c + γ*adv_e)
             - beta * entropy_loss(16 维联合概率)

  三头 value：survive / collect / explore，各自独立 GAE，各自独立 clipped value loss
  三头 actor：flash_gate(2) + move_dir(8) + flash_dir(8)，合成 16 维联合概率算 ratio 和 entropy
"""

import os
import time

import torch
from agent_ppo.conf.conf import Config


# V5 三头动作 logits 切片（与 model.py forward 返回顺序一致）
_IDX_GATE_LO, _IDX_GATE_HI = 0, 2
_IDX_MOVE_LO, _IDX_MOVE_HI = 2, 10
_IDX_FDIR_LO, _IDX_FDIR_HI = 10, 18


class Algorithm:
    def __init__(self, model, optimizer, device=None, logger=None, monitor=None):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.parameters = [p for pg in self.optimizer.param_groups for p in pg["params"]]
        self.logger = logger
        self.monitor = monitor

        self.label_size = Config.ACTION_NUM
        self.value_num = Config.VALUE_NUM
        self.var_beta = Config.BETA_START
        self.vf_coef = Config.VF_COEF
        self.clip_param = Config.CLIP_PARAM
        # V5: advantage 合成权重
        self.adv_w_s = float(Config.ADV_WEIGHT_SURVIVE)
        self.adv_w_c = float(Config.ADV_WEIGHT_COLLECT)
        self.adv_w_e = float(Config.ADV_WEIGHT_EXPLORE)

        self.last_report_monitor_time = 0
        self.train_step = 0

    def learn(self, list_sample_data):
        """Training entry: PPO update on a batch of SampleData.

        训练入口：对一批 SampleData 执行 PPO 更新。
        """
        obs = torch.stack([f.obs for f in list_sample_data]).to(self.device)
        legal_action = torch.stack([f.legal_action for f in list_sample_data]).to(self.device)
        act = torch.stack([f.act for f in list_sample_data]).to(self.device).view(-1, 1)
        old_prob = torch.stack([f.prob for f in list_sample_data]).to(self.device)
        reward = torch.stack([f.reward for f in list_sample_data]).to(self.device)
        advantage = torch.stack([f.advantage for f in list_sample_data]).to(self.device)
        old_value = torch.stack([f.value for f in list_sample_data]).to(self.device)
        reward_sum = torch.stack([f.reward_sum for f in list_sample_data]).to(self.device)

        self.model.set_train_mode()
        self.optimizer.zero_grad()

        logits, value_pred = self.model(obs)

        total_loss, info_list = self._compute_loss(
            logits=logits,
            value_pred=value_pred,
            legal_action=legal_action,
            old_action=act,
            old_prob=old_prob,
            advantage=advantage,
            old_value=old_value,
            reward_sum=reward_sum,
            reward=reward,
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters, Config.GRAD_CLIP_RANGE)
        self.optimizer.step()
        self.train_step += 1
        # Beta线性衰减：探索从高到低，避免后期乱探索
        decay_span = float(max(1, int(Config.BETA_DECAY_STEPS)))
        self.var_beta = max(
            float(Config.BETA_END),
            float(Config.BETA_START)
            - float(self.train_step) * (float(Config.BETA_START) - float(Config.BETA_END)) / decay_span,
        )

        # V5：上报频率 60s → 20s，加入 raw_entropy / explained_variance / 三头 value loss 细分
        now = time.time()
        if now - self.last_report_monitor_time >= 20:
            vloss_s, vloss_c, vloss_e = info_list[3]
            raw_entropy = info_list[4].item()
            exp_var_s, exp_var_c, exp_var_e = info_list[5]
            results = {
                "total_loss": round(total_loss.item(), 4),
                "value_loss": round(info_list[0].item(), 4),
                "policy_loss": round(info_list[1].item(), 4),
                "entropy_loss": round(info_list[2].item(), 4),
                "reward": round(reward.mean().item(), 4),
                "raw_entropy": round(raw_entropy, 4),
                "value_loss_survive": round(vloss_s.item(), 4),
                "value_loss_collect": round(vloss_c.item(), 4),
                "value_loss_explore": round(vloss_e.item(), 4),
                "explained_variance_survive": round(float(exp_var_s), 4),
                "explained_variance_collect": round(float(exp_var_c), 4),
                "explained_variance_explore": round(float(exp_var_e), 4),
            }
            self.logger.info(
                f"[train] total_loss:{results['total_loss']} "
                f"policy_loss:{results['policy_loss']} "
                f"value_loss:{results['value_loss']} "
                f"entropy:{results['raw_entropy']} "
                f"ev_s:{results['explained_variance_survive']} "
                f"ev_c:{results['explained_variance_collect']} "
                f"ev_e:{results['explained_variance_explore']}"
            )
            if self.monitor:
                self.monitor.put_data({os.getpid(): results})
            self.last_report_monitor_time = now

    def _compute_loss(
        self,
        logits,
        value_pred,
        legal_action,
        old_action,
        old_prob,
        advantage,
        old_value,
        reward_sum,
        reward,
    ):
        """V5 三头 PPO loss.

        - logits: (B, 18) 三头扁平 logits
        - value_pred: (B, 3) 三头价值
        - legal_action: (B, 16)
        - old_action: (B, 1) 0~15 整数动作
        - old_prob: (B, 18) 采样时记录的三头概率
        - advantage/reward_sum/old_value/reward: (B, 3)
        """
        B = logits.size(0)

        # ---------- 合法动作切分 ----------
        mask_move = legal_action[:, 0:8]                                     # (B, 8)
        mask_flash = legal_action[:, 8:16]                                   # (B, 8)
        any_flash = (mask_flash.sum(dim=1, keepdim=True) > 0).float()        # (B, 1)
        # gate mask: [不闪总是合法, 有任一闪现方向合法才允许闪]
        mask_gate = torch.cat([torch.ones_like(any_flash), any_flash], dim=1)  # (B, 2)

        # ---------- 三头 masked softmax ----------
        p_gate       = self._masked_softmax(logits[:, _IDX_GATE_LO:_IDX_GATE_HI], mask_gate)     # (B, 2)
        p_move       = self._masked_softmax(logits[:, _IDX_MOVE_LO:_IDX_MOVE_HI], mask_move)     # (B, 8)
        p_flash_dir  = self._masked_softmax(logits[:, _IDX_FDIR_LO:_IDX_FDIR_HI], mask_flash)    # (B, 8)

        # ---------- 合成 16 维联合概率 ----------
        # joint_prob[action=i<8]   = p_gate[:,0] * p_move[:,i]
        # joint_prob[action=8+i]   = p_gate[:,1] * p_flash_dir[:,i]
        joint_move = p_gate[:, 0:1] * p_move                                 # (B, 8)
        joint_flash = p_gate[:, 1:2] * p_flash_dir                           # (B, 8)
        joint_prob_16 = torch.cat([joint_move, joint_flash], dim=1)          # (B, 16)

        # ---------- 对应 old_prob 也合成 16 维 ----------
        old_p_gate      = old_prob[:, _IDX_GATE_LO:_IDX_GATE_HI]             # (B, 2)
        old_p_move      = old_prob[:, _IDX_MOVE_LO:_IDX_MOVE_HI]             # (B, 8)
        old_p_flash_dir = old_prob[:, _IDX_FDIR_LO:_IDX_FDIR_HI]             # (B, 8)
        old_joint_16 = torch.cat([
            old_p_gate[:, 0:1] * old_p_move,
            old_p_gate[:, 1:2] * old_p_flash_dir,
        ], dim=1)

        # ---------- Policy loss（合成联合概率做 ratio）----------
        one_hot = torch.nn.functional.one_hot(old_action[:, 0].long(), self.label_size).float()  # (B, 16)
        new_action_prob = (one_hot * joint_prob_16).sum(1, keepdim=True).clamp(1e-9)
        old_action_prob = (one_hot * old_joint_16).sum(1, keepdim=True).clamp(1e-9)
        ratio = new_action_prob / old_action_prob

        # 三头 advantage 各自归一化后再加权合成（避免某一头量级大就吃掉其它头的尺度）
        def _norm_adv(a):
            return (a - a.mean()) / a.std().clamp(1e-6)

        adv_s_n = _norm_adv(advantage[:, 0:1])
        adv_c_n = _norm_adv(advantage[:, 1:2])
        adv_e_n = _norm_adv(advantage[:, 2:3])
        adv_norm = self.adv_w_s * adv_s_n + self.adv_w_c * adv_c_n + self.adv_w_e * adv_e_n  # (B, 1)

        policy_loss1 = -ratio * adv_norm
        policy_loss2 = -ratio.clamp(1 - self.clip_param, 1 + self.clip_param) * adv_norm
        policy_loss = torch.maximum(policy_loss1, policy_loss2).mean()

        # ---------- Value loss（三头各自 clipped；用单独的 VALUE_CLIP_RANGE，不再用 0.2）----------
        v_clip_range = float(getattr(Config, "VALUE_CLIP_RANGE", 2.0))
        vloss_per_head = []
        for h in range(self.value_num):
            vp = value_pred[:, h:h + 1]
            ov = old_value[:, h:h + 1]
            tdret = reward_sum[:, h:h + 1]
            v_clip = ov + (vp - ov).clamp(-v_clip_range, v_clip_range)
            v_loss_h = 0.5 * torch.maximum(
                torch.square(tdret - vp),
                torch.square(tdret - v_clip),
            ).mean()
            vloss_per_head.append(v_loss_h)
        value_loss = vloss_per_head[0] + vloss_per_head[1] + vloss_per_head[2]

        # ---------- Entropy loss（16 维联合概率）----------
        raw_entropy = -(joint_prob_16 * torch.log(joint_prob_16.clamp(1e-9, 1))).sum(1).mean()
        entropy_loss = raw_entropy  # 与原来口径一致，BETA 退火节奏不变

        # ---------- Explained variance（供监控）----------
        with torch.no_grad():
            ev = []
            for h in range(self.value_num):
                tdret = reward_sum[:, h]
                vp = value_pred[:, h]
                var_t = tdret.var().clamp(1e-6)
                ev_h = 1.0 - (tdret - vp).var() / var_t
                ev.append(ev_h.item())

        # ---------- Total loss ----------
        total_loss = self.vf_coef * value_loss + policy_loss - self.var_beta * entropy_loss

        return total_loss, [
            value_loss,
            policy_loss,
            entropy_loss,
            tuple(vloss_per_head),   # info_list[3]: (vloss_s, vloss_c, vloss_e)
            raw_entropy,             # info_list[4]
            tuple(ev),               # info_list[5]: (ev_s, ev_c, ev_e)
        ]

    def _masked_softmax(self, logits, legal_action):
        """Masked softmax（三头通用）。

        参数:
            logits:       (B, K)
            legal_action: (B, K)，K 可以是 2 / 8 / 16
        """
        # 不合法位置设为极小 logit
        masked = logits + (legal_action - 1.0) * 1e5
        # 防止 legal_action 全 0（理论上不会，但保险）
        probs = torch.nn.functional.softmax(masked, dim=1)
        # 对不合法位置再置零（softmax 后），然后归一化兜底
        probs = probs * legal_action
        probs = probs / probs.sum(dim=1, keepdim=True).clamp(1e-9)
        return probs
