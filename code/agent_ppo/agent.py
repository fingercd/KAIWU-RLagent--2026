#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Agent class for Gorge Chase PPO.
峡谷追猎 PPO Agent 主类。
"""

import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import numpy as np
from kaiwudrl.interface.agent import BaseAgent

from agent_ppo.algorithm.algorithm import Algorithm
from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import ActData, ObsData
from agent_ppo.feature.preprocessor import Preprocessor
from agent_ppo.model.model import Model

# V5 三头动作约定（与 model.py forward 返回的 18 维 logits 切片顺序一致）
_IDX_GATE = (0, 2)          # [0:2]   flash_gate: [不闪, 闪]
_IDX_MOVE = (2, 10)         # [2:10]  move_dir 8 维
_IDX_FLASH_DIR = (10, 18)   # [10:18] flash_dir 8 维


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        torch.manual_seed(0)
        self.device = device
        self.model = Model(device).to(self.device)
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=Config.INIT_LEARNING_RATE_START,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        self.algorithm = Algorithm(self.model, self.optimizer, self.device, logger, monitor)
        self.preprocessor = Preprocessor()
        self.last_action = -1
        self.logger = logger
        self.monitor = monitor
        super().__init__(agent_type, device, logger, monitor)

    def reset(self, env_obs=None):
        """Reset per-episode state.

        每局开始时重置状态。
        """
        self.preprocessor.reset()
        self.last_action = -1

    def observation_process(self, env_obs):
        """Convert raw env_obs to ObsData and remain_info.

        将原始观测转换为 ObsData 和 remain_info。
        V5：reward 是 3 维 [survive, collect, explore]；remain_info 额外带出
        reward_breakdown / flash 质量 / 前后期统计，供 train_workflow 聚合上报 monitor。
        """
        feature, legal_action, reward = self.preprocessor.feature_process(env_obs, self.last_action)
        obs_data = ObsData(
            feature=list(feature),
            legal_action=legal_action,
        )
        remain_info = {
            "reward": reward,
            "reward_breakdown": dict(self.preprocessor.last_reward_breakdown),
            "flash_stats": {
                "use": int(self.preprocessor.flash_use_count),
                "wall": int(self.preprocessor.flash_wall_count),
                "waste": int(self.preprocessor.flash_waste_count),
                "escape_trap": int(self.preprocessor.flash_escape_trap_count),
                "cd_bucket": list(self.preprocessor.flash_cd_bucket),
                "post_dist_change_sum": float(self.preprocessor.flash_post_dist_change_sum),
            },
            "phase_stats": {
                "early_reward": float(self.preprocessor.early_reward_sum),
                "late_reward": float(self.preprocessor.late_reward_sum),
                "early_treasure": int(self.preprocessor.early_treasure),
                "late_treasure": int(self.preprocessor.late_treasure),
            },
        }
        return obs_data, remain_info

    def predict(self, list_obs_data):
        """Stochastic inference for training (exploration).

        训练时随机采样动作（V5 三头采样：先采 is_flash，再采对应方向）。
        """
        feature = list_obs_data[0].feature
        legal_action = list_obs_data[0].legal_action

        logits, value, prob_18 = self._run_model(feature, legal_action)

        action = self._legal_sample(prob_18, legal_action, use_max=False)
        d_action = self._legal_sample(prob_18, legal_action, use_max=True)

        return [
            ActData(
                action=[action],
                d_action=[d_action],
                prob=list(prob_18),   # 18 维扁平概率 [gate(2), move(8), flash_dir(8)]
                value=value,           # (3,) 三头 value
            )
        ]

    def exploit(self, env_obs):
        """Greedy inference for evaluation.

        评估时贪心选择动作（利用）。
        """
        obs_data, _ = self.observation_process(env_obs)
        act_data = self.predict([obs_data])
        return self.action_process(act_data[0], is_stochastic=False)

    def learn(self, list_sample_data):
        """Train the model.

        训练模型。
        """
        return self.algorithm.learn(list_sample_data)

    def save_model(self, path=None, id="1"):
        """Save model checkpoint.

        保存模型检查点。
        """
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        state_dict_cpu = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
        torch.save(state_dict_cpu, model_file_path)
        self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        """Load model checkpoint.

        加载模型检查点。
        """
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        self.model.load_state_dict(torch.load(model_file_path, map_location=self.device))
        self.logger.info(f"load model {model_file_path} successfully")

    def action_process(self, act_data, is_stochastic=True):
        """Unpack ActData to int action and update last_action.

        解包 ActData 为 int 动作并记录 last_action。
        """
        action = act_data.action if is_stochastic else act_data.d_action
        self.last_action = int(action[0])
        return int(action[0])

    def _run_model(self, feature, legal_action):
        """Run model inference (V5 三头).

        执行模型推理，返回：
          - logits_np: (18,) 三头扁平 logits
          - value_np:  (3,) 三头价值
          - prob_18:   (18,) 三头各自 masked softmax 后的概率（gate/move/flash_dir 分别和为 1）
        """
        self.model.set_eval_mode()
        obs_tensor = torch.tensor(np.array([feature]), dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits, value = self.model(obs_tensor, inference=True)

        logits_np = logits.cpu().numpy()[0]   # (18,)
        value_np = value.cpu().numpy()[0]      # (3,)

        # V5 三头合法性切分
        legal_action_np = np.array(legal_action, dtype=np.float32)
        mask_move = legal_action_np[0:8]
        mask_flash = legal_action_np[8:16]
        # gate mask：[不闪总是合法, 有任一闪现方向合法才允许闪]
        any_flash_legal = float(mask_flash.sum() > 0)
        mask_gate = np.array([1.0, any_flash_legal], dtype=np.float32)

        # 三头各自 masked softmax
        p_gate      = self._legal_soft_max(logits_np[0:2],   mask_gate)
        p_move      = self._legal_soft_max(logits_np[2:10],  mask_move)
        p_flash_dir = self._legal_soft_max(logits_np[10:18], mask_flash)

        prob_18 = np.concatenate([p_gate, p_move, p_flash_dir]).astype(np.float32)
        return logits_np, value_np, prob_18

    def _legal_soft_max(self, input_hidden, legal_action):
        """Masked softmax (numpy). 若 legal_action 全 0 则返回均匀分布兜底。

        合法动作掩码下的 softmax（numpy 版）。
        """
        _w, _e = 1e20, 1e-5
        la = np.asarray(legal_action, dtype=np.float32)
        if la.sum() < 0.5:
            # 兜底：全不合法时返回均匀分布，避免 NaN
            n = len(input_hidden)
            return np.ones(n, dtype=np.float32) / float(n)
        tmp = input_hidden - _w * (1.0 - la)
        tmp_max = np.max(tmp, keepdims=True)
        tmp = np.clip(tmp - tmp_max, -_w, 1)
        tmp = (np.exp(tmp) + _e) * la
        return tmp / (np.sum(tmp, keepdims=True) * 1.00001)

    def _legal_sample(self, prob_18, legal_action, use_max=False):
        """V5 三头采样：先采 is_flash，再采对应方向，输出 0~15 整数动作。

        - is_flash=0 → action = move_dir_idx (0~7)
        - is_flash=1 → action = 8 + flash_dir_idx (8~15)

        参数:
            prob_18: (18,) 三头扁平概率 [gate(2), move(8), flash_dir(8)]
            legal_action: (16,) 原始合法动作 mask
            use_max: True=贪心（argmax）, False=随机采样（multinomial）
        """
        p_gate = prob_18[0:2]
        p_move = prob_18[2:10]
        p_flash_dir = prob_18[10:18]

        legal_np = np.asarray(legal_action, dtype=np.float32)
        any_flash = legal_np[8:16].sum() > 0
        any_move = legal_np[0:8].sum() > 0

        if use_max:
            # 贪心：比较 "最好的移动" 和 "最好的闪现" 哪个联合概率高
            # 联合概率 = p_gate[0/1] × 对应方向的最大概率
            best_move_p = float(p_gate[0] * np.max(p_move)) if any_move else -1.0
            best_flash_p = float(p_gate[1] * np.max(p_flash_dir)) if any_flash else -1.0
            if best_flash_p > best_move_p and any_flash:
                return 8 + int(np.argmax(p_flash_dir))
            return int(np.argmax(p_move))

        # 随机：先采 gate，对应不合法时自动回退
        gate = int(np.argmax(np.random.multinomial(1, p_gate, size=1)))
        if gate == 1 and any_flash:
            flash_idx = int(np.argmax(np.random.multinomial(1, p_flash_dir, size=1)))
            return 8 + flash_idx
        else:
            # 不用闪现 或 闪现方向全不合法
            if not any_move:
                # 极端兜底：无任何移动方向合法，强制从 flash 采
                if any_flash:
                    return 8 + int(np.argmax(np.random.multinomial(1, p_flash_dir, size=1)))
                return 0  # 最后兜底
            move_idx = int(np.argmax(np.random.multinomial(1, p_move, size=1)))
            return move_idx
