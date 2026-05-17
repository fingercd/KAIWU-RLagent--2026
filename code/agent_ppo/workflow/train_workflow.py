#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Training workflow for Gorge Chase PPO.
峡谷追猎 PPO 训练工作流。
"""

import copy
import os
import random
import time

import numpy as np
from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import SampleData, sample_process
from tools.metrics_utils import get_training_metrics
from tools.train_env_conf_validate import read_usr_conf
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery


# V5：整个 workflow 使用的 value_num（= 三头）
Config_VALUE_NUM = Config.VALUE_NUM


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    last_save_model_time = time.time()
    env = envs[0]
    agent = agents[0]

    # Read user config / 读取用户配置
    usr_conf = read_usr_conf("agent_ppo/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        logger.error("usr_conf is None, please check agent_ppo/conf/train_env_conf.toml")
        return

    episode_runner = EpisodeRunner(
        env=env,
        agent=agent,
        usr_conf=usr_conf,
        logger=logger,
        monitor=monitor,
    )

    while True:
        for g_data in episode_runner.run_episodes():
            agent.send_sample_data(g_data)
            g_data.clear()

            now = time.time()
            if now - last_save_model_time >= 1800:
                agent.save_model()
                last_save_model_time = now


class EpisodeRunner:
    def __init__(self, env, agent, usr_conf, logger, monitor):
        self.env = env
        self.agent = agent
        self.base_usr_conf = usr_conf
        self.logger = logger
        self.monitor = monitor
        self.episode_cnt = 0
        self.last_report_monitor_time = 0
        self.last_get_training_metrics_time = 0
        # eval 频率：每 40 局评估一次
        self.eval_interval = 40
        # eval 地图：与训练 toml 中的 map 列表一致（1~10）
        self.eval_maps = list(range(1, 11))
        self.last_eval_episode = 0

        # V5：跨局聚合的监控指标（每 20 秒上报一次）
        self._agg_reset()
        self._agg_last_report_time = 0

    def _collect_remain_info(self, remain_info, step):
        """每步把 Agent 带出来的 reward_breakdown / flash_stats 累积到聚合窗口。"""
        if not isinstance(remain_info, dict):
            return
        bd = remain_info.get("reward_breakdown", {})
        if isinstance(bd, dict):
            for k in self._agg_rewards:
                if k in bd:
                    self._agg_rewards[k] += float(bd[k])
        self._agg_steps += 1

    def _finalize_episode_stats(self, step, terminated, remain_info):
        """局结束时把 flash/phase 局级统计合到聚合窗口。"""
        self._agg_episodes += 1
        if terminated:
            self._agg_terminated += 1

        # 前后期死亡率：以加速点 SPEEDUP_THRESHOLD 为分界
        # 旧版 500/1000 双阈值在 max_step=1000 时永远统计不到 late_deaths。
        if terminated:
            if step < int(Config.SPEEDUP_THRESHOLD):
                self._agg_early_deaths += 1
            else:
                self._agg_late_deaths += 1

        if not isinstance(remain_info, dict):
            return

        fs = remain_info.get("flash_stats", {})
        if isinstance(fs, dict):
            self._agg_flash_use += int(fs.get("use", 0))
            self._agg_flash_wall += int(fs.get("wall", 0))
            self._agg_flash_waste += int(fs.get("waste", 0))
            self._agg_flash_escape_trap += int(fs.get("escape_trap", 0))
            cdb = fs.get("cd_bucket", [0, 0, 0])
            if isinstance(cdb, (list, tuple)) and len(cdb) == 3:
                for i in range(3):
                    self._agg_flash_cd_bucket[i] += int(cdb[i])
            self._agg_flash_post_dist_change_sum += float(fs.get("post_dist_change_sum", 0.0))

        ps = remain_info.get("phase_stats", {})
        if isinstance(ps, dict):
            self._agg_early_reward_sum += float(ps.get("early_reward", 0.0))
            self._agg_late_reward_sum += float(ps.get("late_reward", 0.0))
            self._agg_early_treasure += int(ps.get("early_treasure", 0))
            self._agg_late_treasure += int(ps.get("late_treasure", 0))

    def _flush_agg_monitor(self, last_episode_steps):
        """把聚合窗口里的指标上报到 monitor，然后重置。"""
        if self._agg_steps == 0 or self._agg_episodes == 0:
            return

        total_flash = max(self._agg_flash_use, 1)
        total_steps = max(self._agg_steps, 1)
        total_eps = max(self._agg_episodes, 1)

        data = {
            # 组 A 奖励细分（步均值）
            "r_survive_mean": round(self._agg_rewards["r_survive"] / total_steps, 4),
            "r_collect_mean": round(self._agg_rewards["r_collect"] / total_steps, 4),
            "r_explore_mean": round(self._agg_rewards["r_explore"] / total_steps, 4),
            "r_treasure_pickup_mean": round(self._agg_rewards["r_treasure_pickup"] / total_eps, 4),  # 局均
            "r_buff_pickup_mean": round(self._agg_rewards["r_buff_pickup"] / total_eps, 4),         # 局均
            "r_flash_wall_mean": round(self._agg_rewards["r_flash_wall"] / total_steps, 5),
            "r_flash_escape_trap_mean": round(self._agg_rewards["r_flash_escape_trap"] / total_steps, 5),
            "r_flash_cross_monster_mean": round(self._agg_rewards["r_flash_cross_monster"] / total_steps, 5),
            "r_dead_end_mean": round(self._agg_rewards["r_dead_end"] / total_steps, 5),
            "r_post_speedup_alive_mean": round(self._agg_rewards["r_post_speedup_alive"] / total_steps, 5),
            "pressure_feat_mean": round(self._agg_rewards["pressure"] / total_steps, 4),  # 仅作为输入特征监控

            # 探索诊断：新格率/覆盖率/反绕圈惩罚
            "r_explore_raw_mean": round(self._agg_rewards["r_explore_raw"] / total_steps, 5),
            "r_drift_mean": round(self._agg_rewards["r_drift"] / total_steps, 5),
            "r_late_drift_hard_mean": round(self._agg_rewards["r_late_drift_hard"] / total_steps, 5),
            "r_border_approach_mean": round(self._agg_rewards["r_border_approach"] / total_steps, 5),
            "new_cell_rate": round(self._agg_rewards["is_new_cell"] / total_steps, 4),  # 步均=新格步占比
            "coverage_rate_mean": round(self._agg_rewards["coverage_rate"] / total_steps, 4),

            # 组 B 闪现质量
            "flash_use_rate": round(self._agg_flash_use / total_steps, 4),
            "flash_wall_rate": round(self._agg_flash_wall / total_flash, 4),
            "flash_waste_rate": round(self._agg_flash_waste / total_flash, 4),
            "flash_escape_trap_rate": round(self._agg_flash_escape_trap / total_flash, 4),

            # 组 C 前后期（以 SPEEDUP_THRESHOLD=300 为分界）
            "early_reward": round(self._agg_early_reward_sum / total_eps, 4),
            "late_reward": round(self._agg_late_reward_sum / total_eps, 4),
            "early_treasure": round(self._agg_early_treasure / total_eps, 4),
            "late_treasure": round(self._agg_late_treasure / total_eps, 4),
            "early_death_rate": round(self._agg_early_deaths / total_eps, 4),
            "late_death_rate": round(self._agg_late_deaths / total_eps, 4),

            # 组 D 健康度
            "terminated_ratio": round(self._agg_terminated / total_eps, 4),
            "avg_episode_steps": round(self._agg_steps / total_eps, 1),  # 窗口内平均局长

            # 兼容旧 reward 总览面板
            "reward": round(
                (self._agg_rewards["r_survive"]
                 + self._agg_rewards["r_collect"]
                 + self._agg_rewards["r_explore"]) / total_steps,
                4,
            ),
            "episode_cnt": self.episode_cnt,
        }
        self.monitor.put_data({os.getpid(): data})
        self._agg_reset()

    def _agg_reset(self):
        """重置聚合窗口指标（每次上报后调用）。"""
        self._agg_rewards = {
            "r_survive": 0.0, "r_collect": 0.0, "r_explore": 0.0,
            "r_flash_wall": 0.0, "r_flash_effect": 0.0,
            "r_flash_escape_trap": 0.0, "r_flash_cross_monster": 0.0,
            "r_dead_end": 0.0, "pressure": 0.0,
            "r_treasure_pickup": 0.0, "r_buff_pickup": 0.0,
            "r_post_speedup_alive": 0.0,
            # 探索诊断
            "r_explore_raw": 0.0, "r_drift": 0.0,
            "r_late_drift_hard": 0.0, "r_border_approach": 0.0,
            "is_new_cell": 0.0, "coverage_rate": 0.0,
        }
        self._agg_steps = 0
        self._agg_flash_use = 0
        self._agg_flash_wall = 0
        self._agg_flash_waste = 0
        self._agg_flash_escape_trap = 0
        self._agg_flash_cd_bucket = [0, 0, 0]
        self._agg_flash_post_dist_change_sum = 0.0
        self._agg_early_reward_sum = 0.0
        self._agg_late_reward_sum = 0.0
        self._agg_early_treasure = 0
        self._agg_late_treasure = 0
        self._agg_episodes = 0
        self._agg_early_deaths = 0
        self._agg_late_deaths = 0
        self._agg_terminated = 0

    def _run_eval_episode(self, eval_map_id):
        """在指定地图上跑一局评估（贪心策略，不产生训练样本）。

        评估配置与训练 toml 完全一致（用户给定的目标配置）：
          treasure_count=10, buff_count=2, buff_cooldown=200,
          talent_cooldown=200, monster_interval=200, monster_speedup=300,
          max_step=1000。
        """
        eval_conf = copy.deepcopy(self.base_usr_conf)
        ec = eval_conf.get("env_conf", {})
        ec["map"] = [eval_map_id]
        ec["map_random"] = False
        ec["treasure_count"] = 10
        ec["buff_count"] = 2
        ec["buff_cooldown"] = 200
        ec["talent_cooldown"] = 200
        ec["monster_interval"] = 200
        ec["monster_speedup"] = 300
        ec["max_step"] = 1000
        eval_conf["env_conf"] = ec

        env_obs = self.env.reset(eval_conf)
        if handle_disaster_recovery(env_obs, self.logger):
            return None

        self.agent.reset(env_obs)

        done = False
        step = 0
        terminated = False
        while not done:
            obs_data, _ = self.agent.observation_process(env_obs)
            act_data = self.agent.predict(list_obs_data=[obs_data])[0]
            act = self.agent.action_process(act_data, is_stochastic=False)
            _, env_obs = self.env.step(act)
            if handle_disaster_recovery(env_obs, self.logger):
                return None
            terminated = env_obs["terminated"]
            truncated = env_obs["truncated"]
            step += 1
            done = terminated or truncated

        env_info = env_obs["observation"]["env_info"]
        total_score = float(env_info.get("total_score", 0))
        result = "FAIL" if terminated else "WIN"
        self.logger.info(
            f"[EVAL] map:{eval_map_id} steps:{step} result:{result} score:{total_score:.1f}"
        )
        return {"score": total_score, "steps": step}

    def _get_curriculum_conf(self):
        """取消随机课程学习，直接用 train_env_conf.toml 里的固定配置。

        理由：旧版每个阶段都随机化 monster_interval/monster_speedup，结果训练数据节奏混乱，
        而评估固定 monster_speedup=300，训练分布与评估分布脱节，模型学不到稳定策略。
        现在训练和评估完全对齐（均使用 toml 中的 200/300/1000）。
        地图采样仍由 toml 的 map / map_random 控制。
        """
        return copy.deepcopy(self.base_usr_conf)

    def run_episodes(self):
        """Run a single episode and yield collected samples.

        执行单局对局并 yield 训练样本。
        """
        while True:
            # Periodically fetch training metrics / 定期获取训练指标
            now = time.time()
            if now - self.last_get_training_metrics_time >= 60:
                training_metrics = get_training_metrics()
                self.last_get_training_metrics_time = now
                if training_metrics is not None:
                    self.logger.info(f"training_metrics is {training_metrics}")

            # Curriculum: 根据训练阶段动态调整环境参数
            cur_conf = self._get_curriculum_conf()

            # Reset env / 重置环境
            env_obs = self.env.reset(cur_conf)

            # Disaster recovery / 容灾处理
            if handle_disaster_recovery(env_obs, self.logger):
                continue

            # Reset agent & load latest model / 重置 Agent 并加载最新模型
            self.agent.reset(env_obs)
            self.agent.load_model(id="latest")

            # Initial observation / 初始观测处理
            obs_data, remain_info = self.agent.observation_process(env_obs)

            collector = []
            self.episode_cnt += 1
            done = False
            step = 0
            # V5：三路累积 reward（survive / collect / explore）
            total_reward_vec = np.zeros(Config_VALUE_NUM, dtype=np.float32)

            self.logger.info(f"Episode {self.episode_cnt} start")

            while not done:
                # Predict action / Agent 推理（随机采样）
                act_data = self.agent.predict(list_obs_data=[obs_data])[0]
                act = self.agent.action_process(act_data)

                # Step env / 与环境交互
                env_reward, env_obs = self.env.step(act)

                # Disaster recovery / 容灾处理
                if handle_disaster_recovery(env_obs, self.logger):
                    break

                terminated = env_obs["terminated"]
                truncated = env_obs["truncated"]
                step += 1
                done = terminated or truncated

                # Next observation / 处理下一步观测
                _obs_data, _remain_info = self.agent.observation_process(env_obs)

                # V5: reward 已是 (3,) 维
                reward = np.array(_remain_info.get("reward", [0.0, 0.0, 0.0]), dtype=np.float32)
                if reward.shape[0] != Config_VALUE_NUM:
                    # 兜底：若维度不对则补零/截断
                    tmp = np.zeros(Config_VALUE_NUM, dtype=np.float32)
                    tmp[:min(len(reward), Config_VALUE_NUM)] = reward[:Config_VALUE_NUM]
                    reward = tmp
                total_reward_vec += reward

                # V5：聚合 reward breakdown / flash stats（供监控）
                self._collect_remain_info(_remain_info, step)

                # V5 终局奖励：仅进入 survive 组
                final_reward = np.zeros(Config_VALUE_NUM, dtype=np.float32)
                if done:
                    env_info = env_obs["observation"]["env_info"]
                    total_score = env_info.get("total_score", 0)

                    if terminated:
                        final_reward[0] = -1.0
                        result_str = "FAIL"
                    else:
                        final_reward[0] = 1.0
                        result_str = "WIN"

                    self.logger.info(
                        f"[GAMEOVER] episode:{self.episode_cnt} steps:{step} "
                        f"result:{result_str} sim_score:{total_score:.1f} "
                        f"total_reward:{float(total_reward_vec.sum()):.3f} "
                        f"(s={float(total_reward_vec[0]):.2f} "
                        f"c={float(total_reward_vec[1]):.2f} "
                        f"e={float(total_reward_vec[2]):.2f})"
                    )

                # V5：Build sample frame（三头 value/reward/advantage）
                value_arr = np.array(act_data.value, dtype=np.float32).flatten()[:Config_VALUE_NUM]
                if value_arr.shape[0] < Config_VALUE_NUM:
                    pad = np.zeros(Config_VALUE_NUM - value_arr.shape[0], dtype=np.float32)
                    value_arr = np.concatenate([value_arr, pad])

                # done 字段只标 terminated（被怪杀死），不把 truncated（活到 max_step）算 done。
                # 这样 sample_process 在 truncated 时能走 bootstrap 分支用 last.value 续上 GAE，
                # 避免把"活到时间结束"当成 next_value=0 来低估长寿局。
                frame = SampleData(
                    obs=np.array(obs_data.feature, dtype=np.float32),
                    legal_action=np.array(obs_data.legal_action, dtype=np.float32),
                    act=np.array([act_data.action[0]], dtype=np.float32),
                    reward=reward,
                    done=np.array([float(terminated)], dtype=np.float32),
                    reward_sum=np.zeros(Config_VALUE_NUM, dtype=np.float32),
                    value=value_arr,
                    next_value=np.zeros(Config_VALUE_NUM, dtype=np.float32),
                    advantage=np.zeros(Config_VALUE_NUM, dtype=np.float32),
                    prob=np.array(act_data.prob, dtype=np.float32),
                )
                collector.append(frame)

                # Episode end / 对局结束
                if done:
                    if collector:
                        collector[-1].reward = collector[-1].reward + final_reward

                    # V5：聚合局级统计
                    self._finalize_episode_stats(step, terminated, _remain_info)

                    # V5：每 20 秒上报一次聚合指标（替代原每 60s 单一 reward 上报）
                    now = time.time()
                    if now - self._agg_last_report_time >= 20 and self.monitor:
                        self._flush_agg_monitor(step)
                        self._agg_last_report_time = now

                    if collector:
                        collector = sample_process(collector)
                        yield collector

                    # 定期在 1~10 号地图上跑评估，结果上报到监控面板
                    if (self.episode_cnt - self.last_eval_episode) >= self.eval_interval:
                        self.last_eval_episode = self.episode_cnt
                        for emap in self.eval_maps:
                            result = self._run_eval_episode(emap)
                            if result is not None and self.monitor:
                                self.monitor.put_data({os.getpid(): {
                                    f"eval_map{emap}_score": round(result["score"], 1),
                                    f"eval_map{emap}_steps": result["steps"],
                                }})
                    break

                # Update state / 状态更新
                obs_data = _obs_data
                remain_info = _remain_info
