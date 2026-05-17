#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Monitor panel configuration builder for Gorge Chase PPO（精简版）.
峡谷追猎监控面板（精简版：剔除冗余指标，聚焦"模型在不在变好"的核心信号）。

分 6 组：
  1. 算法指标   —— loss / entropy / explained_variance（看训练是否收敛）
  2. 奖励细分   —— 三组 reward + 关键稀疏奖励 + pressure 输入特征
  3. 闪现质量   —— 闪现使用/穿墙/浪费/脱困
  4. 前后期    —— 加速点(默认 300 步)前后死亡率/奖励/拾箱
  5. 训练健康度 —— 死亡比例 + 平均局长
  6. 评估地图  —— map1~map10 各自分数 / 步数（与训练 toml 一致，固定 10 张）
"""


from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


def _add_line_panel(m, name_cn, metric_key):
    return (
        m.add_panel(name=name_cn, name_en=metric_key, type="line")
         .add_metric(metrics_name=metric_key, expr=f"avg({metric_key}{{}})")
         .end_panel()
    )


def build_monitor():
    monitor = MonitorConfigBuilder()
    m = monitor.title("峡谷追猎-精简版")

    # 1. 算法指标
    m = m.add_group(group_name="算法指标", group_name_en="algorithm")
    m = _add_line_panel(m, "总奖励 步均", "reward")
    m = _add_line_panel(m, "总损失", "total_loss")
    m = _add_line_panel(m, "策略损失", "policy_loss")
    m = _add_line_panel(m, "价值损失-总", "value_loss")
    m = _add_line_panel(m, "价值损失-survive", "value_loss_survive")
    m = _add_line_panel(m, "价值损失-collect", "value_loss_collect")
    m = _add_line_panel(m, "价值损失-explore", "value_loss_explore")
    m = _add_line_panel(m, "原始熵", "raw_entropy")
    m = _add_line_panel(m, "EV-survive", "explained_variance_survive")
    m = _add_line_panel(m, "EV-collect", "explained_variance_collect")
    m = _add_line_panel(m, "EV-explore", "explained_variance_explore")
    m = m.end_group()

    # 2. 奖励细分（关键：可直接看 r_collect 是否在涨）
    m = m.add_group(group_name="奖励细分", group_name_en="reward_breakdown")
    m = _add_line_panel(m, "r_survive 步均", "r_survive_mean")
    m = _add_line_panel(m, "r_collect 步均", "r_collect_mean")
    m = _add_line_panel(m, "r_explore 步均", "r_explore_mean")
    m = _add_line_panel(m, "拾宝箱奖 局均", "r_treasure_pickup_mean")
    m = _add_line_panel(m, "拾Buff奖 局均", "r_buff_pickup_mean")
    m = _add_line_panel(m, "加速后生存奖 步均", "r_post_speedup_alive_mean")
    m = _add_line_panel(m, "穿墙闪奖 步均", "r_flash_wall_mean")
    m = _add_line_panel(m, "脱困闪奖 步均", "r_flash_escape_trap_mean")
    m = _add_line_panel(m, "跨怪闪奖 步均", "r_flash_cross_monster_mean")
    m = _add_line_panel(m, "死胡同奖 步均", "r_dead_end_mean")
    m = _add_line_panel(m, "压力分 输入特征", "pressure_feat_mean")
    m = m.end_group()

    # 3. 探索诊断（看模型是不是真的在积极走新格）
    m = m.add_group(group_name="探索诊断", group_name_en="exploration")
    m = _add_line_panel(m, "新格步占比", "new_cell_rate")
    m = _add_line_panel(m, "覆盖率均值", "coverage_rate_mean")
    m = _add_line_panel(m, "新格奖原始值步均", "r_explore_raw_mean")
    m = _add_line_panel(m, "漂移惩罚步均", "r_drift_mean")
    m = _add_line_panel(m, "加速期硬约束步均", "r_late_drift_hard_mean")
    m = _add_line_panel(m, "靠边界奖步均", "r_border_approach_mean")
    m = m.end_group()

    # 4. 闪现质量
    m = m.add_group(group_name="闪现质量", group_name_en="flash_quality")
    m = _add_line_panel(m, "闪现使用率", "flash_use_rate")
    m = _add_line_panel(m, "闪现穿墙率", "flash_wall_rate")
    m = _add_line_panel(m, "闪现浪费率", "flash_waste_rate")
    m = _add_line_panel(m, "被困闪脱困率", "flash_escape_trap_rate")
    m = m.end_group()

    # 4. 前后期（加速点 300 前/后）
    m = m.add_group(group_name="前后期 加速点300", group_name_en="phase_split")
    m = _add_line_panel(m, "加速前奖励 局均", "early_reward")
    m = _add_line_panel(m, "加速后奖励 局均", "late_reward")
    m = _add_line_panel(m, "加速前拾箱 局均", "early_treasure")
    m = _add_line_panel(m, "加速后拾箱 局均", "late_treasure")
    m = _add_line_panel(m, "加速前死亡率", "early_death_rate")
    m = _add_line_panel(m, "加速后死亡率", "late_death_rate")
    m = m.end_group()

    # 5. 训练健康度
    m = m.add_group(group_name="训练健康度", group_name_en="health")
    m = _add_line_panel(m, "被杀比例", "terminated_ratio")
    m = _add_line_panel(m, "平均局长", "avg_episode_steps")
    m = m.end_group()

    # 6. 评估地图（1~10，与 toml 一致）
    m = m.add_group(group_name="评估-地图级", group_name_en="evaluation_per_map")
    for emap in range(1, 11):
        m = _add_line_panel(m, f"Map{emap} 得分", f"eval_map{emap}_score")
        m = _add_line_panel(m, f"Map{emap} 步数", f"eval_map{emap}_steps")
    m = m.end_group()

    return m.build()
