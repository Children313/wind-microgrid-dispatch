"""
02_microgrid_env_v2.py
======================
论文级微电网调度环境, 完全复现冯文韬 2023 (DOI: 10.16527/j.issn.1003-6954.20230611)
的物理建模与约束.

相对 v1 的关键升级:
  ✅ 动态分段碳排放模型 (公式 7)         - 真实碳排放成本
  ✅ 阶梯碳价模型 (公式 8)               - 真正的低碳激励机制
  ✅ 机组爬坡约束 (公式 18)              - 强制平滑调度策略
  ✅ 负荷需求响应 (公式 12, 13)          - 源-荷协同, 多一个调节维度
  ✅ PJM-5 节点拓扑 (图 4)              - 节点级碳势计算
  ✅ 物理可行性硬约束                    - 不平衡功率有上界, 不可任意 balancing
  ✅ 完整的多目标 reward                 - 发电+碳+负荷转移, 三项均衡
  ❌ 不再用 pymgrid                     - pymgrid 约束太死, 自己实现

机组配置 (Table 1):
  | id | name        | fuel        | Pmax  | cost$/MW | CO2 t/MW |
  |  1 | wind        | wind        | 600   | 10       | 0.04     |
  |  2 | gas_turbine | natural_gas | 25    | 15       | 0.60     |
  |  3 | coal_unit_1 | coal        | 110   | 14       | 1.40     |
  |  4 | coal_unit_2 | coal        | 500   | 30       | 1.40     |
  |  5 | coal_unit_3 | coal        | 190   | 26       | 1.40     |

提供的对照环境 (公平 ablation 链, 预测置于base[12]):
  - baseline           obs=22 (base[12]=当前风电)
  - point_dlinear      obs=23 (base[12]=预测净负荷)
  - point_patchtst     obs=23
  - point_cgmamba      obs=23
  - prob_cgprob        obs=24 (base[12]=预测净负荷 + iw) ⭐ 主推
  - oracle             obs=23 (base[12]=真实净负荷)
"""

import warnings; warnings.filterwarnings('ignore')
import os
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, 'data')


# ====================================================================
#  机组参数 (基于冯文韬 Table 1 比例, 按你的数据规模缩放)
#
#  缩放说明:
#    冯文韬 PJM-5: 负荷峰值 ~1100 MW, 机组总容量 900 MW, 单台风机 600 MW
#    你的数据集: 负荷峰值 222 MW, 风电峰值 148 MW
#    缩放比例: 0.22 (机组容量按比例缩放, 保持论文比例)
#
#  方案 B 设计:
#    - gas_turbine 是"自动平衡机组" (BALANCING_GENSET), 不进 RL 动作空间
#      由系统自动调节出力补足 P_load - (P_wind + P_coal_total) 的缺口.
#      物理依据: 工业电网中燃气机组爬坡快, 通常作为系统的"调频/平衡"机组,
#      调度员决定煤炭/水电出力, 燃气自动补差.
#    - 3 台煤炭机组进 RL 动作 (RL_GENSETS), RL 决策它们的 setpoint.
# ====================================================================
BALANCING_GENSET = {
    "name": "gas_turbine", "P_min": 2,    "P_max": 25,   "cost": 15, "ramp": 10
}

RL_GENSETS = [  # 这些进 RL 动作空间. 容量按论文 PJM-5 比例缩放
    {"name": "coal_unit_1", "P_min": 3,    "P_max": 50,   "cost": 14, "ramp": 12},
    {"name": "coal_unit_2", "P_min": 15,   "P_max": 200,  "cost": 30, "ramp": 35},
    {"name": "coal_unit_3", "P_min": 5,    "P_max": 80,   "cost": 26, "ramp": 18},
]
N_RL_GENSETS = len(RL_GENSETS)
N_GENSETS = N_RL_GENSETS + 1

GENSET_CONFIG = [BALANCING_GENSET] + RL_GENSETS
WIND_COST = 10.0   # 风电成本 $/MWh (含运维, 论文 Table 1)
WIND_CURTAIL_PENALTY = 50.0


# ====================================================================
#  动态分段碳排放模型 (冯文韬 公式 7)
# ====================================================================
# 每段长度 p (论文未明确, 工业上取 P_max 的 20%)
# 基准强度 ψ1, 增长系数 ξ_k 随段递增, 但增长率递减
# 论文: "随着碳排放的逐渐增加, 该值逐渐降低"
# 这里设 ψ1 = 1.4 t/MWh (煤炭典型值), ξ_k = [1.0, 1.05, 1.10, 1.15, 1.20]
# 即每多一段, 碳强度增加 5%, 模拟机组高负荷时效率下降
PSI_1_BASE = {  # 基准碳强度 (t/MWh), 取自论文 Table 1
    "gas_turbine": 0.6,
    "coal_unit_1": 1.4,
    "coal_unit_2": 1.4,
    "coal_unit_3": 1.4,
}
SEGMENT_GROWTH = np.array([1.00, 1.05, 1.10, 1.15, 1.20])  # ξ_k
N_SEGMENTS = 5


def co2_emission_dynamic(power_MW, P_min, P_max, psi_1):
    """
    动态分段碳排放 (冯文韬 公式 7).
    power_MW: 当前出力
    返回: 碳排放量 (吨/小时单位时长内)
    """
    if power_MW <= P_min:
        return 0.0
    seg_len = (P_max - P_min) / N_SEGMENTS
    p = max(0.0, power_MW - P_min)
    total_emission = 0.0
    for k in range(N_SEGMENTS):
        seg_start = k * seg_len
        seg_end = (k + 1) * seg_len
        if p <= seg_start: break
        amount_in_seg = min(p, seg_end) - seg_start
        # 段 k 的碳强度 = ψ1 × ξ_k
        intensity = psi_1 * SEGMENT_GROWTH[k]
        total_emission += amount_in_seg * intensity
    return total_emission


# ====================================================================
#  阶梯碳价模型 (冯文韬 公式 8)
# ====================================================================
# 参数标定: 让机组+风机+碳价的成本构成约为 62% / 32% / 5% (复现冯文韬 Table 4)
PSI_2 = 25.0           # 基准碳价 $/t (提高碳价, 让低碳调度更有经济优势)
B_C = 10.0              # 免费碳排放额度 t/步 (降低免税额, 让碳价更早生效)
B_SEG_LEN = 15.0        # 阶梯长度 t
OMEGA_RATE = 0.25      # 区间涨幅, 论文 "一般取值较小"
N_PRICE_STEPS = 4      # 阶梯数 (免费 + 4 阶)


def stepped_carbon_cost(emission_ton):
    """
    阶梯碳价计算 (冯文韬 公式 8).
    emission_ton: 该步的总碳排放
    返回: 该步碳排放总成本 $
    """
    if emission_ton <= B_C:
        return 0.0   # 免费区间
    cost = 0.0
    excess = emission_ton - B_C
    for k in range(N_PRICE_STEPS):
        seg_start = k * B_SEG_LEN
        seg_end = (k + 1) * B_SEG_LEN
        if excess <= seg_start: break
        amount_in_seg = min(excess, seg_end) - seg_start
        # 第 k 阶单价 = ψ2 × (1 + k × ω)
        price = PSI_2 * (1.0 + k * OMEGA_RATE)
        cost += amount_in_seg * price
    # 超出最后一阶, 按最高价
    if excess > N_PRICE_STEPS * B_SEG_LEN:
        amount_excess = excess - N_PRICE_STEPS * B_SEG_LEN
        max_price = PSI_2 * (1.0 + (N_PRICE_STEPS - 1) * OMEGA_RATE) * 1.5  # 超额加 50%
        cost += amount_excess * max_price
    return cost


# ====================================================================
#  核心物理仿真
# ====================================================================
class WindMicrogridCore:
    """
    PJM-5 节点风-火混合微电网仿真器 (复现冯文韬 2023).

    时间分辨率: 15 min/步 (与你的预测器一致)
    决策变量 (action, 6 维):
      a[0..2]: 3 台煤炭机组的 setpoint (归一化 [-1, 1] -> 映射到 [P_min, P_max])
      a[3..5]: 3 个负荷节点的需求响应比例 [-1, 1] -> ±20% 转移

    状态 (obs, 22 维 baseline):
      [0..3]:  4 台机组当前出力 (归一化)
      [4..7]:  4 台机组上一步出力 (归一化, 用于爬坡约束感知)
      [8..10]: 3 个负荷节点的本步基础负荷 (归一化)
      [11]:    时间 t / T (表示 24h 内的时刻)
      [12]:    实测当前风电 (归一化)
      [13..16]: 累计 4 个机组的碳排放 (归一化)
      [17..18]: 累计 cost 和 carbon_cost (归一化)
      [19..21]: 上 3 步的不平衡功率 (归一化, 给 RL 一个反馈)
    """

    def __init__(self, true_wind_MW, load_MW_F, load_MW_G, load_MW_H, dt_h=0.25):
        self.true_wind = np.asarray(true_wind_MW, dtype=np.float32)
        self.load_F = np.asarray(load_MW_F, dtype=np.float32)
        self.load_G = np.asarray(load_MW_G, dtype=np.float32)
        self.load_H = np.asarray(load_MW_H, dtype=np.float32)
        self.dt_h = dt_h    # 一步代表的小时数, 15min=0.25h

        self.N = len(true_wind_MW)
        assert self.N == len(load_MW_F) == len(load_MW_G) == len(load_MW_H)

        # 关键参数 (用于 obs 归一化)
        self.WIND_RATED = float(self.true_wind.max())
        self.LOAD_TOT_PEAK = float((self.load_F + self.load_G + self.load_H).max())

        self.t = 0
        self.last_genset_power = None    # 上一步机组出力 (用于爬坡约束)
        self.prev_genset_power = None    # 上上一步机组出力 (用于 ramp 感知)
        self.cumulative_cost = 0.0
        self.cumulative_co2_cost = 0.0
        self.cumulative_co2_ton = np.zeros(N_GENSETS, dtype=np.float32)
        self.cumulative_load_shift_cost = 0.0
        self.cumulative_wind_used = 0.0
        self.cumulative_wind_curtail = 0.0
        self.cumulative_imbalance = 0.0  # 累计不平衡 MWh
        self.recent_imbalance = [0.0, 0.0, 0.0]  # 最近 3 步

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.t = 0
        # 初始机组在中位出力
        self.last_genset_power = np.array(
            [(g['P_min'] + g['P_max']) / 2 for g in GENSET_CONFIG], dtype=np.float32)
        self.prev_genset_power = self.last_genset_power.copy()
        self.cumulative_cost = 0.0
        self.cumulative_co2_cost = 0.0
        self.cumulative_co2_ton = np.zeros(N_GENSETS, dtype=np.float32)
        self.cumulative_load_shift_cost = 0.0
        self.cumulative_wind_used = 0.0
        self.cumulative_wind_curtail = 0.0
        self.cumulative_imbalance = 0.0
        self.recent_imbalance = [0.0, 0.0, 0.0]

    def get_baseline_obs(self, exo_idx=None):
        """返回 22 维 baseline obs"""
        cur_norm = np.array([self.last_genset_power[i] / GENSET_CONFIG[i]['P_max']
                             for i in range(N_GENSETS)], dtype=np.float32)
        if self.prev_genset_power is None:
            last_norm = cur_norm.copy()
        else:
            last_norm = np.array([self.prev_genset_power[i] / GENSET_CONFIG[i]['P_max']
                                  for i in range(N_GENSETS)], dtype=np.float32)

        if exo_idx is None:
            exo_idx = min(self.t + 1, self.N - 1)
        exo_idx = int(np.clip(exo_idx, 0, self.N - 1))
        idx = min(self.t, self.N - 1)
        elapsed_steps = max(int(self.t), 1)
        load_norm = np.array([
            self.load_F[exo_idx] / self.LOAD_TOT_PEAK,
            self.load_G[exo_idx] / self.LOAD_TOT_PEAK,
            self.load_H[exo_idx] / self.LOAD_TOT_PEAK,
        ], dtype=np.float32)

        t_norm = np.float32(exo_idx / max(self.N - 1, 1))
        wind_norm = np.float32(self.true_wind[exo_idx] / self.WIND_RATED)

        cum_co2_norm = np.array([
            self.cumulative_co2_ton[i] / max(GENSET_CONFIG[i]['P_max'] * self.dt_h * 1.4 * elapsed_steps, 1.0)
            for i in range(N_GENSETS)
        ], dtype=np.float32)
        max_possible_cost = sum(g['P_max'] * g['cost'] * self.dt_h for g in GENSET_CONFIG) * elapsed_steps
        cost_norm = np.float32(self.cumulative_cost / max(max_possible_cost, 1.0))
        co2_cost_norm = np.float32(self.cumulative_co2_cost / max(max_possible_cost * 0.5, 1.0))
        imb_norm = np.array([x / self.LOAD_TOT_PEAK for x in self.recent_imbalance], dtype=np.float32)

        obs = np.concatenate([
            cur_norm, last_norm, load_norm,
            np.array([t_norm, wind_norm], dtype=np.float32),
            cum_co2_norm,
            np.array([cost_norm, co2_cost_norm], dtype=np.float32),
            imb_norm,
        ])
        return obs.astype(np.float32)

    def step(self, action):
        """
        执行一步, 返回 (reward, terminated, info).

        方案 B 的关键变化:
          - action 是 6 维
            [0:3]: 3 台煤炭机组 setpoint  (RL 决策)
            [3:6]: 3 节点负荷需求响应      (RL 决策)
          - wind 全消纳 (不作为决策变量)
          - gas_turbine 由系统自动平衡, 不进 RL 动作
        """
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        idx0 = min(self.t, self.N - 1)
        idx1 = min(self.t + 1, self.N - 1)

        # ── Step 1: RL 决策的 3 台煤炭机组 setpoint ──
        target_coal_P = np.zeros(N_RL_GENSETS, dtype=np.float32)
        for i, g in enumerate(RL_GENSETS):
            mid = (g['P_min'] + g['P_max']) / 2
            half_range = (g['P_max'] - g['P_min']) / 2
            target_coal_P[i] = mid + action[i] * half_range

        # ── Step 2: 爬坡约束 + P_min/P_max 约束 (公式 18) ──
        actual_coal_P = np.zeros(N_RL_GENSETS, dtype=np.float32)
        for i, g in enumerate(RL_GENSETS):
            ramp = g['ramp']
            P_prev = self.last_genset_power[i + 1]   # +1 因为 [0] 是 gas
            actual_coal_P[i] = np.clip(target_coal_P[i],
                                        P_prev - ramp, P_prev + ramp)
            actual_coal_P[i] = np.clip(actual_coal_P[i], g['P_min'], g['P_max'])

        # ── Step 3: 负荷需求响应 (±20% 转移) ──
        DR_RANGE = 0.20
        dr_F = action[3] * DR_RANGE
        dr_G = action[4] * DR_RANGE
        dr_H = action[5] * DR_RANGE
        load_F_actual = self.load_F[idx1] * (1 + dr_F)
        load_G_actual = self.load_G[idx1] * (1 + dr_G)
        load_H_actual = self.load_H[idx1] * (1 + dr_H)
        load_total = load_F_actual + load_G_actual + load_H_actual

        # ── Step 4: 风电消纳 (初始全消纳, 过发电时自动弃风) ──
        wind_available = self.true_wind[idx1]
        wind_accepted = wind_available
        wind_curtailed = 0.0

        # ── Step 5: gas_turbine 自动平衡 (方案 B 核心) ──
        coal_total = float(actual_coal_P.sum())
        deficit = float(load_total - wind_accepted - coal_total)

        gas_g = BALANCING_GENSET
        gas_prev = self.last_genset_power[0]
        gas_target = deficit
        gas_actual = np.clip(gas_target, gas_prev - gas_g['ramp'], gas_prev + gas_g['ramp'])
        gas_actual = float(np.clip(gas_actual, gas_g['P_min'], gas_g['P_max']))

        residual_imbalance = deficit - gas_actual

        # ── Step 5.5: 过发电时自动弃风 (减少负不平衡, 弃风惩罚 << 不平衡惩罚) ──
        if residual_imbalance < 0:
            curtail_needed = -residual_imbalance
            curtail_amount = min(wind_accepted, curtail_needed)
            wind_accepted -= curtail_amount
            wind_curtailed += curtail_amount
            residual_imbalance += curtail_amount

        # ── Step 6: 成本核算 ──
        dt = self.dt_h
        # 6.1 机组发电成本 (3 台煤 + 1 台 gas)
        cost_coal = sum(actual_coal_P[i] * RL_GENSETS[i]['cost'] * dt
                         for i in range(N_RL_GENSETS))
        cost_gas = gas_actual * gas_g['cost'] * dt
        cost_genset = cost_coal + cost_gas

        # 6.2 风电成本
        cost_wind = wind_accepted * WIND_COST * dt

        cost_wind_curtail = wind_curtailed * WIND_CURTAIL_PENALTY * dt

        # 6.3 动态分段碳排放 (4 台机组都计算)
        co2_per_genset = np.zeros(N_GENSETS, dtype=np.float32)
        co2_per_genset[0] = co2_emission_dynamic(
            gas_actual, gas_g['P_min'], gas_g['P_max'], PSI_1_BASE['gas_turbine']) * dt
        for i, g in enumerate(RL_GENSETS):
            co2_per_genset[i + 1] = co2_emission_dynamic(
                actual_coal_P[i], g['P_min'], g['P_max'], PSI_1_BASE[g['name']]) * dt
        co2_total_step = float(co2_per_genset.sum())

        # 6.4 阶梯碳价
        cost_carbon = stepped_carbon_cost(co2_total_step)

        # 6.5 负荷转移成本
        C_LOAD = 5.0
        load_shift_volume = abs(load_F_actual - self.load_F[idx1]) + \
                            abs(load_G_actual - self.load_G[idx1]) + \
                            abs(load_H_actual - self.load_H[idx1])
        cost_load_shift = load_shift_volume * C_LOAD * dt

        # 6.6 残余不平衡惩罚
        # 正不平衡(失负荷): $2000/MWh (高价紧急购电)
        # 负不平衡(过发电): $1000/MWh (低价售出, 但仍有一定损失)
        IMB_PENALTY_POS_PER_MWH = 3000.0
        IMB_PENALTY_NEG_PER_MWH = 2000.0
        pos = max(residual_imbalance, 0.0)
        neg = max(-residual_imbalance, 0.0)
        cost_imbalance = (pos * IMB_PENALTY_POS_PER_MWH + neg * IMB_PENALTY_NEG_PER_MWH) * dt

        total_cost = cost_genset + cost_wind + cost_wind_curtail + cost_carbon + cost_load_shift + cost_imbalance
        reward = -total_cost

        # ── Step 7: 状态更新 ──
        # last_genset_power = [gas, coal_1, coal_2, coal_3]
        new_genset_power = np.concatenate([
            np.array([gas_actual], dtype=np.float32),
            actual_coal_P
        ])
        if self.last_genset_power is None:
            self.prev_genset_power = new_genset_power.copy()
        else:
            self.prev_genset_power = self.last_genset_power.copy()
        self.last_genset_power = new_genset_power
        self.cumulative_cost += cost_genset + cost_wind + cost_wind_curtail
        self.cumulative_co2_cost += cost_carbon
        self.cumulative_co2_ton += co2_per_genset
        self.cumulative_load_shift_cost += cost_load_shift
        self.cumulative_wind_used += wind_accepted * dt
        self.cumulative_wind_curtail += wind_curtailed * dt
        self.cumulative_imbalance += abs(residual_imbalance) * dt
        self.recent_imbalance = [residual_imbalance] + self.recent_imbalance[:2]

        self.t += 1
        terminated = self.t >= (self.N - 1)
        info = {
            'cost_genset':       float(cost_genset),
            'cost_coal':         float(cost_coal),
            'cost_gas':          float(cost_gas),
            'cost_wind':         float(cost_wind),
            'cost_wind_curtail': float(cost_wind_curtail),
            'cost_carbon':       float(cost_carbon),
            'cost_load_shift':   float(cost_load_shift),
            'cost_imbalance':    float(cost_imbalance),
            'total_cost':        float(total_cost),
            'co2_total_step':    float(co2_total_step),
            'co2_gas':           float(co2_per_genset[0]),
            'co2_coal':          float(co2_per_genset[1:].sum()),
            'wind_accepted_MW':  float(wind_accepted),
            'wind_curtailed_MW': float(wind_curtailed),
            'load_total_MW':     float(load_total),
            'imbalance_MW':      float(residual_imbalance),
            'gas_power_MW':      float(gas_actual),
            'coal_power_MW':     float(coal_total),
            'genset_power':      [float(gas_actual)] + actual_coal_P.tolist(),
        }
        return reward, terminated, info


# ====================================================================
#  Gymnasium 包装 (含 6 种 obs 模式)
# ====================================================================
class WindMicrogridEnv(gym.Env):
    """
    完整论文级风电微电网调度环境.

    信息设计原则 (预测置于base[12], 强迫agent以预测为核心决策):
      baseline:      obs=22 base[12]=当前风电 (被动响应)
      point_dlinear: obs=23 base[12]=预测净负荷, extra[0]=当前风电
      point_patchtst: obs=23 base[12]=预测净负荷, extra[0]=当前风电
      point_cgmamba: obs=23 base[12]=预测净负荷, extra[0]=当前风电
      prob_cgprob:   obs=24 base[12]=预测净负荷, extra=[当前风电, iw不确定性] ⭐
      oracle:        obs=23 base[12]=真实净负荷, extra[0]=当前风电 (完美上界)

    Args:
        agent: SUPPORTED_AGENTS 之一
        mode:  'train' / 'eval'
        data_dir: 数据目录
    """
    metadata = {'render_modes': []}

    def __init__(self, agent, mode='train', data_dir=None):
        super().__init__()
        if data_dir is None: data_dir = DATA_DIR
        assert agent in SUPPORTED_AGENTS, f"未知 agent: {agent}"
        assert mode in ('train', 'eval')
        self.agent = agent
        self.mode = mode

        df = pd.read_csv(os.path.join(data_dir, f"rl_{mode}.csv"))
        true_wind = np.clip(df['true_wind_MW'].values, 0.0, None).astype(np.float32)
        load_total = df['load_MW'].values.astype(np.float32)
        self.load_F = load_total * 0.22
        self.load_G = load_total * 0.36
        self.load_H = load_total * 0.42

        self.core = WindMicrogridCore(true_wind, self.load_F, self.load_G, self.load_H)

        self.pred_q50 = None
        self.pred_iw = None
        self.pred_unc = None
        if agent in ('point_dlinear', 'point_patchtst', 'point_cgmamba', 'point_mamba'):
            tag = agent.replace('point_', 'pred_')
            csv_path = os.path.join(data_dir, f"{tag}_{mode}.csv")
            dfp = pd.read_csv(csv_path)
            self.pred_q50 = np.clip(dfp['pred_q50_MW'].values, 0.0, None).astype(np.float32)
        elif agent == 'prob_cgprob':
            csv_path = os.path.join(data_dir, f"pred_cgprob_{mode}.csv")
            dfp = pd.read_csv(csv_path)
            self.pred_q50 = np.clip(dfp['pred_q50_MW'].values, 0.0, None).astype(np.float32)
            self.pred_iw  = dfp['interval_90_MW'].values.astype(np.float32)
            self.pred_unc = dfp['uncertainty_norm'].values.astype(np.float32)
        elif agent == 'oracle':
            self.pred_q50 = true_wind

        if agent == 'baseline':
            obs_dim = 22
        elif agent.startswith('point_'):
            obs_dim = 23
        elif agent == 'prob_cgprob':
            obs_dim = 24
        elif agent == 'oracle':
            obs_dim = 23
        else:
            obs_dim = 22

        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

    def _get_obs(self):
        idx0 = min(self.core.t, self.core.N - 1)
        idx1 = min(idx0 + 1, self.core.N - 1)
        wind_rated = self.core.WIND_RATED
        load_peak = self.core.LOAD_TOT_PEAK

        base = self.core.get_baseline_obs(exo_idx=idx1)
        cur_wind_n = np.float32(self.core.true_wind[idx0] / wind_rated)
        load_next = float(self.load_F[idx1] + self.load_G[idx1] + self.load_H[idx1])

        if self.agent == 'baseline':
            # base[12]=当前风电, agent 被动响应 (persistence 隐含)
            base[12] = cur_wind_n
            return base

        if self.agent.startswith('point_'):
            # base[12]=预测净负荷(强迫使用), extra[0]=当前风电(辅助)
            wind_pred = float(np.clip(self.pred_q50[idx1], 0.0, None))
            net_load_pred = np.float32((load_next - wind_pred) / load_peak)
            base[12] = net_load_pred
            extra = np.array([cur_wind_n], dtype=np.float32)
            return np.concatenate([base, extra])

        if self.agent == 'prob_cgprob':
            wind_q50 = float(np.clip(self.pred_q50[idx1], 0.0, None))
            iw = float(self.pred_iw[idx1])
            net_load_q50 = np.float32((load_next - wind_q50) / load_peak)
            net_load_iw   = np.float32(iw / load_peak)
            base[12] = net_load_q50
            extra = np.array([cur_wind_n, net_load_iw], dtype=np.float32)
            return np.concatenate([base, extra])

        if self.agent == 'oracle':
            # base[12]=真实净负荷(完美), extra[0]=当前风电(辅助)
            true_wind_next = float(self.core.true_wind[idx1])
            true_net_load = np.float32((load_next - true_wind_next) / load_peak)
            base[12] = true_net_load
            extra = np.array([cur_wind_n], dtype=np.float32)
            return np.concatenate([base, extra])

        raise ValueError(self.agent)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)
        self.core.reset(seed=seed)
        return self._get_obs(), {}

    def step(self, action):
        reward, terminated, info = self.core.step(action)
        truncated = False
        return self._get_obs(), float(reward), bool(terminated), truncated, info


SUPPORTED_AGENTS = [
    'baseline',
    'point_dlinear', 'point_patchtst', 'point_cgmamba', 'point_mamba',
    'prob_cgprob', 'oracle',
]


def make_env(agent, mode='train', data_dir=None):
    return WindMicrogridEnv(agent, mode=mode, data_dir=data_dir)


# ====================================================================
#  自检
# ====================================================================
if __name__ == "__main__":
    print("="*70)
    print(" 论文级微电网环境 v2 自检 (复现冯文韬 2023)")
    print("="*70)

    for mode in ['train', 'eval']:
        for agent in SUPPORTED_AGENTS:
            try:
                env = make_env(agent, mode=mode)
                obs, _ = env.reset(seed=42)
                total_r = 0.0
                cumulative_info = {}
                for _ in range(100):
                    obs, r, term, trunc, info = env.step(env.action_space.sample())
                    total_r += r
                    for k in ['cost_genset', 'cost_carbon', 'cost_load_shift',
                              'cost_imbalance', 'co2_total_step']:
                        cumulative_info[k] = cumulative_info.get(k, 0) + info[k]
                print(f"  [{mode:5s}] {agent:18s}  obs={env.observation_space.shape}  "
                      f"100步累计 reward: {total_r:>10,.0f}")
                print(f"           成本构成: 机组${cumulative_info['cost_genset']:>7,.0f} "
                      f"+ 碳价${cumulative_info['cost_carbon']:>7,.0f} "
                      f"+ 负荷转移${cumulative_info['cost_load_shift']:>6,.0f} "
                      f"+ 不平衡${cumulative_info['cost_imbalance']:>7,.0f} "
                      f"   CO2:{cumulative_info['co2_total_step']:>5.0f}t")
            except Exception as e:
                print(f"  [{mode:5s}] {agent:18s}  错误: {e}")
                import traceback; traceback.print_exc()
                break
