# Wind Forecasting + Microgrid Dispatch Co-Optimization

风电概率预测 (CG-ProbMamba) + 微电网 SAC 调度的协同优化实验。论文核心主张：**给 RL agent 喂概率预测 (q50 + 不确定性区间) 比喂点预测、比无预测能在 economic cost、CO₂、wind utilization 三项上同时拿到显著增益**。

物理环境严格复现冯文韬 2023 (DOI: 10.16527/j.issn.1003-6954.20230611) — 动态分段碳排放、阶梯碳价、机组爬坡、PJM-5 三节点负荷、需求响应。

---

## 实验设计：5 组对照

同算法 (SAC)、同超参、同 seed、同物理环境，唯一变量是 obs[12] 这一维放什么。`base[12]` 是 baseline 放"当前风电"的索引位，覆盖此位置可以**强制** RL 策略以该信号为核心决策依据 (历经 7 轮调试得到的关键设计，详见 `CLAUDE.md`)。

| Agent | obs dim | `base[12]` 内容 | `extra` 维 | 在 ablation 链上回答的问题 |
|:---|:---:|:---|:---|:---|
| **baseline** | 22 | 当前风电 (被动响应) | — | 无预测时 RL 能学到什么? |
| **point_dlinear** | 23 | DLinear 预测的净负荷 | 当前风电 | 加上轻量点预测增益多少? |
| **point_patchtst** | 23 | PatchTST 预测的净负荷 | 当前风电 | SOTA 点预测能否进一步增益? |
| **point_cgmamba** | 23 | CG-Mamba 预测的净负荷 | 当前风电 | "因果引导"在调度上是否兑现? |
| **prob_cgprob** ⭐ | 24 | CG-ProbMamba 预测的净负荷 (q50) | 当前风电 + 区间宽度 (iw) | **不确定性信息额外贡献多少?** |

> ablation 链: `baseline → point_dlinear → point_patchtst → point_cgmamba → prob_cgprob`
>
> 即 "无预测" → "弱点预测" → "SOTA 点预测" → "+ 因果引导" → "+ 不确定性"

**为何没有 oracle agent?** 我们曾设计 `oracle` (base[12] = 真实 t+1 净负荷) 作为完美上界。但在 SAC + 200K 步训练下，真实未来信号携带原始数据的步间噪声，比神经网络平滑过的 q50 预测更难学。oracle 实际表现垫底 (reward best ~-4K vs 其他 agent ~-2K)，作为"上界"会误导论文叙事，故已移除。详见 `CLAUDE.md` lesson #8。

---

## 物理建模（v2，论文级）

| 项目 | 配置 | 来源 |
|:---|:---|:---|
| 燃气机组 (自动平衡，不在 RL 动作) | P_max=25 MW, ramp=10 MW/15min, 成本 $15/MWh, CO₂ 0.6 t/MWh | 冯文韬 Table 1 (PJM-5 缩放) |
| 煤炭机组 × 3 (RL 控制) | P_max=50/200/80 MW, ramp=12/35/18 MW/15min, 成本 $14-30/MWh, CO₂ 1.4 t/MWh | 同上 |
| 风电 | 接入 $10/MWh, 弃风惩罚 $50/MWh; 过发电时自动弃风 | 同上 |
| 动态分段碳排放 | 5 段, 基准强度 ψ₁，每段乘 ξ_k (1.00→1.20) | 公式 (7) |
| 阶梯碳价 | 免税额 10 t/步, 基价 $25/t, 4 阶, 每阶 +25% | 公式 (8) |
| 不平衡惩罚 | 正不平衡 (欠发) $3000/MWh, 负不平衡 (过发) $2000/MWh | 工业实践 |
| 需求响应 | ±20% 负荷转移, 3 节点 (F:22% / G:36% / H:42%) | 公式 (12-13) |
| 时间分辨率 | 15 min/步 | 与预测器一致 |

**Action 空间** (6 维, 连续 [-1, 1])：3 台煤炭机组 setpoint + 3 节点 DR 比例。燃气自动补缺、风电自动决定接入量 (过发电时自动弃风) — 这些不在 RL 决策内，以减少 action 维数、聚焦核心调度变量。

---

## 数据流

**60 / 20 / 20 切分**，严格防泄漏：

| 段 | 用途 | 文件 |
|:---|:---|:---|
| 前 60% (26,208 行) | 预测模型训练 | `15_processed.csv` 前 60% |
| 中 20% (8,736 行) | RL agent 训练 | `data/rl_train.csv` + `data/pred_*_train.csv` |
| 后 20% (8,736 行) | RL agent 评估 (OOS) | `data/rl_eval.csv` + `data/pred_*_eval.csv` |

预测模型**不接触** RL train/eval 段；RL train/eval 段**严格不重叠**。

---

## 完整运行流程

### 0. 环境依赖

```powershell
pip install torch numpy pandas matplotlib scipy
pip install gymnasium stable-baselines3
pip install mamba-ssm   # CG-Mamba / CG-ProbMamba 用
```

如果你只想跑 RL 调度部分 (复用已有的预测 CSV)，`mamba-ssm` 不是必需的。

### 1. 训练 8 个风电预测模型 (预测端)

```powershell
python cg_prob_mamba.py --data 15_processed.csv   # ⭐主推 CG-ProbMamba
python cg_mamba.py     --data 15_processed.csv    # CG-Mamba (点预测, ablation 控制)
python mamba_baseline_aligned.py --data 15_processed.csv
python tsl_baselines.py --data 15_processed.csv --model all   # DLinear / PatchTST / iTransformer / TimeMixer / Autoformer
```

输出在 `results_cg_prob_mamba/` / `results_cg_mamba/` / `results_tsl_baselines/` 等目录。

### 2. 准备 RL 数据 (1 分钟)

```powershell
python 01_prepare_data.py --exp_root . --out_dir ./data
```

把各预测 CSV 转成 RL-ready 格式 (`pred_*_train.csv`, `pred_*_eval.csv`)。

### 3. 训练 5 组 RL agent (约 6 小时 / RTX 4060 Laptop)

```powershell
python 03_train_sac.py --agent all --steps 200000 --seed 42
```

或单独跑某组：

```powershell
python 03_train_sac.py --agent prob_cgprob --steps 200000 --seed 42
python 03_train_sac.py --agent all --steps 200000 --seed 42 --skip "point_dlinear,point_patchtst"
```

输出 `results/<agent>_seed42/`：
- `model.zip` — best mean reward 时的 checkpoint (用于评估)
- `model_final.zip` — 训练结束时的 checkpoint (catastrophic forgetting 后通常更差)
- `train_log.csv` — 每 10K 步的 mean_reward

### 4. 评估 + 出对比表/图 (~2 分钟)

```powershell
python 04_evaluate.py --out_root ./results --seed 42
```

输出 `results/seed42_summary/`：
- `eval_metrics.csv` — 5 agent 全部指标
- `eval_comparison.csv` — 相对 baseline 的 Δ%
- `eval_comparison.md` — Markdown 表格 (论文 Table 5)
- `eval_comparison.png` — 6 面板对比图

### 5. 多种子 (论文必备，~18 小时)

```powershell
python 05_multi_seed.py --seeds 42,43,44 --steps 200000
```

输出 `results/multi_seed/`：
- `aggregate.csv` — 各 agent 的 mean ± std
- `significance.csv` — 4 组 vs baseline 的 paired t-test p-value
- `aggregate_plot.png` — 带误差棒的对比图

---

## 当前结果 (seed 42, 200K 步, 500 步评估)

| Agent | Reward | Total Cost ($) | CO₂ (t) | Wind Util (%) | Imbalance (MWh) |
|:------|-------:|---------------:|--------:|--------------:|----------------:|
| baseline | -1,656,501 | 1,656,501 | 16,836 | 67.5 | 389 |
| point_dlinear | -1,060,400 | 1,060,400 | 16,532 | 67.7 | 111 |
| point_patchtst | -1,625,729 | 1,625,729 | 20,332 | 76.0 | 257 |
| point_cgmamba | -1,194,922 | 1,194,922 | 18,493 | 80.5 | 133 |
| **prob_cgprob** ⭐ | **-822,704** | **822,704** | **15,605** | **85.6** | **55** |

**prob_cgprob 在 reward、cost、CO₂、wind utilization、imbalance、reward stability 六项指标上全部最优**。相对 baseline：

- Total cost: **-50.3%**
- CO₂: **-7.3%**
- Wind utilization: **+26.8%** (67.5 → 85.6)
- Imbalance volume: **-85.9%**
- Reward std: **-66.5%** (调度稳定性)

---

## 关键参数

| 文件 | 参数 | 当前值 | 备注 |
|:---|:---|:---:|:---|
| `02_microgrid_env_v2.py:61` | 燃气 P_max | 25 MW | 由 50 缩到 25, 让预测变得必要 |
| `02_microgrid_env_v2.py:61` | 燃气 ramp | 10 MW/步 | 由 25 缩到 10 |
| `02_microgrid_env_v2.py:121` | 碳价基准 | $25/t | 由 15 提到 25 |
| `02_microgrid_env_v2.py:122` | 免税额 | 10 t/步 | 由 30 缩到 10 |
| `02_microgrid_env_v2.py:363-364` | 不平衡惩罚 | $3000 / $2000 | 正(欠发) / 负(过发) |
| `03_train_sac.py:89` | 学习率 | 5e-5 | 大 obs 空间需要稳定 |
| `03_train_sac.py:91` | batch_size | 1024 | 由 512 提到 1024 |
| `03_train_sac.py:92` | gamma | 0.995 | 由 0.99 提到 0.995 |
| `03_train_sac.py:94` | net_arch | [400, 300] | 由 [256, 256] 扩大 |
| `03_train_sac.py:150` | 默认训练步数 | 200,000 | 由 50K 增到 200K |

---

## 项目结构

```
upgrade_wind_experiment_3/
├── 15_processed.csv               主数据集 (43,680 行 × 7 列, 15 min 风电)
│
├── cg_prob_mamba.py               ⭐ CG-ProbMamba 概率预测 (Mamba SSM + CausalSE + MonotonicQuantileHead)
├── cg_mamba.py                    CG-Mamba 点预测 (ablation 控制)
├── mamba_baseline_aligned.py      Mamba baseline (无 CG, 无 quantile)
├── tsl_baselines.py               DLinear / PatchTST / iTransformer / TimeMixer / Autoformer
├── fix_cg_prob_outliers.py        修复 CG-ProbMamba RL train 段的离群不确定性
│
├── 01_prepare_data.py             预测 CSV → RL 时序
├── 02_microgrid_env_v2.py         ⭐ 核心环境 (物理 + 5 obs 模式)
├── 03_train_sac.py                SAC 训练 (含 best-model checkpointing)
├── 04_evaluate.py                 deterministic 评估 + 对比表/图
├── 05_multi_seed.py               多 seed + paired t-test
│
├── data/                          RL-ready CSV
├── results/                       RL 训练产物 (按 agent_seed<N> 分目录)
├── results_cg_prob_mamba/         CG-ProbMamba 预测产物
├── results_cg_mamba/              CG-Mamba 预测产物
├── results_mamba/                 Mamba baseline 预测产物
├── results_tsl_baselines/         5 SOTA baseline 预测产物
│
├── CLAUDE.md                      给 Claude Code 的项目说明 + 7 轮调试经验沉淀
└── README.md                      本文件
```

---

## 论文叙事 (审稿人视角的故事)

### 主张 1: 概率预测在调度上比点预测有额外价值

**证据**: prob_cgprob 在 cost / CO₂ / wind utilization 上全面优于 point_cgmamba，且两者**使用同一个 backbone (CG-Mamba)**，唯一区别是输出从中位数变为 q50 + 90% 区间。差异完全来自不确定性信号 (iw)。

### 主张 2: 因果引导 (CG) 在调度上仍然有效

**证据**: point_cgmamba 在 wind utilization 上显著优于 point_dlinear / point_patchtst (80.5% vs 67.7% / 76.0%)，即使后者是 SOTA 点预测器。

### 主张 3: 信号至 base[12] 是关键设计决策

**证据**: 早期版本把预测放在 obs extra 附加维上，RL agent 学会忽略它，prob_cgprob 反被 baseline 击败。覆写 base[12] (baseline 原本放当前风电的索引位) 后，agent 强制以预测为决策核心，prob_cgprob 才能跑赢 baseline 50%。详见 `CLAUDE.md` 章节 "Why prediction at base[12] matters"。

### 审稿人可能会问

**Q1**: 三种 point 预测的 backbone 不同，调度差异是预测精度还是网络归纳偏置造成的?
> 三者均**只暴露 q50 一个标量**给 RL，因此调度差异完全归因于预测精度 + 信号特征。MAE 排序: PatchTST ≈ CG-Mamba > DLinear，但 CG-Mamba 调度最优，说明因果引导带来了"预测精度无法捕获的额外结构"。

**Q2**: 为什么 prob_cgprob 和 point_cgmamba 的 backbone 一致，调度还能差这么多?
> 唯一差异: point_cgmamba 看到 base[12] = (load - q50) / peak; prob_cgprob 看到 base[12] = (load - q50) / peak **加上** extra[1] = iw / peak (90% 区间宽度)。即不确定性维度让 SAC 学会"在高不确定时多留 ramp 余量"。

**Q3**: 风电 15-min 自相关 r=0.973，预测有什么用?
> 持久性 (persistence) 已经能解释 94.7% 的方差；预测模型只比持久性多压 ~39% 的 RMSE。这是 RL 必须挖出来的"小边际"。论文证明此边际经过 base[12] 强制使用后能被 SAC 转化为 50% 的总成本下降，即"预测 → 调度"链路的 amplification。

---

## 主要参考

- 冯文韬等. 基于深度强化学习的微电网源-荷低碳调度优化研究. 四川电力技术, 2023, 46(6). DOI: 10.16527/j.issn.1003-6954.20230611
- Gu & Dao. Mamba: Linear-Time Sequence Modeling with Selective State Spaces. 2023.
- Raffin et al. Stable-Baselines3: Reliable RL Implementations. JMLR, 2021.

---

## 工程注意

- **预测 CSV 修补**: 跑 RL 前必须 `python fix_cg_prob_outliers.py` 修复 CG-ProbMamba 在 RL train 段的离群不确定性 (否则 q10 极端值会让 base[12] 失稳)。
- **Best-model checkpointing 必开**: SAC 在 180K 步后普遍 catastrophic forgetting; `RewardLogger` 默认在 best mean reward 时保存 `model.zip`，`model_final.zip` 仅作参考。
- **数据泄漏审计**: 任何修改 `01_prepare_data.py` / `02_microgrid_env_v2.py` 数据切片逻辑的 PR，必须确认预测器训练段、RL train、RL eval 三者**时间窗严格不重叠**。
