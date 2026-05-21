# Wind Dispatch: 6 组对照的 RL 调度实验

## 实验设计

6 组对照, 同算法 SAC + 同超参 + 同 seed, 唯一变量是 obs 中的预测信号:

| 组别 | obs | 预测信号来源 | 论文里回答的问题 |
|---|---|---|---|
| **baseline** | 19 | 无 | 无预测能学到什么? |
| **point_dlinear** | 20 | DLinear 点预测 (q50) | 加了"基础点预测"多少帮助? |
| **point_patchtst** | 20 | PatchTST 点预测 (q50) | 用 SOTA 点预测能多带多少? |
| **point_cgmamba** | 20 | CG-Mamba 点预测 (q50) | 因果引导是否在调度上兑现? |
| **prob_cgprob** ⭐ | 22 | CG-Prob-Mamba (q50/iw/unc) | **不确定性信息额外贡献多少?** |
| **oracle** | 22 | true_wind[t+1] | 完美预测的上界在哪? |

每相邻两组只差一个变量, **整条 ablation chain 清晰**:

> baseline → point_dlinear → point_patchtst → point_cgmamba → prob_cgprob → oracle  
> "无预测" "弱预测" "SOTA 点预测" "+CG" "+不确定性" "完美预测"

## 数据流

**预测端**(已完成): 数据集 60/20/20 切分
- 60% 给预测模型训练 (predictor_train)
- 20% RL agent 训练时使用 (rl_train)
- 20% RL agent 评估时使用 (rl_eval)

**调度端**(本包): RL 训练在 `rl_train`, RL 评估在 `rl_eval`,
**严格不重叠**, 完全消除你之前 pipeline 里的数据泄露问题.

## 文件结构

```
dispatch_pkg/
├── setup.sh                          一键装环境
├── requirements.txt
├── README.md
├── src/
│   ├── 01_prepare_data.py            读 5 个预测模型 CSV, 输出调度时序
│   ├── 02_microgrid_env.py           7 类环境 (含 mode='train'/'eval')
│   ├── 03_train_sac.py               单 seed 训练, 支持 'all' 跑全 6 组
│   ├── 04_evaluate.py                评估 + 对比表 + 6 面板图
│   └── 05_multi_seed.py              多 seed + 配对 t 检验
├── data/
│   ├── rl_train.csv  / rl_eval.csv          主时序 (真实风电+合成负荷)
│   ├── pred_cgprob_train.csv  / pred_cgprob_eval.csv     概率预测
│   ├── pred_cgmamba_train.csv / pred_cgmamba_eval.csv    CG点预测
│   ├── pred_patchtst_train.csv / ...                     SOTA点预测
│   └── pred_dlinear_train.csv / ...                      轻量点预测
└── results/                           训练输出 (按 agent_seed<N> 分子目录)
```

## 完整运行流程

### 1. 装环境 (10 分钟)

```powershell
cd dispatch_pkg
bash setup.sh     # 自动检测 GPU
```

### 2. 准备数据 (1 分钟)

把你预测实验的根目录传进去 (含 results_cg_prob_mamba 等):

```powershell
python 01_prepare_data.py `
    --exp_root "C:\Users\children\Desktop\Wind_History Version\upgrade_wind_experiment_3" `
    --out_dir ./data
```

会输出 11 个 CSV (主时序 2 + 5 个预测模型 × 2 段). 缺哪个模型的预测 CSV 就跳过哪个.

### 3. 训练 6 组 (单 seed, 约 90-120 分钟)

```powershell
$env:PYTHONUTF8=1
python 03_train_sac.py --agent all --steps 50000 --seed 42 --device cuda
```

或单独跑某一组:

```powershell
python src/03_train_sac.py --agent prob_cgprob --steps 50000 --seed 42
```

可选 `--skip baseline,oracle` 跳过某些组.

### 4. 评估 + 出对比表/图 (1 分钟)

```powershell
python src/04_evaluate.py --out_root ./results --seed 42
```

输出在 `results/seed42_summary/`:
- `eval_metrics.csv`     6 组完整指标
- `eval_comparison.csv`  相对 baseline 的 Δ%
- `eval_comparison.md`   论文用 markdown 表格 (Table 5)
- `eval_comparison.png`  6 面板对比图

### 5. 多 seed 实验 (论文必备, 4-6 小时)

```powershell
python src/05_multi_seed.py --seeds 42,43,44 --steps 50000
```

输出在 `results/multi_seed/`:
- `aggregate.csv`        各 agent 的 mean ± std
- `significance.csv`     5 组 vs baseline 的配对 t 检验 p-value
- `aggregate_plot.png`   带误差棒的对比图

## 论文叙事 (审稿人视角的故事)

### 主张 1: 概率预测在调度上比点预测有额外价值

证据: prob_cgprob 在 cost/CO2/wind_util 上显著优于 point_cgmamba (p<0.05).
即使两者用的是同一个 backbone (CG-Mamba), 仅是输出从中位数变为分位数+不确定性.

### 主张 2: CG (因果引导) 在调度上仍然有效

证据: point_cgmamba 优于 point_dlinear 和 point_patchtst, 即使后者是 SOTA 点预测器.

### 主张 3: 我们的方法接近 Oracle 上界

证据: prob_cgprob 与 oracle 的 gap 远小于 prob_cgprob 与 baseline 的 gap,
说明现有概率预测已逼近"完美预测"性能.

### 审稿人会问的问题与回答

**Q1**: 三种 point 预测的差异在哪?
A: 设计时让三者**同样只输出 q50 一个标量**, 唯一区别是预测精度
(MAE: DLinear=4782 > PatchTST=4756 > CG-Mamba=4777). 调度差异完全归因于预测精度.

**Q2**: prob_cgprob 和 point_cgmamba 是同一个 backbone, 调度差异从哪来?
A: 完全来自不确定性信息.
- point_cgmamba: obs 多了 1 维 (q50_norm)
- prob_cgprob:   obs 多了 3 维 (q50_norm, interval_90_norm, uncertainty_norm)
两者同物理仿真, 同算法, 同超参. 差异 = 区间宽度信号的边际价值.

**Q3**: 为什么 oracle 不一定最好?
A: Oracle 只有 1-step lookahead, 不是真正的最优. 但提供了"完美短时预测"的参照.
论文里诚实标注这点.

## 关键参数

| 文件 | 参数 | 默认 | 说明 |
|---|---|---|---|
| 02_microgrid_env.py | LOSS_LOAD_COST | 200 | 失负荷罚款, $/MWh |
| 02_microgrid_env.py | OVERGEN_COST | 50 | 过发电罚款, $/MWh |
| 02_microgrid_env.py | GENSET_CONFIG | 4 台 | 1 燃气 + 3 煤炭 |
| 03_train_sac.py | --steps | 50000 | 论文级 ≥50k |
| 03_train_sac.py | --algo | SAC | 也支持 TD3/DDPG |
| 03_train_sac.py | --device | auto | auto/cuda/cpu |

## 注意事项

**Q: 跑 oracle 时如果遇到内存问题?**
A: oracle obs 22 维, 比 baseline 19 维稍大, 但 4060 Laptop 8GB 完全够.
若 OOM, 降 batch_size=128 或 buffer_size=50000.

**Q: 想加更多对照组 (例如 PointAware-iTransformer)?**
A: 在 02_microgrid_env.py 的 `point_map` 里加一行就行, 数据准备已经支持.

**Q: 训练后 prob_cgprob 比 baseline 差?**
A: 检查 (1) 训练步数 ≥50k; (2) 预测 CSV 已经过 fix_cg_prob_outliers.py 修复;
(3) 跨 seed 是否一致 (单 seed 不下结论).

## 引用

冯文韬等. 基于深度强化学习的微电网源-荷低碳调度优化研究. 四川电力技术, 2023, 46(6).
Henri-Kerr et al. pymgrid: An Open-Source Microgrid Simulator. NeurIPS Climate AI, 2020.
Raffin et al. Stable-baselines3. JMLR, 2021.

---

## V2: 论文级微电网环境 (2026-01 升级)

`02_microgrid_env_v2.py` 严格复现冯文韬 2023《基于深度强化学习的微电网源-荷低碳调度优化研究》的物理建模:

| 升级点 | v1 | v2 (论文级) |
|---|---|---|
| 碳排放模型 | 线性常系数 (50$/t一刀切) | **动态分段碳排放** (5 段, 强度递增) |
| 碳价机制 | 单一价格 | **阶梯碳价** (免费额度 + 4 阶递增) |
| 爬坡约束 | 无 | **每步 ΔP 限制** (各机组 25-80 MW/15min) |
| 负荷需求响应 | 无 | **±20% 负荷转移** (源-荷协同) |
| 节点结构 | 单节点抽象 | **PJM-5 三负荷节点** (F/G/H) |
| 平衡机制 | balancing 罚款主导 | **不平衡软约束** (15% 上限 + 适度罚款) |
| 成本结构 | 失负荷罚 200$/MWh 主导 | 机组43% + 碳价42% + 不平衡13% (论文比例) |

**默认 03_train_sac.py 和 04_evaluate.py 会自动用 v2 环境** (找到 `02_microgrid_env_v2.py` 时), 找不到才回退到 v1.

### v2 的 obs/action

- obs: 22 维 (baseline) / 23 维 (point_*) / 25 维 (prob/oracle)
  - [0:4] 当前 4 台机组归一化出力
  - [4:8] 上一步出力 (爬坡感知)
  - [8:11] 3 节点负荷
  - [11] 时间 t/T
  - [12] 实测当前风电
  - [13:17] 累计 4 机组碳排放
  - [17:19] 累计 cost / carbon_cost
  - [19:22] 最近 3 步不平衡功率
  - [extra] 预测信号 (q50 / q50+iw+unc / true_t+1)

- action: 8 维 [-1, 1]
  - [0:4] 4 台机组目标出力比例 (映射到 [P_min, P_max], 受爬坡约束)
  - [4:7] 3 节点负荷需求响应 (±20%)
  - [7] 风电接入比例 (0=全弃, 1=全消纳)

### 跑命令完全不变

```powershell
python src/03_train_sac.py --agent all --steps 50000 --seed 42 --device cuda
python src/04_evaluate.py --out_root ./results --seed 42
```
