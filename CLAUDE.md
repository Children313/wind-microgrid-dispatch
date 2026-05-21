# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Wind power forecasting + microgrid dispatch co-optimization. A **CG-ProbMamba** probabilistic wind forecast model feeds an RL dispatch agent (SAC). The goal is to prove that probabilistic predictions (with uncertainty) improve downstream dispatch decisions on three metrics: **economic cost, CO2 emissions, wind utilization**.

The project has two halves:
1. **Prediction side**: 8 wind forecast models (CG-ProbMamba, CG-Mamba, Mamba, DLinear, PatchTST, iTransformer, TimeMixer, Autoformer) trained on `15_processed.csv` (43,680 rows, 15-min resolution)
2. **Dispatch side**: 6 SAC agents sharing the same microgrid physics but receiving different prediction signals — an ablation chain from "no prediction" to "perfect prediction"

## Key Architectural Decisions

### Prediction-dispatch data split (60/20/20)
- 60%: predictor training (predictor never sees the last 40%)
- 20%: RL agent training (`rl_train.csv`)
- 20%: RL agent evaluation (`rl_eval.csv`)
- RL train and eval segments are strictly non-overlapping in time

### Observation design (after 7 rounds of iteration)

The critical design choice: **prediction occupies `base[12]`** — the same index where baseline has current wind. This forces every agent to use its dispatch-relevant signal at index 12.

| Agent | obs dim | `base[12]` contains | `extra` contains |
|:------|:-------:|:--------------------|:-----------------|
| baseline | 22 | current wind (reactive) | — |
| point_* | 23 | **predicted net load** = (load−q50)/peak | current wind |
| prob_cgprob | 24 | **predicted net load** (q50) | current wind, iw uncertainty |
| oracle | 23 | **true net load** (perfect) | current wind |

`base[12]` was chosen because that's where `get_baseline_obs()` puts `true_wind[t+1]` (exogenous future data). We overwrite it to prevent data leaks and to force prediction use.

Base observation (22 dims): `[genset_outputs(4), prev_genset(4), node_loads(3), time, wind, cumulative_co2(4), cumulative_costs(2), recent_imbalance(3)]`

### Why prediction at base[12] matters

Previous designs put predictions in "extra" appended dimensions. RL agents learned to ignore them — baseline (no prediction) consistently beat all prediction-equipped agents. The prediction signal was too weak among 22 other features. Putting it at index 12 (the primary wind-related index) forces the policy network to use it.

### Wind autocorrelation and prediction value

Wind at 15-min lag has autocorrelation r=0.973 (R²=0.947). This means `wind[t]` already explains 94.7% of `wind[t+1]` variance. Baseline implicitly uses persistence. Prediction models (MAE 4.7 MW, R²=0.947) barely beat persistence at this horizon. The marginal value is small but real — models reduce RMSE by ~39% vs persistence. The challenge is making the RL agent exploit this margin.

### Microgrid physics (冯文韬 2023)

- **Gas turbine** (auto-balancing, NOT in RL action): P_max=25 MW, ramp=10 MW/step, cost=$15/MWh, CO2=0.6 t/MWh
- **3 coal units** (RL-controlled): P_max=50/200/80 MW, ramp=12/35/18 MW/step, cost=$14-30/MWh, CO2=1.4 t/MWh
- **Wind**: accepted at cost $10/MWh, curtailed at penalty $50/MWh
- **Dynamic segmented carbon emissions** (5 segments, increasing intensity)
- **Stepped carbon pricing**: free allowance 10 t/step, base price $25/t, 4 price steps with 25% increments
- **Imbalance penalties**: $3000/MWh (under-generation), $2000/MWh (over-generation)
- **Demand response**: ±20% load shifting at 3 nodes (F:22%, G:36%, H:42%)
- **Action space**: 6-dim [coal1, coal2, coal3, dr_F, dr_G, dr_H]
- **Automatic wind curtailment**: when residual_imbalance < 0 (excess generation), wind is curtailed before incurring imbalance penalties

### RL training (SAC)

- Algorithm: SAC with `ent_coef='auto_0.05'`
- Network: [400, 300], learning_rate=5e-5, batch_size=1024, gamma=0.995
- Buffer: 200K, learning_starts=5000
- Default 200K steps per agent (agents still converging at 150K)
- **Best-model checkpointing**: `RewardLogger` saves `model.zip` whenever a new best mean reward is achieved. `model_final.zip` is also saved at training end.

## Commands

### Run prediction models (generate forecast CSVs for RL)

```powershell
python cg_prob_mamba.py --data 15_processed.csv
python cg_mamba.py --data 15_processed.csv
python mamba_baseline_aligned.py --data 15_processed.csv
python tsl_baselines.py --data 15_processed.csv --model all
```

### Prepare RL data from prediction outputs

```powershell
python 01_prepare_data.py --exp_root . --out_dir ./data
```

### Train RL agents

```powershell
# All 6 agents (baseline, point_dlinear, point_patchtst, point_cgmamba, prob_cgprob, oracle)
python 03_train_sac.py --agent all --steps 200000 --seed 42

# Single agent
python 03_train_sac.py --agent prob_cgprob --steps 200000 --seed 42

# Skip some agents
python 03_train_sac.py --agent all --skip "point_mamba,point_dlinear" --seed 42
```

### Evaluate and produce comparison tables/figures

```powershell
python 04_evaluate.py --out_root ./results --seed 42
```

Output in `results/seed42_summary/`: `eval_metrics.csv`, `eval_comparison.csv`, `eval_comparison.md`, `eval_comparison.png`

### Multi-seed statistical testing

```powershell
python 05_multi_seed.py --seeds 42,43,44 --steps 200000
```

## File Map

| File | Role |
|:-----|:-----|
| `02_microgrid_env_v2.py` | **Core environment** — WindMicrogridCore (physics) + WindMicrogridEnv (gym wrapper with 6 obs modes) |
| `03_train_sac.py` | RL training with best-model checkpointing |
| `04_evaluate.py` | Deterministic evaluation + comparison tables/plots |
| `01_prepare_data.py` | Converts prediction CSVs to RL-ready format |
| `05_multi_seed.py` | Multi-seed paired t-tests |
| `cg_prob_mamba.py` | **CG-ProbMamba** — causal-guided probabilistic forecast (Mamba SSM + CausalSE + MonotonicQuantileHead) |
| `cg_mamba.py` | CG-Mamba — same backbone, point forecast output (ablation control) |
| `mamba_baseline_aligned.py` | Pure Mamba baseline (no CG, no quantiles) |
| `tsl_baselines.py` | 5 SOTA baselines: DLinear, PatchTST, iTransformer, TimeMixer, Autoformer |
| `fix_cg_prob_outliers.py` | Fix outlier-inflated uncertainty in CG-ProbMamba RL train segment |
| `15_processed.csv` | Main dataset: 43,680 rows × 7 cols, 15-min wind power |

### Data directories

- `data/` — RL-ready CSVs (`rl_train.csv`, `rl_eval.csv`, `pred_*_{train,eval}.csv`)
- `results/` — Trained RL models (`{agent}_seed42/model.zip`) + summary
- `results_cg_prob_mamba/` — Trained CG-ProbMamba model + predictions + metrics
- `results_cg_mamba/` — Trained CG-Mamba model + predictions
- `results_mamba/` — Trained vanilla Mamba
- `results_tsl_baselines/` — 5 SOTA baseline models + comparison table

## Important Parameters

| Where | Parameter | Value | Notes |
|:------|:----------|:-----:|:------|
| `02_microgrid_env_v2.py:61` | Gas turbine P_max | 25 MW | Reduced from 50 to make predictions necessary |
| `02_microgrid_env_v2.py:61` | Gas turbine ramp | 10 MW/step | Reduced from 25 |
| `02_microgrid_env_v2.py:121` | Carbon price (PSI_2) | $25/t | Increased from 15 |
| `02_microgrid_env_v2.py:122` | Free carbon allowance (B_C) | 10 t/step | Reduced from 30 |
| `02_microgrid_env_v2.py:363-364` | Imbalance penalty | $3000/$3000 per MWh | symmetric (no over-gen bias) |
| `03_train_sac.py:89` | SAC learning rate | 5e-5 | Reduced from 1e-4 for stability |
| `03_train_sac.py:91` | SAC batch size | 1024 | Increased from 512 |
| `03_train_sac.py:92` | gamma | 0.995 | Increased from 0.99 |
| `03_train_sac.py:94` | net_arch | [400, 300] | Increased from [256, 256] |
| `03_train_sac.py:150` | Default training steps | 200000 | Increased from 50000 |

## Lessons from 7 Rounds of Debugging

1. **Don't put predictions in "extra" observation dimensions** — the agent will ignore them. Put predictions at the same index where baseline has its dispatch signal.

2. **Align prediction horizon with action horizon** — giving t+4 predictions when actions affect t+1 creates misalignment the agent can't resolve.

3. **Check wind autocorrelation before designing the prediction interface** — at 15-min resolution, r=0.97 means persistence is a strong baseline. The prediction interface must make the marginal value of better forecasts visible to the RL agent.

4. **SAC with large observation spaces needs careful tuning** — reduce LR (5e-5), increase batch (1024), increase gamma (0.995), increase network ([400,300]), increase steps (200K). Training instability (catastrophic forgetting) was a recurring problem.

5. **Wind curtailment must be a real mechanism, not hardcoded to zero** — otherwise "wind utilization" is a meaningless metric.

6. **Asymmetric imbalance penalties ($3000 vs $2000) bias the agent toward over-generation** — which causes excessive genset use, high carbon, and high wind curtailment. **Fixed in v2: penalties now symmetric ($3000/$3000).**

7. **Best-model checkpointing is essential** — training reward often degrades in later steps due to catastrophic forgetting. Always save `model.zip` at best mean reward, not just at the end.
