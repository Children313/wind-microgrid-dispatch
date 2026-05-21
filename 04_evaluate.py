"""
04_evaluate.py
==============
读 6 组训练好的模型, 在 *eval 段* 评估 (out-of-sample!), 出论文用的表和图.

输出:
  results/seed<S>_summary/
    eval_metrics.csv        每组的完整指标
    eval_comparison.csv     5 组 vs Baseline 的相对差值
    eval_comparison.md      Markdown 表格 (论文 Table 5)
    eval_comparison.png     6 面板对比图

使用:
  python 04_evaluate.py --out_root ./results --seed 42
"""

import warnings; warnings.filterwarnings('ignore')
import os, sys, argparse
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.resolve()))
from importlib import import_module
try:
    mge = import_module("02_microgrid_env_v2")
except ImportError:
    mge = import_module("02_microgrid_env")

from stable_baselines3 import SAC, TD3, DDPG
ALGOS = {"SAC": SAC, "TD3": TD3, "DDPG": DDPG}


# ====================================================================
#  Evaluate one model on its eval env
# ====================================================================
def _safe_float(v):
    if v is None: return 0.0
    try: return float(v)
    except Exception:
        try: return float(np.asarray(v).item())
        except Exception: return 0.0


def evaluate(model, env, name, eval_steps=500, seed=43):
    obs, _ = env.reset(seed=seed)
    rewards = []
    cost_genset_total = 0.0
    cost_wind_total = 0.0
    cost_wind_curtail_total = 0.0
    cost_carbon_total = 0.0
    cost_load_shift_total = 0.0
    cost_imbalance_total = 0.0
    co2_total = 0.0
    wind_used = 0.0
    wind_curtail = 0.0
    imbalance_total = 0.0
    load_total_cum = 0.0

    for step in range(eval_steps):
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        rewards.append(r)
        if isinstance(info, dict):
            cost_genset_total      += info.get('cost_genset', 0.0)
            cost_wind_total        += info.get('cost_wind', 0.0)
            cost_wind_curtail_total += info.get('cost_wind_curtail', 0.0)
            cost_carbon_total      += info.get('cost_carbon', 0.0)
            cost_load_shift_total  += info.get('cost_load_shift', 0.0)
            cost_imbalance_total   += info.get('cost_imbalance', 0.0)
            co2_total              += info.get('co2_total_step', 0.0)
            # MWh = MW * dt_h, 这里 info 的 wind_accepted_MW 是瞬时值
            dt_h = 0.25
            wind_used     += info.get('wind_accepted_MW', 0.0) * dt_h
            wind_curtail  += info.get('wind_curtailed_MW', 0.0) * dt_h
            imbalance_total += abs(info.get('imbalance_MW', 0.0)) * dt_h
            load_total_cum  += info.get('load_total_MW', 0.0) * dt_h
        if term or trunc: break

    rewards = np.asarray(rewards)
    cost_total_dollar = (cost_genset_total + cost_wind_total + cost_wind_curtail_total + cost_carbon_total
                          + cost_load_shift_total + cost_imbalance_total)  # 含不平衡, 反映系统真实成本
    return {
        'name': name,
        'eval_steps': len(rewards),
        'total_reward':       float(rewards.sum()),
        'mean_step_reward':   float(rewards.mean()),
        'reward_std':         float(rewards.std()),
        'total_cost_dollar':  float(cost_total_dollar),
        'cost_genset':        float(cost_genset_total),
        'cost_wind':          float(cost_wind_total),
        'cost_wind_curtail':  float(cost_wind_curtail_total),
        'cost_carbon':        float(cost_carbon_total),
        'cost_load_shift':    float(cost_load_shift_total),
        'cost_imbalance':     float(cost_imbalance_total),
        'total_co2_ton':      float(co2_total),
        'wind_used_MWh':      float(wind_used),
        'wind_curtailed_MWh': float(wind_curtail),
        'wind_utilization':   float(wind_used / max(wind_used+wind_curtail, 1e-6)) * 100,
        'imbalance_MWh':      float(imbalance_total),
        'load_total_MWh':     float(load_total_cum),
    }


# ====================================================================
#  Plot: 6 面板对比
# ====================================================================
def plot_comparison(metrics_list, train_curves, out_path, algo='SAC'):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    # 颜色: baseline 蓝, point* 暖色系, prob* 红, oracle 绿
    colors = {
        'baseline':       '#1f77b4',
        'point_dlinear':  '#ff7f0e',
        'point_patchtst': '#bcbd22',
        'point_cgmamba':  '#9467bd',
        'point_mamba':    '#8c564b',
        'prob_cgprob':    '#d62728',
        'oracle':         '#2ca02c',
    }
    names = [m['name'] for m in metrics_list]

    # (a) 训练曲线
    ax = axes[0, 0]
    for n, df in train_curves.items():
        if df is None or len(df) == 0: continue
        smooth = df['mean_reward'].rolling(window=3, min_periods=1).mean()
        ax.plot(df['step'], smooth, '-', label=n, color=colors.get(n, 'gray'), linewidth=1.5)
    ax.set_xlabel('Training Step'); ax.set_ylabel('Mean Reward (smoothed)')
    ax.set_title(f'(a) {algo} Training Curves')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (b) Total Cost
    ax = axes[0, 1]
    costs = [m['total_cost_dollar'] for m in metrics_list]
    bars = ax.bar(names, costs, color=[colors.get(n, 'gray') for n in names])
    for b, v in zip(bars, costs):
        ax.text(b.get_x()+b.get_width()/2, b.get_height(), f'${v/1000:.0f}k',
                ha='center', va='bottom', fontsize=8)
    ax.set_ylabel('Total Cost ($)'); ax.set_title('(b) Operating Cost (lower better)')
    ax.tick_params(axis='x', labelrotation=30); ax.grid(alpha=0.3, axis='y')

    # (c) CO2
    ax = axes[0, 2]
    co2s = [m['total_co2_ton'] for m in metrics_list]
    bars = ax.bar(names, co2s, color=[colors.get(n, 'gray') for n in names])
    for b, v in zip(bars, co2s):
        ax.text(b.get_x()+b.get_width()/2, b.get_height(), f'{v/1000:.0f}kt',
                ha='center', va='bottom', fontsize=8)
    ax.set_ylabel('CO2 (tons)'); ax.set_title('(c) CO2 Emission (lower better)')
    ax.tick_params(axis='x', labelrotation=30); ax.grid(alpha=0.3, axis='y')

    # (d) Wind Utilization
    ax = axes[1, 0]
    util = [m['wind_utilization'] for m in metrics_list]
    bars = ax.bar(names, util, color=[colors.get(n, 'gray') for n in names])
    ax.set_ylim(0, max(105, max(util) * 1.1))
    for b, v in zip(bars, util):
        ax.text(b.get_x()+b.get_width()/2, b.get_height(), f'{v:.1f}%',
                ha='center', va='bottom', fontsize=8)
    ax.set_ylabel('Wind Utilization (%)')
    ax.set_title('(d) Wind Utilization (higher better)')
    ax.tick_params(axis='x', labelrotation=30); ax.grid(alpha=0.3, axis='y')

    # (e) Reward Std
    ax = axes[1, 1]
    stds = [m['reward_std'] for m in metrics_list]
    bars = ax.bar(names, stds, color=[colors.get(n, 'gray') for n in names])
    for b, v in zip(bars, stds):
        ax.text(b.get_x()+b.get_width()/2, b.get_height(), f'{v:.0f}',
                ha='center', va='bottom', fontsize=8)
    ax.set_ylabel('Reward Std'); ax.set_title('(e) Stability (lower better)')
    ax.tick_params(axis='x', labelrotation=30); ax.grid(alpha=0.3, axis='y')

    # (f) Imbalance Volume
    ax = axes[1, 2]
    bal = [m['imbalance_MWh'] for m in metrics_list]
    bars = ax.bar(names, bal, color=[colors.get(n, 'gray') for n in names])
    for b, v in zip(bars, bal):
        ax.text(b.get_x()+b.get_width()/2, b.get_height(), f'{v/1000:.1f}k',
                ha='center', va='bottom', fontsize=8)
    ax.set_ylabel('Imbalance (MWh)')
    ax.set_title('(f) Imbalance Volume (lower better)')
    ax.tick_params(axis='x', labelrotation=30); ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  对比图 → {out_path}")


# ====================================================================
#  Main
# ====================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_root', type=str, default='./results')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--algo', type=str, default='SAC')
    parser.add_argument('--eval_steps', type=int, default=500)
    parser.add_argument('--eval_seed', type=int, default=43,
                        help="环境 reset seed (与训练 seed 不同, 增加多样性)")
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    Algo = ALGOS[args.algo]

    # 找所有训练过的 agent 目录
    EVAL_ORDER = [
        'baseline',
        'point_dlinear', 'point_patchtst', 'point_cgmamba', 'point_mamba',
        'prob_cgprob',
        'oracle',
    ]

    summary_dir = os.path.join(args.out_root, f"seed{args.seed}_summary")
    os.makedirs(summary_dir, exist_ok=True)

    metrics_list = []
    train_curves = {}
    for agent in EVAL_ORDER:
        adir = os.path.join(args.out_root, f"{agent}_seed{args.seed}")
        model_path = os.path.join(adir, "model")
        log_path   = os.path.join(adir, "train_log.csv")
        if not os.path.exists(model_path + ".zip"):
            print(f"[skip] {agent}: 没找到 {model_path}.zip")
            continue

        print(f"\n[Loading] {agent}: {model_path}")
        model = Algo.load(model_path, device=args.device)
        # 关键: 评估用 eval 段数据 (out-of-sample!)
        env_eval = mge.make_env(agent, mode='eval')
        m = evaluate(model, env_eval, agent, args.eval_steps, args.eval_seed)
        metrics_list.append(m)
        for k, v in m.items():
            if k == 'name': continue
            print(f"    {k:24s} {v:>14,.2f}" if isinstance(v, float) else f"    {k:24s} {v}")
        if os.path.exists(log_path):
            train_curves[agent] = pd.read_csv(log_path)
        else:
            train_curves[agent] = None

    if not metrics_list:
        print("\n没有任何模型可评估"); return

    # 写完整 metrics CSV
    pd.DataFrame(metrics_list).to_csv(
        os.path.join(summary_dir, "eval_metrics.csv"), index=False)

    # 写对比表 (相对 baseline)
    bl = next((m for m in metrics_list if m['name'] == 'baseline'), None)
    if bl:
        rows = []
        for k in ['total_reward', 'total_cost_dollar', 'cost_genset', 'cost_carbon',
                  'cost_load_shift', 'cost_imbalance',
                  'total_co2_ton', 'wind_utilization', 'reward_std', 'imbalance_MWh']:
            row = {'metric': k, 'baseline': bl[k]}
            for m in metrics_list:
                if m['name'] == 'baseline': continue
                row[m['name']] = m[k]
                if bl[k] != 0:
                    row[f"Δ%_{m['name']}"] = (m[k] - bl[k]) / abs(bl[k]) * 100
                else:
                    row[f"Δ%_{m['name']}"] = 0
            rows.append(row)
        cmp_df = pd.DataFrame(rows)
        cmp_df.to_csv(os.path.join(summary_dir, "eval_comparison.csv"), index=False)
        print(f"\n{'='*70}\n论文用对比表 (相对 baseline 的 Δ%)\n{'='*70}")
        print(cmp_df.to_string(index=False))

        # Markdown 版
        md = ["# Dispatch Comparison: 6 RL Agents on Wind Microgrid (Paper-Level Env)\n"]
        md.append(f"Algorithm: {args.algo} | Seed: {args.seed} | Eval steps: {args.eval_steps}")
        md.append(f"Eval data: rl_eval.csv (out-of-sample, 严格不重叠 RL 训练数据)\n")
        md.append("## Main Comparison\n")
        md.append("| Agent | Reward | Total Cost ($) | Genset Cost | Wind Cost | Curtail Cost | Carbon Cost | CO2 (t) | Wind Util (%) | Imbalance (MWh) |")
        md.append("|:------|-------:|---------------:|------------:|----------:|-------------:|------------:|--------:|--------------:|----------------:|")
        for m in metrics_list:
            md.append(f"| {m['name']} | {m['total_reward']:,.0f} | "
                      f"{m['total_cost_dollar']:,.0f} | "
                      f"{m.get('cost_genset', 0.0):,.0f} | "
                      f"{m.get('cost_wind', 0.0):,.0f} | {m.get('cost_wind_curtail', 0.0):,.0f} | "
                      f"{m.get('cost_carbon', 0.0):,.0f} | "
                      f"{m['total_co2_ton']:,.0f} | "
                      f"{m['wind_utilization']:.1f} | {m['imbalance_MWh']:,.0f} |")
        with open(os.path.join(summary_dir, "eval_comparison.md"), 'w', encoding='utf-8') as f:
            f.write('\n'.join(md))

    # 出 6 面板图
    plot_comparison(metrics_list, train_curves,
        os.path.join(summary_dir, "eval_comparison.png"), algo=args.algo)
    print(f"\n全部产出在: {summary_dir}/")


if __name__ == "__main__":
    main()
