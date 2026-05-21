"""
05_multi_seed.py
================
多 seed 重跑 + 配对 t 检验 (论文必备的统计显著性).

使用:
  python 05_multi_seed.py --seeds 42,43,44 --steps 50000

输出:
  results/multi_seed/
    summary.csv          每个 (seed, agent) 的 6 指标
    aggregate.csv        各 agent 的 mean ± std
    significance.csv     5 组 vs baseline 的配对 t 检验 p-value
    aggregate_plot.png   带误差棒的对比图
"""

import warnings; warnings.filterwarnings('ignore')
import os, sys, argparse, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

THIS = Path(__file__).parent.resolve()


def run_one_seed(seed, steps, algo, device, out_root):
    print(f"\n{'#'*70}\n# Seed {seed}\n{'#'*70}")
    py = sys.executable
    cmd = [py, str(THIS/"03_train_sac.py"),
           "--steps", str(steps), "--seed", str(seed),
           "--algo", algo, "--device", device,
           "--out_root", out_root, "--agent", "all"]
    subprocess.run(cmd, check=True)
    cmd = [py, str(THIS/"04_evaluate.py"),
           "--out_root", out_root, "--seed", str(seed),
           "--algo", algo, "--device", device]
    subprocess.run(cmd, check=True)
    summary_dir = os.path.join(out_root, f"seed{seed}_summary")
    return pd.read_csv(os.path.join(summary_dir, "eval_metrics.csv"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=str, default='42,43,44')
    parser.add_argument('--steps', type=int, default=50000)
    parser.add_argument('--algo', type=str, default='SAC')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--out_root', type=str, default='./results')
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    print(f"[plan] seeds={seeds}, steps={args.steps}, algo={args.algo}")

    multi_dir = os.path.join(args.out_root, 'multi_seed')
    os.makedirs(multi_dir, exist_ok=True)

    all_dfs = []
    for s in seeds:
        df = run_one_seed(s, args.steps, args.algo, args.device, args.out_root)
        df['seed'] = s
        all_dfs.append(df)

    summary = pd.concat(all_dfs, ignore_index=True)
    summary.to_csv(os.path.join(multi_dir, 'summary.csv'), index=False)
    print(f"\n[saved] {multi_dir}/summary.csv ({len(summary)} 行)")

    # 聚合 mean ± std
    metrics = ['total_reward', 'total_cost_dollar', 'total_co2_ton',
               'wind_utilization', 'reward_std', 'balancing_MWh']
    agg = summary.groupby('name')[metrics].agg(['mean', 'std']).round(2)
    agg.to_csv(os.path.join(multi_dir, 'aggregate.csv'))
    print(f"\n[聚合] {multi_dir}/aggregate.csv")
    print(agg)

    # 配对 t 检验 (其他 5 组 vs baseline)
    print(f"\n{'='*70}\n配对 t 检验 (相对 baseline)\n{'='*70}")
    sig_rows = []
    if 'baseline' not in summary['name'].unique():
        print("没有 baseline 数据, 跳过 t 检验")
    else:
        bl_rows = summary[summary['name'] == 'baseline'].sort_values('seed')
        for other_name in summary['name'].unique():
            if other_name == 'baseline': continue
            ot_rows = summary[summary['name'] == other_name].sort_values('seed')
            if len(ot_rows) != len(bl_rows): continue
            for metric in metrics:
                x = bl_rows[metric].values
                y = ot_rows[metric].values
                if len(x) < 2: continue
                t_stat, p_val = stats.ttest_rel(x, y)
                sig_rows.append({
                    'comparison': f"{other_name} vs baseline",
                    'metric': metric,
                    'baseline_mean': float(np.mean(x)),
                    'other_mean':    float(np.mean(y)),
                    'diff_mean':     float(np.mean(y - x)),
                    't_stat':        float(t_stat),
                    'p_value':       float(p_val),
                    'sig_05':        bool(p_val < 0.05),
                })
    if sig_rows:
        sig_df = pd.DataFrame(sig_rows)
        sig_df.to_csv(os.path.join(multi_dir, 'significance.csv'), index=False)
        print(sig_df.to_string(index=False))

    # 带误差棒图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics_plot = ['total_cost_dollar', 'total_co2_ton', 'wind_utilization']
    titles = ['Total Cost ($)', 'CO2 (tons)', 'Wind Utilization (%)']
    colors = {
        'baseline':       '#1f77b4',
        'point_dlinear':  '#ff7f0e',
        'point_patchtst': '#bcbd22',
        'point_cgmamba':  '#9467bd',
        'point_mamba':    '#8c564b',
        'prob_cgprob':    '#d62728',
        'oracle':         '#2ca02c',
    }
    NAME_ORDER = ['baseline','point_dlinear','point_patchtst',
                  'point_cgmamba','point_mamba','prob_cgprob','oracle']
    for ax, metric, title in zip(axes, metrics_plot, titles):
        names = [n for n in NAME_ORDER if n in summary['name'].unique()]
        means = [summary[summary['name']==n][metric].mean() for n in names]
        stds  = [summary[summary['name']==n][metric].std()  for n in names]
        bars = ax.bar(names, means, yerr=stds, capsize=8,
            color=[colors.get(n, 'gray') for n in names],
            edgecolor='black', linewidth=1)
        for b, m, s in zip(bars, means, stds):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+s,
                    f'{m:,.0f}\n±{s:,.0f}',
                    ha='center', va='bottom', fontsize=8)
        ax.set_title(title); ax.tick_params(axis='x', labelrotation=30)
        ax.grid(alpha=0.3, axis='y')

    plt.suptitle(f"{args.algo} 多 seed 对照 (n={len(seeds)} seeds, "
                 f"{args.steps} steps each)")
    plt.tight_layout()
    out = os.path.join(multi_dir, 'aggregate_plot.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\n[saved] {out}")
    print(f"\n✓ 完成. 论文 Table 5 用:\n  - {multi_dir}/aggregate.csv (mean±std)\n  - {multi_dir}/significance.csv (p-values)\n  - {multi_dir}/aggregate_plot.png")


if __name__ == "__main__":
    main()
