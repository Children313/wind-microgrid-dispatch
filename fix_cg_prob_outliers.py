"""
fix_cg_prob_outliers.py
=======================
修复 CG-Prob-Mamba 在 RL train 段的离群预测问题, 不需要重新训练模型.

发现的问题:
  - RL train 段 2 个样本的 q95 > 1,000,000 kW (峰值 148,000 的 7-40倍)
  - 中位数预测最大值 2,361,555 kW (峰值的 16倍)
  - interval_width_90 最大值 5,726,647 kW (污染整个归一化基准)
  - 导致 uncertainty_norm 的 RL test 段全压缩到 [0.0003, 0.0285], 信号失效

修复策略:
  1. 用真值的 1.2 倍作为物理上限, clip 所有分位数和中位数
  2. 重算 interval_width 列
  3. 用 clip 后的 RL train 段 max 作为统一归一化基准, 重算两段的 uncertainty_norm
  4. 强制分位数单调性 (q05 <= q10 <= q25 <= q50 <= q75 <= q90 <= q95)

使用:
    python fix_cg_prob_outliers.py --result_dir results_cg_prob_mamba
"""

import os
import argparse
import numpy as np
import pandas as pd


def fix_predictions(result_dir: str, phys_max_factor: float = 1.2):
    """
    Args:
        result_dir: 包含 cg_prob_predictions_rl_train.csv 和 _rl_test.csv 的目录
        phys_max_factor: 物理上限相对于真值最大值的倍数 (默认 1.2)
                         比真值最大值大 20% 留 buffer, 但远小于离群预测
    """
    train_path = os.path.join(result_dir, "cg_prob_predictions_rl_train.csv")
    test_path  = os.path.join(result_dir, "cg_prob_predictions_rl_test.csv")
    legacy_path = os.path.join(result_dir, "cg_prob_predictions.csv")

    df_tr = pd.read_csv(train_path)
    df_te = pd.read_csv(test_path)
    print(f"[加载]  train: {len(df_tr)} 行, test: {len(df_te)} 行")

    # ── Step 1: 确定物理上限 ──
    # 用两段真值最大值的 1.2 倍, 既宽松又能压住离群预测
    true_max = max(df_tr['true_kW'].max(), df_te['true_kW'].max())
    PHYS_MAX = float(true_max * phys_max_factor)
    print(f"[物理上限] 真值最大值={true_max:.0f}, clip上限={PHYS_MAX:.0f} (×{phys_max_factor})")

    # ── Step 2: clip 所有分位数预测 + 中位数 ──
    quantile_cols = ['pred_q05_kW', 'pred_q10_kW', 'pred_q25_kW',
                     'pred_median_kW', 'pred_q75_kW', 'pred_q90_kW', 'pred_q95_kW']

    for df_name, df in [('train', df_tr), ('test', df_te)]:
        n_clipped = 0
        for c in quantile_cols:
            n_above = (df[c] > PHYS_MAX).sum()
            n_below = (df[c] < 0).sum()
            df[c] = df[c].clip(0, PHYS_MAX)
            n_clipped += n_above + n_below
        print(f"[clip {df_name}] 共修正 {n_clipped} 个超界值")

    # ── Step 3: 强制分位数单调性 ──
    # 离群点修了之后, 大概率不再有违反, 但保险起见做一遍
    quantile_order = ['pred_q05_kW', 'pred_q10_kW', 'pred_q25_kW',
                      'pred_median_kW', 'pred_q75_kW', 'pred_q90_kW', 'pred_q95_kW']
    for df_name, df in [('train', df_tr), ('test', df_te)]:
        Q = df[quantile_order].values
        # 沿分位数维度做 cummax, 保证单调非降
        Q_mono = np.maximum.accumulate(Q, axis=1)
        n_violations = int((Q != Q_mono).any(axis=1).sum())
        if n_violations > 0:
            print(f"[单调化 {df_name}] 修正 {n_violations} 行违反样本")
            for i, c in enumerate(quantile_order):
                df[c] = Q_mono[:, i]

    # ── Step 4: 重算区间宽度 ──
    for df in [df_tr, df_te]:
        df['interval_width_90_kW'] = df['pred_q95_kW'] - df['pred_q05_kW']
        df['interval_width_80_kW'] = df['pred_q90_kW'] - df['pred_q10_kW']

    # ── Step 5: 重算 uncertainty_norm (用 RL train 段的 max 作为统一基准) ──
    iw_max_global = float(df_tr['interval_width_90_kW'].max())
    print(f"[归一化基准] 用 RL train 段 interval_90 max = {iw_max_global:.0f} 作为统一基准")
    for df in [df_tr, df_te]:
        df['uncertainty_norm'] = df['interval_width_90_kW'] / (iw_max_global + 1e-8)
        df['uncertainty_norm'] = df['uncertainty_norm'].clip(0.0, 1.0)

    # ── Step 6: 验证修复后的统计 ──
    print("\n========== 修复后诊断 ==========")
    for name, df in [('rl_train', df_tr), ('rl_test', df_te)]:
        print(f"\n--- {name} ---")
        print(f"  pred_median_kW  range: [{df['pred_median_kW'].min():.0f}, {df['pred_median_kW'].max():.0f}]")
        print(f"  pred_q05_kW     range: [{df['pred_q05_kW'].min():.0f}, {df['pred_q05_kW'].max():.0f}]")
        print(f"  pred_q95_kW     range: [{df['pred_q95_kW'].min():.0f}, {df['pred_q95_kW'].max():.0f}]")
        print(f"  interval_90_kW  range: [{df['interval_width_90_kW'].min():.0f}, {df['interval_width_90_kW'].max():.0f}]")
        print(f"  uncertainty_norm range: [{df['uncertainty_norm'].min():.4f}, {df['uncertainty_norm'].max():.4f}]")

        # 重算指标 (基于中位数)
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        y, yhat = df['true_kW'].values, df['pred_median_kW'].values
        mae  = mean_absolute_error(y, yhat)
        rmse = np.sqrt(mean_squared_error(y, yhat))
        r2   = 1 - ((y - yhat)**2).mean() / y.var()
        print(f"  Metrics: MAE={mae:.0f}, RMSE={rmse:.0f}, R²={r2:.4f}")

    # ── Step 7: 保存 (覆盖原文件, 同时备份) ──
    for path in [train_path, test_path]:
        bak = path + ".outliers_bak"
        if not os.path.exists(bak):
            os.rename(path, bak)
            print(f"\n[备份] 原文件 → {os.path.basename(bak)}")

    df_tr.to_csv(train_path, index=False)
    df_te.to_csv(test_path, index=False)
    df_te.to_csv(legacy_path, index=False)  # 兼容旧名
    print(f"[保存] {train_path}")
    print(f"[保存] {test_path}")
    print(f"[保存] {legacy_path}  (同 _rl_test, 兼容旧脚本)")
    print("\n✓ 修复完成. RL 下游可以放心使用 _rl_train 和 _rl_test.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str,
                        default='results_cg_prob_mamba',
                        help='CG-Prob-Mamba 的结果目录')
    parser.add_argument('--phys_max_factor', type=float, default=1.2,
                        help='物理上限 = 真值max * 该因子 (默认 1.2)')
    args = parser.parse_args()

    fix_predictions(args.result_dir, args.phys_max_factor)
