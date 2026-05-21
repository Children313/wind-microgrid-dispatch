"""
01_prepare_data.py
==================
把 5 个预测模型的 RL train/test CSV 转成调度环境需要的时间序列.

输入: 5 个预测模型的 _rl_train.csv 和 _rl_test.csv
  - results_cg_prob_mamba/cg_prob_predictions_rl_{train,test}.csv  (主推, 含分位数+不确定性)
  - results_cg_mamba/cg_mamba_predictions_rl_{train,test}.csv      (CG 点预测)
  - results_tsl_baselines/PatchTST/patchtst_predictions_rl_{train,test}.csv (SOTA 点预测)
  - results_tsl_baselines/DLinear/dlinear_predictions_rl_{train,test}.csv   (轻量点预测)
  (mamba_predictions 也可以加, 但和 CG-Mamba 类似, 默认不用)

输出 (在 ./data/):
  - rl_train.csv  / rl_eval.csv   (主时序: 真实风电+合成负荷, 给 Baseline+Oracle 用)
  - pred_cgprob_train.csv / pred_cgprob_eval.csv   (CG-Prob-Mamba 概率预测)
  - pred_cgmamba_train.csv / pred_cgmamba_eval.csv (CG-Mamba 点预测)
  - pred_patchtst_train.csv / pred_patchtst_eval.csv (PatchTST 点预测)
  - pred_dlinear_train.csv / pred_dlinear_eval.csv (DLinear 点预测)

合成负荷曲线: PJM-5 风格, 日周期 + 周周期 + 月级波动, 峰值=风电峰值×1.5
"""

import os, argparse
import numpy as np
import pandas as pd


def synth_load(N, wind_peak_MW, steps_per_day=96, seed=42):
    """合成 PJM-5 风格负荷曲线"""
    LOAD_PEAK = wind_peak_MW * 1.5
    LOAD_BASE = LOAD_PEAK * 0.5
    rng = np.random.RandomState(seed)
    t = np.arange(N)
    daily   = 0.30 * np.sin(2*np.pi*t/steps_per_day - np.pi/3)
    weekly  = 0.05 * np.sin(2*np.pi*t/(steps_per_day*7))
    trend   = 0.10 * np.sin(2*np.pi*t/(steps_per_day*30))
    noise   = 0.03 * rng.randn(N)
    shape = np.clip(1.0 + daily + weekly + trend + noise, 0.4, 1.4)
    load = LOAD_BASE + (LOAD_PEAK - LOAD_BASE) * \
           (shape - shape.min()) / (shape.max() - shape.min())
    return np.clip(load, LOAD_BASE*0.8, LOAD_PEAK)


def process_main(true_kW, segment_name, out_dir, seed):
    """主时序: 真实风电 + 合成负荷"""
    true_MW = np.clip(true_kW, 0.0, None) / 1000.0
    wind_peak = float(true_MW.max())
    load_MW = synth_load(len(true_MW), wind_peak, seed=seed)

    df = pd.DataFrame({
        'true_wind_MW': true_MW,
        'load_MW':      load_MW,
    })
    out_path = os.path.join(out_dir, f"rl_{segment_name}.csv")
    df.to_csv(out_path, index=False)
    return df, wind_peak, out_path


def process_pred_point(pred_csv, segment_name, model_tag, out_dir):
    """点预测模型: 只输出 pred_q50_MW"""
    df_p = pd.read_csv(pred_csv)
    pred_kW = np.clip(df_p['pred_kW'].values, 0.0, None)
    out = pd.DataFrame({'pred_q50_MW': pred_kW / 1000.0})
    path = os.path.join(out_dir, f"pred_{model_tag}_{segment_name}.csv")
    out.to_csv(path, index=False)
    return path


def process_pred_prob(pred_csv, segment_name, out_dir):
    """概率预测模型 (CG-Prob-Mamba): q05/q50/q95/interval/uncertainty"""
    df_p = pd.read_csv(pred_csv)
    out = pd.DataFrame({
        'pred_q05_MW':       np.clip(df_p['pred_q05_kW'].values, 0.0, None) / 1000.0,
        'pred_q50_MW':       np.clip(df_p['pred_median_kW'].values, 0.0, None) / 1000.0,
        'pred_q95_MW':       np.clip(df_p['pred_q95_kW'].values, 0.0, None) / 1000.0,
        'interval_90_MW':    df_p['interval_width_90_kW'].values / 1000.0,
        'uncertainty_norm':  df_p['uncertainty_norm'].values,
    })
    path = os.path.join(out_dir, f"pred_cgprob_{segment_name}.csv")
    out.to_csv(path, index=False)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_root', type=str, required=True,
                        help="预测实验根目录, 含 results_cg_prob_mamba 等子目录")
    parser.add_argument('--out_dir', type=str, default='./data')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("="*60)
    print(" 调度数据准备: 转 5 个预测模型的输出 → 调度时序")
    print("="*60)

    # 主时序 (从 CG-Prob-Mamba CSV 拿 true_kW, 因为所有模型 true 一致)
    cgprob_dir = os.path.join(args.exp_root, "results_cg_prob_mamba")

    # ---- RL train 段 ----
    df_p_tr = pd.read_csv(os.path.join(cgprob_dir, "cg_prob_predictions_rl_train.csv"))
    df_main_tr, wind_peak, path = process_main(df_p_tr['true_kW'].values, "train",
                                                args.out_dir, args.seed)
    print(f"\n[RL train] 主时序 → {path}")
    print(f"           {len(df_main_tr)} 行, 风电峰值={wind_peak:.1f} MW, "
          f"负荷峰值={df_main_tr['load_MW'].max():.1f} MW")

    # ---- RL eval 段 ----
    df_p_te = pd.read_csv(os.path.join(cgprob_dir, "cg_prob_predictions_rl_test.csv"))
    df_main_te, _, path = process_main(df_p_te['true_kW'].values, "eval",
                                        args.out_dir, args.seed + 1)
    print(f"\n[RL eval]  主时序 → {path}")
    print(f"           {len(df_main_te)} 行, "
          f"风电均值={df_main_te['true_wind_MW'].mean():.1f} MW (vs train 的 "
          f"{df_main_tr['true_wind_MW'].mean():.1f} MW)")

    # ---- 概率预测 (CG-Prob-Mamba) ----
    print("\n[概率预测] CG-Prob-Mamba:")
    for seg in ['train', 'eval']:
        suffix = 'rl_train' if seg == 'train' else 'rl_test'
        path = process_pred_prob(
            os.path.join(cgprob_dir, f"cg_prob_predictions_{suffix}.csv"),
            seg, args.out_dir)
        print(f"  {seg} → {path}")

    # ---- 4 个点预测模型 ----
    point_models = {
        'cgmamba':  ('results_cg_mamba',                  'cg_mamba_predictions'),
        'patchtst': ('results_tsl_baselines/PatchTST',   'patchtst_predictions'),
        'dlinear':  ('results_tsl_baselines/DLinear',    'dlinear_predictions'),
        'mamba':    ('results_mamba',                    'mamba_predictions'),
    }
    for tag, (subdir, prefix) in point_models.items():
        full_dir = os.path.join(args.exp_root, subdir)
        if not os.path.isdir(full_dir):
            print(f"\n[跳过] {tag}: 找不到 {full_dir}")
            continue
        print(f"\n[点预测] {tag}:")
        for seg in ['train', 'eval']:
            suffix = 'rl_train' if seg == 'train' else 'rl_test'
            csv_path = os.path.join(full_dir, f"{prefix}_{suffix}.csv")
            if not os.path.exists(csv_path):
                print(f"  [跳过] {csv_path} 不存在")
                continue
            out = process_pred_point(csv_path, seg, tag, args.out_dir)
            print(f"  {seg} → {out}")

    # ---- 摘要常量 ----
    print(f"\n{'='*60}\n摘要 (供 02_microgrid_env.py 配置)\n{'='*60}")
    print(f"  WIND_RATED_MW  = {wind_peak:.1f}")
    print(f"  LOAD_PEAK_MW   = {df_main_tr['load_MW'].max():.1f}")
    print(f"  N_TRAIN_STEPS  = {len(df_main_tr)}")
    print(f"  N_EVAL_STEPS   = {len(df_main_te)}")


if __name__ == "__main__":
    main()
