"""
CG + Mamba 模型（因果先验引导的纯 Mamba）

设计动机
────────────────────────────────────────────────────────
在完整消融表里，CG（因果先验SE注意力）是唯一被证明有正贡献的模块：
  CEEMDAN+MSCNN+Mamba       → MAE=11080, R²=0.9237
  CEEMDAN+CG-MSCNN+Mamba    → MAE=10996, R²=0.9253   (CG 贡献 ΔMAE=-84)

而 CEEMDAN 和 MSCNN 均被证明是"拖累项":
  MSCNN+Mamba   → MAE 比 Mamba 恶化 +655
  CEEMDAN+Mamba → MAE 比 Mamba 恶化 +1062

核心假设：如果 CG 在 CEEMDAN+MSCNN 的噪声基础上都能做出正贡献，
那么把 CG 单独加在本来就最强的 Mamba 上，应该获得更干净的正向效果。
本实验正是对这一假设的直接检验。

与 CG-MSCNN+Mamba 的关键区别
────────────────────────────────────────────────────────
  CG-MSCNN+Mamba:
    输入 → MSCNN(6→48) → SE注意力作用于48个卷积输出通道 → Mamba
    ↑ 注意力作用的是"抽象特征通道"，物理含义间接
  
  CG-Mamba（本模型）:
    输入 → SE注意力直接作用于6个原始物理通道 → 投影 → Mamba
    ↑ 注意力作用的是"物理变量通道"，每个权重直接对应一个气象量
    ↑ 可解释性更强：SE输出的6个权重就是"当前窗口下风速/温度/...的相对重要性"

与纯 Mamba 的关键区别
────────────────────────────────────────────────────────
  Mamba:       输入 → input_proj(6→96) → Mamba×3 → FC
  CG-Mamba:    输入 → CG-SE(6→6) → input_proj(6→96) → Mamba×3 → FC
                    ↑ 仅增加 58 个参数
  
  参数量变化：191,041 → 191,099（+0.03%）
  这意味着如果性能提升，只可能来自 CG 模块本身（消融变量唯一）

CG 模块机制
────────────────────────────────────────────────────────
1. PCMCI 在训练集上发现气象变量对功率的因果强度
2. 与物理先验（基于 P∝v³·ρ 的理论）融合，得到 6 维因果权重
3. 用融合权重在 logit 空间初始化 SE 注意力的 fc2.bias
   即令 sigmoid(bias_init) = causal_weight
4. 训练中 SE 的权重可自由更新 —— 因果先验是"起点"而非"终点"

运行方式:
    python cg_mamba.py --data 风电机组数据集.xls
    python cg_mamba.py --demo
    python cg_mamba.py --data 风电机组数据集.xls \
        --causal-weights results_ceemdan_cg_mscnn_mamba/causal_weights.npy
"""

import os
import sys
import argparse
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# ══════════════════════════════════════════════════════
#  mamba-ssm CUDA 加速 (可选, 找不到则 fallback 到 Python loop)
# ══════════════════════════════════════════════════════
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    USE_MAMBA_SSM_KERNEL = True
    print("[加速] 使用 mamba-ssm CUDA kernel (实测 ~30x)")
except ImportError:
    selective_scan_fn = None
    USE_MAMBA_SSM_KERNEL = False
    print("[加速] mamba-ssm 未安装, 使用 Python loop")
    print("       安装: pip install mamba-ssm causal-conv1d --no-build-isolation")
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")
matplotlib.rcParams["axes.unicode_minus"] = False


# ══════════════════════════════════════════════════════
#  0. 全局配置（与 Mamba / MSCNN+Mamba 完全对齐）
# ══════════════════════════════════════════════════════

DATA_PATH    = "风电机组数据集.xls"
RESULT_DIR   = "results_cg_mamba"

FEATURE_COLS = ["WINDSPEED", "WINDDIRECTION", "TEMPERATURE", "HUMIDITY", "PRESSURE"]
TARGET_COL   = "YD15"

SEQ_LEN      = 96
TRAIN_RATIO  = 0.6   # 60% 用于预测模型训练
RL_TRAIN_RATIO = 0.2  # 20% 作为 RL agent 训练时使用 (预测模型不见)
                      # 剩下 20% 自动作为 RL test 段

# PCMCI 参数（与 ceemdan_cg_mscnn_mamba 完全一致）
PCMCI_TAU_MAX = 4       # 最大因果时滞（4×15min = 1小时）
PCMCI_ALPHA   = 0.05
CAUSAL_MIN_W  = 0.10    # 因果权重最小值（防止通道完全死亡）
FUSION_ALPHA  = 0.5     # PCMCI 与物理先验的融合权重

# Mamba 超参数（与对照组一致）
D_MODEL      = 96
D_STATE      = 16
D_CONV       = 4
EXPAND       = 2
N_LAYERS     = 3

BATCH_SIZE   = 64
NUM_EPOCHS   = 100
LR           = 0.005
LR_PATIENCE  = 8
EARLY_STOP   = 30
SEED         = 42


# ══════════════════════════════════════════════════════
#  日志
# ══════════════════════════════════════════════════════

class Logger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")
    def write(self, msg):
        self.terminal.write(msg); self.log.write(msg)
    def flush(self):
        self.terminal.flush(); self.log.flush()
    def close(self):
        self.log.flush(); self.log.close()


# ══════════════════════════════════════════════════════
#  1. 合成数据（demo 模式）
# ══════════════════════════════════════════════════════

def generate_demo_data(n: int = 16354) -> str:
    np.random.seed(SEED)
    t = np.linspace(0, 8 * np.pi, n)
    wind = np.clip(np.random.weibull(2.0, n) * 8 + 3 * np.sin(t / 15), 0, 25)
    power = np.clip(0.5 * wind ** 3 * np.random.uniform(0.8, 1.2, n), 0, 200)
    power[wind < 3] = 0; power[wind > 20] = 0
    df = pd.DataFrame({
        "WINDSPEED": wind, "WINDDIRECTION": np.random.uniform(0, 360, n),
        "TEMPERATURE": np.random.uniform(-10, 30, n),
        "HUMIDITY": np.random.uniform(20, 90, n),
        "PRESSURE": np.random.uniform(830, 850, n),
        "YD15": np.clip(power + np.random.normal(0, 2, n), 0, None),
    })
    path = "demo_data.csv"
    df.to_csv(path, index=False)
    print(f"[演示模式] 生成合成数据 {n} 条 → {path}")
    return path


# ══════════════════════════════════════════════════════
#  2. 数据加载与清洗
# ══════════════════════════════════════════════════════

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path) if path.endswith(".csv") else \
         pd.read_excel(path, engine="xlrd" if path.endswith(".xls") else "openpyxl")
    print(f"[加载] 原始行数: {len(df)}, 列名: {list(df.columns)}")
    power_col = TARGET_COL
    if power_col not in df.columns:
        candidates = [c for c in df.columns if any(k in c.upper() for k in ["POWER", "YD15"])]
        if not candidates:
            raise KeyError(f"找不到功率列，可用列: {list(df.columns)}")
        power_col = candidates[0]
        print(f"[加载] 自动识别功率列: {power_col}")
    df.rename(columns={power_col: "POWER"}, inplace=True)
    col_upper = {c.upper(): c for c in df.columns}
    for feat in FEATURE_COLS:
        if feat not in df.columns and feat.upper() in col_upper:
            df.rename(columns={col_upper[feat.upper()]: feat}, inplace=True)
    df = df[df["POWER"] >= 0].copy()
    df[FEATURE_COLS + ["POWER"]] = (
        df[FEATURE_COLS + ["POWER"]]
        .replace([float("inf"), float("-inf")], float("nan"))
        .interpolate(method="linear", limit_direction="both")
    )
    df.dropna(subset=FEATURE_COLS + ["POWER"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"[清洗] 清洗后行数: {len(df)}")
    return df


# ══════════════════════════════════════════════════════
#  3. PCMCI 因果发现 + 物理先验融合
#     （与 ceemdan_cg_mscnn_mamba 完全一致，保证因果权重可复用）
# ══════════════════════════════════════════════════════

def run_pcmci(power_tr: np.ndarray,
              feat_tr:  np.ndarray,
              save_dir: str) -> np.ndarray:
    """
    用 PCMCI 算法发现气象变量对功率的因果强度，与物理先验融合。

    参数:
        power_tr: 归一化功率序列 (T,)
        feat_tr:  归一化气象数据 (T, 5)，列顺序与 FEATURE_COLS 一致
        save_dir: 因果图保存路径

    返回:
        causal_weights: (6,) 数组
            [0] = 1.0          → 历史功率通道（始终保留）
            [1] = ws因果强度    (对应 WINDSPEED)
            [2] = wd因果强度    (对应 WINDDIRECTION)
            [3] = temp因果强度  (对应 TEMPERATURE)
            [4] = hum因果强度   (对应 HUMIDITY)
            [5] = pres因果强度  (对应 PRESSURE)
    """
    var_names   = ["Power"] + FEATURE_COLS
    n_vars      = len(var_names)
    data_matrix = np.column_stack([power_tr, feat_tr])   # (T, 6)

    try:
        from tigramite import data_processing as pp
        from tigramite.pcmci import PCMCI
        from tigramite.independence_tests.parcorr import ParCorr

        print("  [PCMCI] 使用 tigramite 进行因果发现...")
        dataframe = pp.DataFrame(
            data_matrix, var_names=var_names,
            datatime=np.arange(len(data_matrix)))
        pcmci = PCMCI(dataframe=dataframe,
                      cond_ind_test=ParCorr(significance="analytic"),
                      verbosity=0)
        results = pcmci.run_pcmci(
            tau_min=1, tau_max=PCMCI_TAU_MAX, pc_alpha=PCMCI_ALPHA)

        val_matrix = results["val_matrix"]
        p_matrix   = results["p_matrix"]

        weather_strengths = []
        weather_lags      = []
        for i in range(1, n_vars):
            sig_mask = p_matrix[i, 0, 1:] < PCMCI_ALPHA
            vals     = np.abs(val_matrix[i, 0, 1:])
            if sig_mask.any():
                best_lag = int(np.argmax(vals * sig_mask)) + 1
                strength = float(vals[sig_mask].max())
            else:
                best_lag = 1
                strength = 0.0
            weather_strengths.append(strength)
            weather_lags.append(best_lag)
            print(f"    {var_names[i]:15s} → Power | "
                  f"强度={strength:.4f}, 最优时滞={best_lag}步"
                  f"({'显著' if sig_mask.any() else '不显著'})")
        method = "PCMCI"

    except ImportError:
        print("  [PCMCI] tigramite 未安装，退化为 Pearson 相关系数")
        print("  建议: pip install tigramite")
        weather_strengths = []
        weather_lags      = []
        for i in range(1, n_vars):
            corr = float(np.abs(np.corrcoef(
                data_matrix[1:, 0], data_matrix[:-1, i])[0, 1]))
            weather_strengths.append(corr)
            weather_lags.append(1)
            print(f"    {var_names[i]:15s} → Power | 相关系数={corr:.4f}")
        method = "Pearson"

    # 归一化 PCMCI 强度
    strengths = np.array(weather_strengths, dtype=np.float32)
    s_max = float(strengths.max()) if strengths.size else 0.0
    w_pcmci = strengths / s_max if s_max > 1e-8 else np.zeros_like(strengths)

    # 物理先验（基于 P = 0.5·ρ·Cp·A·v³ 的直觉）
    # ws: 功率的三次方关系，最重要
    # wd: 影响扫风面积 Cp
    # pres: 影响空气密度 ρ
    # temp: 微小影响 ρ
    # hum: 最小影响 ρ
    w_physics = np.array([1.00, 0.50, 0.30, 0.20, 0.35], dtype=np.float32)

    if w_physics.shape[0] != w_pcmci.shape[0]:
        raise ValueError(f"物理先验长度不匹配 FEATURE_COLS={FEATURE_COLS}")

    # 线性融合
    w_fused = FUSION_ALPHA * w_pcmci + (1.0 - FUSION_ALPHA) * w_physics

    # 重新缩放到 [CAUSAL_MIN_W, 1]，防止某通道权重塌缩到 0
    w_min = float(w_fused.min()) if w_fused.size else 0.0
    w_max = float(w_fused.max()) if w_fused.size else 0.0
    if (w_max - w_min) > 1e-8:
        strengths = CAUSAL_MIN_W + (1.0 - CAUSAL_MIN_W) * \
                    (w_fused - w_min) / (w_max - w_min)
    else:
        strengths = np.full_like(w_fused, (CAUSAL_MIN_W + 1.0) / 2)

    # 第 0 位给历史功率通道（始终 = 1.0）
    causal_weights = np.concatenate([[1.0], strengths])

    print(f"\n  物理-统计融合权重（α={FUSION_ALPHA}）:")
    print(f"  {'变量':15s} {'PCMCI':>10s} {'物理先验':>10s} {'融合':>10s} {'最终':>10s}")
    for name, wp, wphy, wf, wfin in zip(FEATURE_COLS, w_pcmci, w_physics,
                                        w_fused, strengths):
        print(f"  {name:15s} {wp:>10.4f} {wphy:>10.4f} "
              f"{wf:>10.4f} {wfin:>10.4f}")

    print(f"\n  最终因果权重（{method}）:")
    for name, w in zip(["POWER(历史)"] + FEATURE_COLS, causal_weights):
        bar = "█" * int(w * 20)
        print(f"    {name:15s}: {w:.4f}  {bar}")

    np.save(os.path.join(save_dir, "causal_weights.npy"), causal_weights)
    _plot_causal_weights(causal_weights, ["POWER"] + FEATURE_COLS,
                         method, save_dir)
    return causal_weights


def _plot_causal_weights(weights: np.ndarray, labels: list,
                          method: str, save_dir: str):
    """绘制因果强度柱状图（论文可用）"""
    os.makedirs(save_dir, exist_ok=True)
    colors = ["#2196F3" if w > 0.5 else
              "#FF9800" if w > 0.2 else
              "#9E9E9E" for w in weights]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, weights, color=colors,
                   edgecolor="white", linewidth=0.8)
    ax.axhline(0.5, color="red", ls="--", lw=1,
               label="Strong causal threshold (0.5)")
    ax.axhline(CAUSAL_MIN_W, color="gray", ls=":", lw=1,
               label=f"Minimum weight ({CAUSAL_MIN_W})")
    for bar, w in zip(bars, weights):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01, f"{w:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Causal Weight (normalized)")
    ax.set_title(f"Causal Strength of Input Channels on Wind Power ({method})\n"
                 "Blue=Strong  Orange=Weak  Gray=Spurious")
    ax.legend(fontsize=8)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    path = os.path.join(save_dir, "causal_weights.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  [图表] 因果权重图 → {path}")


# ══════════════════════════════════════════════════════
#  4. 滑动窗口构建
# ══════════════════════════════════════════════════════

def build_windows(feat: np.ndarray, power: np.ndarray, seq_len: int = SEQ_LEN):
    """
    输入: feat(T,5), power(T,)
    输出: X(N, seq_len, 6), y(N,)
      6 = 1 个历史功率 + 5 个气象特征
      通道顺序: [POWER, WS, WD, TEMP, HUM, PRESSURE]
      → 与 causal_weights 的顺序完全一致
    """
    X_list, y_list = [], []
    for i in range(len(power) - seq_len):
        pw_win   = power[i: i + seq_len].reshape(-1, 1)
        feat_win = feat[i: i + seq_len]
        X_list.append(np.concatenate([pw_win, feat_win], axis=1))
        y_list.append(power[i + seq_len])
    return (np.array(X_list, dtype=np.float32),
            np.array(y_list,  dtype=np.float32))


# ══════════════════════════════════════════════════════
#  5. Mamba SSM（与 mamba_baseline 完全一致）
# ══════════════════════════════════════════════════════

class MambaSSM(nn.Module):
    """单层 Mamba SSM Block（纯 PyTorch 实现）"""

    def __init__(self, d_model: int, d_state: int = D_STATE,
                 d_conv: int = D_CONV, expand: int = EXPAND):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.inner   = d_model * expand

        self.in_proj = nn.Linear(d_model, self.inner * 2, bias=False)
        self.conv1d  = nn.Conv1d(
            self.inner, self.inner, kernel_size=d_conv,
            padding=d_conv - 1, groups=self.inner, bias=True)
        self.x_proj  = nn.Linear(self.inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.inner, bias=True)
        nn.init.uniform_(self.dt_proj.bias, -4, -1)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)
        A = A.repeat(self.inner, 1)
        self.register_buffer("A_log", torch.log(A))

        self.D        = nn.Parameter(torch.ones(self.inner))
        self.out_proj = nn.Linear(self.inner, d_model, bias=False)
        self.norm     = nn.LayerNorm(d_model)

    def selective_scan(self, u, delta, A, B, C):
        batch, d_inner, L = u.shape
        delta_A   = torch.exp(-A.unsqueeze(0).unsqueeze(-1) * delta.unsqueeze(2))
        delta_B_u = delta.unsqueeze(2) * B.unsqueeze(1) * u.unsqueeze(2)
        h = torch.zeros(batch, d_inner, self.d_state,
                        dtype=u.dtype, device=u.device)
        ys = []
        for t in range(L):
            h = delta_A[..., t] * h + delta_B_u[..., t]
            ys.append((h * C[:, :, t].unsqueeze(1)).sum(dim=2))
        return torch.stack(ys, dim=2)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        batch, L, _ = x.shape
        xz = self.in_proj(x)
        x_main, z = xz.chunk(2, dim=-1)
        x_conv = self.conv1d(x_main.permute(0, 2, 1))[..., :L]
        x_conv = F.silu(x_conv)
        x_dbc  = self.x_proj(x_conv.permute(0, 2, 1))
        delta_log = x_dbc[..., :1]
        B = x_dbc[..., 1: 1 + self.d_state].permute(0, 2, 1)
        C = x_dbc[..., 1 + self.d_state:].permute(0, 2, 1)
        delta = F.softplus(self.dt_proj(delta_log)).permute(0, 2, 1)
        A = torch.exp(self.A_log).to(x.dtype)
        if USE_MAMBA_SSM_KERNEL and x_conv.is_cuda:
            # mamba-ssm 期望 A < 0; D 由库内部加, 不需要外面再加
            y = selective_scan_fn(
                x_conv, delta, -A, B, C, D=self.D,
                z=None, delta_bias=None, delta_softplus=False
            )
        else:
            y = self.selective_scan(x_conv, delta, A, B, C)
            y = y + self.D.unsqueeze(0).unsqueeze(-1) * x_conv
        y = y * F.silu(z.permute(0, 2, 1))
        return self.out_proj(y.permute(0, 2, 1)) + residual


# ══════════════════════════════════════════════════════
#  6. 因果先验 SE 注意力（作用于原始 6 维输入通道）
# ══════════════════════════════════════════════════════

class CausalSEAttention(nn.Module):
    """
    因果先验引导的 SE 通道注意力 —— 作用于原始 6 维输入通道。

    与 CG-MSCNN 中 SE 注意力的对比:
      CG-MSCNN 版本:
        作用对象: MSCNN 输出的 48 个抽象特征通道
        初始化:   每个输出通道取所有输入因果权重的均值 → 无区分度
      
      CG-Mamba 版本（本模块）:
        作用对象: 原始 6 个物理变量通道（功率历史 + 5 气象）
        初始化:   每个通道直接用对应因果权重 → 保留物理语义

    机制:
      1. Squeeze:  全局平均池化，每个通道一个标量（全局时序信息）
      2. Excitation: FC → ReLU → FC → Sigmoid
         - fc2.bias 用 logit(causal_weights) 初始化
         - 意味着若 fc2.weight=0, sigmoid(bias) = causal_weights
         - 模型起点就在因果先验上，训练过程可自由修正
      3. Scale:    输入通道 × 注意力权重
    """

    def __init__(self, causal_weights: np.ndarray, reduction: int = 2):
        super().__init__()
        n_ch = len(causal_weights)              # 6
        mid  = max(n_ch // reduction, 4)         # 4

        self.n_ch    = n_ch
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.fc1     = nn.Linear(n_ch, mid)
        self.fc2     = nn.Linear(mid, n_ch)

        # 存下因果先验权重（仅作日志/可视化用，不参与训练）
        self.register_buffer("causal_prior",
                             torch.tensor(causal_weights, dtype=torch.float32))

        # ── 关键：用因果权重初始化 fc2.bias（logit 空间）──
        # 当 fc1 输出接近 0 时，sigmoid(fc2.bias) ≈ causal_weights
        # 保证训练开局就符合因果先验，但不是硬约束（bias 是可学习参数）
        with torch.no_grad():
            w = torch.tensor(causal_weights, dtype=torch.float32)
            w = torch.clamp(w, 1e-4, 1 - 1e-4)
            logit_prior = torch.log(w / (1 - w))
            self.fc2.bias.data.copy_(logit_prior)

        nn.init.xavier_uniform_(self.fc1.weight)
        # fc2.weight 初始化为小值，让初始输出尽量由 bias 主导
        nn.init.xavier_uniform_(self.fc2.weight)
        self.fc2.weight.data *= 0.1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, n_ch, seq_len)
        返回: (batch, n_ch, seq_len)  每个通道按注意力权重缩放
        """
        s = self.squeeze(x).squeeze(-1)    # (batch, n_ch)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))     # (batch, n_ch) ∈ (0,1)
        return x * s.unsqueeze(-1)         # 逐通道缩放

    def get_average_weights(self, x: torch.Tensor) -> np.ndarray:
        """推理辅助：返回当前 batch 的平均通道权重（可视化用）"""
        with torch.no_grad():
            s = self.squeeze(x).squeeze(-1)
            s = F.relu(self.fc1(s))
            s = torch.sigmoid(self.fc2(s))
            return s.mean(dim=0).cpu().numpy()


# ══════════════════════════════════════════════════════
#  7. CG-Mamba 主模型
# ══════════════════════════════════════════════════════

class CGMambaModel(nn.Module):
    """
    CG-Mamba: 因果先验 SE 注意力 + 纯 Mamba

    网络结构:
      输入 (B, seq_len=96, n_features=6)
        → permute → (B, 6, L)
        → CausalSEAttention (物理通道级加权)  (B, 6, L)
        → permute → (B, L, 6)
        → input_proj(6→96)                   (B, L, 96)
        → N 层 MambaSSM                       (B, L, 96)
        → 取最后时间步                        (B, 96)
        → LayerNorm → FC → (B,)

    与纯 Mamba 的唯一区别：
      在 input_proj 前增加了 CausalSEAttention。
      参数量新增 = 6*4+4 + 4*6+6 = 58 个 （约 0.03%）。
    """

    def __init__(self,
                 causal_weights: np.ndarray,
                 n_features: int = 6,
                 d_model:    int = D_MODEL,
                 d_state:    int = D_STATE,
                 d_conv:     int = D_CONV,
                 expand:     int = EXPAND,
                 n_layers:   int = N_LAYERS):
        super().__init__()
        assert len(causal_weights) == n_features, \
            f"causal_weights 长度 {len(causal_weights)} != n_features {n_features}"

        # 因果先验 SE 注意力
        self.cg_se = CausalSEAttention(causal_weights)

        # 输入投影（与 mamba_baseline 一致）
        self.input_proj = nn.Linear(n_features, d_model)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

        # Mamba 堆叠（与 mamba_baseline 一致）
        self.mamba_layers = nn.ModuleList([
            MambaSSM(d_model, d_state, d_conv, expand)
            for _ in range(n_layers)
        ])

        # 输出头
        self.norm_out = nn.LayerNorm(d_model)
        self.fc_out   = nn.Linear(d_model, 1)
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, n_features)"""
        # CG-SE 注意力（通道级加权）
        x = x.permute(0, 2, 1)        # (B, 6, L)
        x = self.cg_se(x)             # (B, 6, L)
        x = x.permute(0, 2, 1)        # (B, L, 6)

        # 纯 Mamba 部分
        x = self.input_proj(x)        # (B, L, 96)
        for layer in self.mamba_layers:
            x = layer(x)              # (B, L, 96)
        x = x[:, -1, :]               # (B, 96)
        x = self.norm_out(x)
        return self.fc_out(x).squeeze(-1)


# ══════════════════════════════════════════════════════
#  8. 训练函数
# ══════════════════════════════════════════════════════

def train(model, X_tr, y_tr, X_vl, y_vl, device):
    torch.manual_seed(SEED); np.random.seed(SEED)
    X_tr = torch.FloatTensor(X_tr).to(device)
    y_tr = torch.FloatTensor(y_tr).to(device)
    X_vl = torch.FloatTensor(X_vl).to(device)
    y_vl = torch.FloatTensor(y_vl).to(device)

    train_loader = DataLoader(TensorDataset(X_tr, y_tr),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_vl, y_vl),
                              batch_size=512, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.2, patience=LR_PATIENCE)
    criterion = nn.MSELoss()

    best_val, best_state, patience_cnt = float("inf"), None, 0
    train_hist, val_hist = [], []
    t0 = time.time()

    print(f"\n{'─'*55}")
    print(f" 开始训练 CG-Mamba 模型")
    print(f" 训练样本: {len(X_tr)} | 验证样本: {len(X_vl)}")
    print(f"{'─'*55}")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item() * len(yb)
        tr_loss /= len(X_tr)

        model.eval()
        vl_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                vl_loss += criterion(model(Xb), yb).item() * len(yb)
        vl_loss /= len(X_vl)

        train_hist.append(tr_loss); val_hist.append(vl_loss)
        scheduler.step(vl_loss)

        if vl_loss < best_val:
            best_val     = vl_loss
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1

        if patience_cnt >= EARLY_STOP:
            print(f"  早停于 Epoch {epoch}, 最佳验证损失: {best_val:.6f}, "
                  f"耗时: {time.time()-t0:.1f}s")
            break

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}/{NUM_EPOCHS} | "
                  f"Train: {tr_loss:.5f} | Val: {vl_loss:.5f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.5f} | "
                  f"耗时: {time.time()-t0:.1f}s")

    model.load_state_dict(best_state)
    model.eval()
    return train_hist, val_hist


# ══════════════════════════════════════════════════════
#  9. 评价指标 / 绘图
# ══════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred):
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_r = np.sum((y_true - y_pred) ** 2)
    ss_t = np.sum((y_true - y_true.mean()) ** 2)
    r2   = float(1 - ss_r / (ss_t + 1e-10))
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def plot_prediction(y_true, y_pred, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(np.arange(len(y_true)), y_true / 1000, "b-",  lw=0.7, label="True Power")
    ax.plot(np.arange(len(y_pred)), y_pred / 1000, "r--", lw=0.7, label="Predicted Power")
    ax.set_xlabel("Time Step"); ax.set_ylabel("Power (MW)")
    ax.set_title("CG-Mamba Prediction"); ax.legend()
    plt.tight_layout()
    path = os.path.join(save_dir, "cg_mamba_prediction.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"[图表] 预测结果图 → {path}")


def plot_loss(train_hist, val_hist, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_hist, label="Train Loss")
    ax.plot(val_hist,   label="Val Loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
    ax.set_title("CG-Mamba Training Curve"); ax.legend()
    plt.tight_layout()
    path = os.path.join(save_dir, "cg_mamba_loss_curve.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"[图表] 损失曲线图 → {path}")


def plot_learned_weights(model, X_sample, causal_prior, save_dir):
    """对比训练后学到的通道权重 vs 因果先验权重（论文可用）"""
    os.makedirs(save_dir, exist_ok=True)
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        x = torch.FloatTensor(X_sample).to(device)
        x = x.permute(0, 2, 1)
        learned = model.cg_se.get_average_weights(x)

    labels = ["POWER"] + FEATURE_COLS
    x_pos = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x_pos - width/2, causal_prior, width,
           label="Causal Prior (initial)", color="#9E9E9E")
    ax.bar(x_pos + width/2, learned, width,
           label="Learned (after training)", color="#2196F3")
    for i, (p, l) in enumerate(zip(causal_prior, learned)):
        ax.text(i - width/2, p + 0.01, f"{p:.2f}",
                ha="center", va="bottom", fontsize=8)
        ax.text(i + width/2, l + 0.01, f"{l:.2f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x_pos); ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Channel Weight")
    ax.set_title("Causal Prior vs Learned Channel Weights in CG-Mamba\n"
                 "(shows how training updates the causal initialization)")
    ax.set_ylim(0, 1.15); ax.legend()
    plt.tight_layout()
    path = os.path.join(save_dir, "learned_vs_prior_weights.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"[图表] 学习前后通道权重对比 → {path}")


# ══════════════════════════════════════════════════════
#  10. 主流程
# ══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="CG-Mamba 模型（因果先验引导的纯 Mamba）")
    parser.add_argument("--data", default=DATA_PATH)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--causal-weights", default=None,
                        help="可选：从文件加载已有的 causal_weights.npy，"
                             "跳过 PCMCI 计算（推荐使用 CG-MSCNN 产出的权重以保证一致性）")
    args = parser.parse_args()

    os.makedirs(RESULT_DIR, exist_ok=True)
    logger = Logger(os.path.join(RESULT_DIR, "training_log.txt"))
    sys.stdout = logger

    torch.manual_seed(SEED); np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[系统] 使用设备: {device}")

    print("\n消融实验说明:")
    print("  本组: CG + Mamba (因果先验SE注意力直接作用于原始6维输入)")
    print("  对比1: Mamba          MAE=10030.8 R²=0.9289 (无CG)")
    print("  对比2: CEEMDAN+CG-MSCNN+Mamba MAE=10996.3 R²=0.9253 (CG+MSCNN+CEEMDAN)")
    print("  目的: 检验 CG 模块在不含 MSCNN/CEEMDAN 时是否对 Mamba 有正贡献")
    print("  控制: Mamba 本体与对照组1完全一致，仅新增 58 个参数 (~0.03%)")

    if args.demo or not os.path.exists(args.data):
        print("[提示] 未找到数据文件，切换演示模式")
        data_path = generate_demo_data()
    else:
        data_path = args.data

    # ── Step 1: 加载数据 ──────────────────────────────
    print("\n" + "="*55)
    print(" Step 1: 加载与清洗数据")
    print("="*55)
    df = load_data(data_path)

    # ── Step 2: 划分 + 归一化 ─────────────────────────
    print("\n" + "="*55)
    print(" Step 2: 数据划分与归一化 (60/20/20 三段切分)")
    print("="*55)
    n_total = len(df)
    n_train = int(n_total * TRAIN_RATIO)                    # 60% 预测训练
    n_rl_tr = int(n_total * RL_TRAIN_RATIO)                 # 20% RL 训练段
    rl_tr_start, rl_tr_end = n_train, n_train + n_rl_tr
    rl_te_start, rl_te_end = rl_tr_end, n_total
    print(f"  总样本: {n_total}")
    print(f"  预测训练: 0..{n_train}")
    print(f"  RL 训练段:  {rl_tr_start}..{rl_tr_end}  ({n_rl_tr})")
    print(f"  RL 测试段:  {rl_te_start}..{rl_te_end}  ({rl_te_end-rl_te_start})")

    feat_scaler  = MinMaxScaler()
    power_scaler = MinMaxScaler()
    feat_tr  = feat_scaler.fit_transform(df.iloc[:n_train][FEATURE_COLS].values)
    power_tr = power_scaler.fit_transform(df.iloc[:n_train][["POWER"]].values).ravel()
    feat_rl_tr  = feat_scaler.transform(df.iloc[rl_tr_start:rl_tr_end][FEATURE_COLS].values)
    power_rl_tr = power_scaler.transform(df.iloc[rl_tr_start:rl_tr_end][["POWER"]].values).ravel()
    feat_te  = feat_scaler.transform(df.iloc[rl_te_start:rl_te_end][FEATURE_COLS].values)
    power_te = power_scaler.transform(df.iloc[rl_te_start:rl_te_end][["POWER"]].values).ravel()

    # ── Step 3: 获取因果权重 ─────────────────────────
    print("\n" + "="*55)
    print(" Step 3: 因果权重获取")
    print("="*55)

    if args.causal_weights and os.path.exists(args.causal_weights):
        causal_weights = np.load(args.causal_weights)
        print(f"  [加载] 从 {args.causal_weights} 读取因果权重")
        print(f"  权重: {dict(zip(['POWER'] + FEATURE_COLS, causal_weights.round(4)))}")
        # 同步保存一份到本实验目录
        np.save(os.path.join(RESULT_DIR, "causal_weights.npy"), causal_weights)
    else:
        print("  [计算] 在训练集上运行 PCMCI 因果发现")
        causal_weights = run_pcmci(power_tr, feat_tr, RESULT_DIR)

    if len(causal_weights) != 6:
        raise ValueError(
            f"因果权重维度错误: 期望 6 维 [POWER,WS,WD,TEMP,HUM,PRESSURE]，"
            f"实际 {len(causal_weights)} 维")

    print(f"\n  因果权重将作为 SE 注意力 fc2.bias 的 logit 初始值")
    print(f"  训练开始时 sigmoid(bias) ≈ 因果权重")
    print(f"  训练过程中可自由更新（非硬约束）")

    # ── Step 4: 滑动窗口 ──────────────────────────────
    print("\n" + "="*55)
    print(" Step 4: 构建96步滑动窗口样本")
    print("="*55)
    X_tr, y_tr = build_windows(feat_tr, power_tr)
    X_te, y_te = build_windows(feat_te, power_te)
    X_rl_tr, y_rl_tr = build_windows(feat_rl_tr, power_rl_tr)
    print(f"  预测训练: X={X_tr.shape}, y={y_tr.shape}")
    print(f"  RL 训练:  X={X_rl_tr.shape}, y={y_rl_tr.shape}")
    print(f"  RL 测试:  X={X_te.shape}, y={y_te.shape}")
    print(f"  通道顺序: [POWER, WS, WD, TEMP, HUM, PRESSURE]")

    n_val = max(100, int(len(X_tr) * 0.1))
    X_fit, y_fit = X_tr[:-n_val], y_tr[:-n_val]
    X_val, y_val = X_tr[-n_val:], y_tr[-n_val:]
    print(f"  训练子集: {len(X_fit)} | 验证子集: {len(X_val)}")

    # ── Step 5: 构建并训练模型 ────────────────────────
    print("\n" + "="*55)
    print(" Step 5: 构建 CG-Mamba 模型")
    print("="*55)
    n_features = X_tr.shape[2]
    model = CGMambaModel(
        causal_weights = causal_weights,
        n_features     = n_features,
        d_model        = D_MODEL,
        d_state        = D_STATE,
        d_conv         = D_CONV,
        expand         = EXPAND,
        n_layers       = N_LAYERS,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    cg_params    = sum(p.numel() for p in model.cg_se.parameters())
    mamba_params = total_params - cg_params

    print(f"  总参数量:     {total_params:,}")
    print(f"    CG-SE 模块: {cg_params:,}  (仅新增部分)")
    print(f"    Mamba 主体: {mamba_params:,}  (与 mamba_baseline 对齐)")
    print(f"  Mamba: d_model={D_MODEL}, d_state={D_STATE}, "
          f"d_conv={D_CONV}, expand={EXPAND}, n_layers={N_LAYERS}")
    print(f"  网络结构: CG-SE(6ch) → 投影({D_MODEL}d) → Mamba×{N_LAYERS} → FC")

    train_hist, val_hist = train(model, X_fit, y_fit, X_val, y_val, device)

    # ── Step 6: 预测 ──────────────────────────────────
    print("\n" + "="*55)
    print(" Step 6: 推理 (RL train + RL test 两段)")
    print("="*55)
    model.eval()

    def _infer(X_input):
        with torch.no_grad():
            loader = DataLoader(
                TensorDataset(torch.FloatTensor(X_input).to(device)),
                batch_size=512, shuffle=False)
            return np.concatenate(
                [model(Xb).cpu().numpy() for (Xb,) in loader])

    preds_norm        = _infer(X_te)         # RL test
    preds_norm_rl_tr  = _infer(X_rl_tr)      # RL train

    y_pred_kw       = np.clip(power_scaler.inverse_transform(preds_norm.reshape(-1, 1)).ravel(), 0, None)
    y_pred_kw_rl_tr = np.clip(power_scaler.inverse_transform(preds_norm_rl_tr.reshape(-1, 1)).ravel(), 0, None)
    y_true_kw       = power_scaler.inverse_transform(y_te.reshape(-1, 1)).ravel()
    y_true_kw_rl_tr = power_scaler.inverse_transform(y_rl_tr.reshape(-1, 1)).ravel()

    # ── Step 7: 评价指标 ──────────────────────────────
    print("\n" + "="*55)
    print(" Step 7: 评价指标（反归一化后，单位 kW）")
    print("="*55)
    m = compute_metrics(y_true_kw, y_pred_kw)
    print(f"  MAE  = {m['MAE']:.4f} kW  ({m['MAE']/1000:.4f} MW)")
    print(f"  RMSE = {m['RMSE']:.4f} kW  ({m['RMSE']/1000:.4f} MW)")
    print(f"  R²   = {m['R2']:.4f}")

    print("\n-------------------------------------------------------")
    print(" 消融对比（参考基准）:")
    print("   Mamba                    MAE=10030.8 kW  R²=0.9289")
    print("   CEEMDAN+CG-MSCNN+Mamba   MAE=10996.3 kW  R²=0.9253")
    print("   CG-Mamba (本模型)        MAE={:>7.1f} kW  R²={:.4f}".format(
        m['MAE'], m['R2']))
    d_vs_mamba = m['MAE'] - 10030.8
    print(f"\n   ΔMAE vs 纯 Mamba = {d_vs_mamba:+.1f} kW  "
          f"({'↓更好' if d_vs_mamba < 0 else '↑更差'})")
    print("-------------------------------------------------------")

    # ── Step 8: 保存结果 + 可视化 ─────────────────────
    # 主 CSV (RL test 段, 维持原文件名兼容旧脚本)
    result_df = pd.DataFrame({"true_kW": y_true_kw, "pred_kW": y_pred_kw})
    csv_path  = os.path.join(RESULT_DIR, "cg_mamba_predictions.csv")
    result_df.to_csv(csv_path, index=False)
    # 显式命名的两段 CSV (供下游 RL PointAware 使用)
    pd.DataFrame({"true_kW": y_true_kw, "pred_kW": y_pred_kw}).to_csv(
        os.path.join(RESULT_DIR, "cg_mamba_predictions_rl_test.csv"), index=False)
    pd.DataFrame({"true_kW": y_true_kw_rl_tr, "pred_kW": y_pred_kw_rl_tr}).to_csv(
        os.path.join(RESULT_DIR, "cg_mamba_predictions_rl_train.csv"), index=False)
    print(f"\n[结果] CSV(RL test) → {csv_path}  ({len(result_df)} 行)")
    print(f"[结果] CSV(RL train) → cg_mamba_predictions_rl_train.csv  ({len(y_true_kw_rl_tr)} 行)")

    plot_prediction(y_true_kw, y_pred_kw, RESULT_DIR)
    plot_loss(train_hist, val_hist, RESULT_DIR)

    # 可视化：训练后的学习权重 vs 因果先验
    plot_learned_weights(model, X_te[:512], causal_weights, RESULT_DIR)

    torch.save(model.state_dict(),
               os.path.join(RESULT_DIR, "cg_mamba_model.pt"))
    print(f"[结果] 模型权重 → {RESULT_DIR}/cg_mamba_model.pt")

    print(f"\n{'='*55}")
    print(f" ✅ CG-Mamba 训练完成，结果保存于: {RESULT_DIR}/")
    print(f"{'='*55}\n")

    sys.stdout = logger.terminal
    logger.close()
    print(f"[日志] 训练记录已保存: {RESULT_DIR}/training_log.txt")

    return y_true_kw, y_pred_kw, m


if __name__ == "__main__":
    main()
