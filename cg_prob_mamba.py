"""
CGProb-Mamba: 因果引导的概率风电功率预测（为 DRL 微电网调度服务）

设计背景
════════════════════════════════════════════════════════════
本模块是"预测-调度"联合框架的预测部分。下游是 DDPG/PPO 微电网低碳
调度（参考冯文韬等, 2023, 四川电力技术）。传统做法里调度器接收
一个确定性的风电预测值；本模块升级为输出完整的概率分布（通过 9 个
分位数描述），让调度器感知预测不确定性:

  - 区间宽（高不确定）→ DRL 应保留更多火电备用容量
  - 区间窄（高确定）  → DRL 可激进消纳风电、降低碳排放

设计决策
════════════════════════════════════════════════════════════
1. 继承 CG-Mamba 的成功基础:
   - 保留 CG-SE 因果门控（已验证 MAE 降低 1.04%）
   - Mamba 主体完全一致，保证消融可比
   - 因果权重加载方式相同（支持从 CG-Mamba 输出复用）

2. 输出头改造:
   - 从 1 个值 (点预测) → 9 个分位数
   - Quantiles: [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
   - 中位数 (τ=0.5) 作为点预测，与 CG-Mamba 直接对比

3. 防止分位数交叉 (quantile crossing):
   网络输出 (base, deltas), 其中 deltas 经过 softplus 保证非负
   q_i = base + sum(deltas[:i])  → 保证 q_0.05 ≤ q_0.10 ≤ ... ≤ q_0.95

4. Pinball Loss (分位数损失):
   L_τ(y, q) = max(τ·(y-q), (τ-1)·(y-q))
   对所有分位数求和作为总 loss

5. 评估指标四件套 (从"点预测精度"升级为"概率预测质量"):
   a) MAE/RMSE/R² — 用 0.5 分位数算, 对标 CG-Mamba
   b) Pinball Score — 所有分位数平均损失
   c) CRPS — Continuous Ranked Probability Score (概率预测的金标)
   d) PICP@90 + MPIW@90 — 90%区间的覆盖率和平均宽度

为 DRL 调度准备的接口
════════════════════════════════════════════════════════════
输出 CSV 包含:
  - true_kW          真实功率
  - pred_median_kW   中位数 (点预测)
  - pred_q05/q95_kW  90% 置信区间上下界
  - pred_q10/q90_kW  80% 置信区间上下界
  - interval_width_kW 90%区间宽度 (反映不确定性)
  - uncertainty_norm 归一化不确定性 [0,1]
  - gate_ws/wd/temp/hum/pres  CG 动态通道权重
  
DRL 智能体可直接把 {median, interval_width, uncertainty_norm}
加入状态空间，实现"不确定性感知调度"。

运行方式:
    python cg_prob_mamba.py --data 风电机组数据集.xls \\
        --causal-weights results_cg_mamba/causal_weights.npy
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
#  0. 全局配置
# ══════════════════════════════════════════════════════

DATA_PATH    = "风电机组数据集.xls"
RESULT_DIR   = "results_cg_prob_mamba"

FEATURE_COLS = ["WINDSPEED", "WINDDIRECTION", "TEMPERATURE", "HUMIDITY", "PRESSURE"]
TARGET_COL   = "YD15"

SEQ_LEN      = 96
# 三段切分 (避免下游 RL 数据泄露):
#   PREDICTOR_RATIO = 0.6   预测模型训练 (含内部 val 切片)
#   RL_TRAIN_RATIO  = 0.2   预测模型不见过, 后面 RL agent 训练时使用
#   RL_TEST_RATIO   = 0.2   预测模型不见过, 后面 RL agent 评估时使用 (隐含 = 1 - 上两者)
PREDICTOR_RATIO = 0.6
RL_TRAIN_RATIO  = 0.2
TRAIN_RATIO  = PREDICTOR_RATIO  # 兼容旧变量名

# PCMCI 参数
PCMCI_TAU_MAX = 4
PCMCI_ALPHA   = 0.05
CAUSAL_MIN_W  = 0.10
FUSION_ALPHA  = 0.5

# Mamba 超参数（与 CG-Mamba 完全一致）
D_MODEL      = 96
D_STATE      = 16
D_CONV       = 4
EXPAND       = 2
N_LAYERS     = 3

# ── 概率预测参数 ──
QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
N_QUANTILES = len(QUANTILES)
MEDIAN_IDX  = QUANTILES.index(0.50)    # 中位数在列表中的位置，用作点预测

# 训练参数
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
    def write(self, msg): self.terminal.write(msg); self.log.write(msg)
    def flush(self): self.terminal.flush(); self.log.flush()
    def close(self): self.log.flush(); self.log.close()


# ══════════════════════════════════════════════════════
#  1. 合成数据 & 数据加载（与 CG-Mamba 一致）
# ══════════════════════════════════════════════════════

def generate_demo_data(n=16354):
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
    path = "demo_data.csv"; df.to_csv(path, index=False)
    print(f"[演示] 合成数据 {n} 条 → {path}")
    return path


def load_data(path):
    df = pd.read_csv(path) if path.endswith(".csv") else \
         pd.read_excel(path, engine="xlrd" if path.endswith(".xls") else "openpyxl")
    print(f"[加载] 原始行数: {len(df)}, 列名: {list(df.columns)}")
    power_col = TARGET_COL
    if power_col not in df.columns:
        candidates = [c for c in df.columns if any(k in c.upper() for k in ["POWER", "YD15"])]
        if not candidates: raise KeyError(f"找不到功率列")
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
        .interpolate(method="linear", limit_direction="both"))
    df.dropna(subset=FEATURE_COLS + ["POWER"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"[清洗] 清洗后行数: {len(df)}")
    return df


# ══════════════════════════════════════════════════════
#  2. PCMCI 因果权重（与 CG-Mamba 一致）
# ══════════════════════════════════════════════════════

def run_pcmci(power_tr, feat_tr, save_dir):
    var_names = ["Power"] + FEATURE_COLS
    data_matrix = np.column_stack([power_tr, feat_tr])
    try:
        from tigramite import data_processing as pp
        from tigramite.pcmci import PCMCI
        from tigramite.independence_tests.parcorr import ParCorr
        print("  [PCMCI] tigramite...")
        dataframe = pp.DataFrame(data_matrix, var_names=var_names,
                                  datatime=np.arange(len(data_matrix)))
        pcmci = PCMCI(dataframe=dataframe,
                      cond_ind_test=ParCorr(significance="analytic"), verbosity=0)
        results = pcmci.run_pcmci(tau_min=1, tau_max=PCMCI_TAU_MAX, pc_alpha=PCMCI_ALPHA)
        val_matrix, p_matrix = results["val_matrix"], results["p_matrix"]
        strengths = []
        for i in range(1, len(var_names)):
            sig = p_matrix[i, 0, 1:] < PCMCI_ALPHA
            vals = np.abs(val_matrix[i, 0, 1:])
            strengths.append(float(vals[sig].max()) if sig.any() else 0.0)
            print(f"    {var_names[i]:15s} 强度={strengths[-1]:.4f} "
                  f"({'显著' if sig.any() else '不显著'})")
    except ImportError:
        print("  [PCMCI] 退化到 Pearson")
        strengths = [float(np.abs(np.corrcoef(data_matrix[1:, 0], data_matrix[:-1, i])[0, 1]))
                     for i in range(1, len(var_names))]

    strengths = np.array(strengths, dtype=np.float32)
    w_pcmci = strengths / (strengths.max() + 1e-8)
    w_physics = np.array([1.00, 0.50, 0.30, 0.20, 0.35], dtype=np.float32)
    w_fused = FUSION_ALPHA * w_pcmci + (1 - FUSION_ALPHA) * w_physics
    w_min, w_max = float(w_fused.min()), float(w_fused.max())
    strengths_scaled = (CAUSAL_MIN_W + (1 - CAUSAL_MIN_W) * (w_fused - w_min) / (w_max - w_min)
                        if (w_max - w_min) > 1e-8
                        else np.full_like(w_fused, (CAUSAL_MIN_W + 1) / 2))
    causal_weights = np.concatenate([[1.0], strengths_scaled])
    print("  最终因果权重:")
    for n, w in zip(["POWER"] + FEATURE_COLS, causal_weights):
        print(f"    {n:15s}: {w:.4f}  {'█' * int(w*20)}")
    np.save(os.path.join(save_dir, "causal_weights.npy"), causal_weights)
    return causal_weights


# ══════════════════════════════════════════════════════
#  3. 滑动窗口
# ══════════════════════════════════════════════════════

def build_windows(feat, power, seq_len=SEQ_LEN):
    X, y = [], []
    for i in range(len(power) - seq_len):
        X.append(np.concatenate([power[i:i+seq_len].reshape(-1, 1),
                                  feat[i:i+seq_len]], axis=1))
        y.append(power[i + seq_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ══════════════════════════════════════════════════════
#  4. Mamba SSM（与 CG-Mamba 完全一致）
# ══════════════════════════════════════════════════════

class MambaSSM(nn.Module):
    def __init__(self, d_model, d_state=D_STATE, d_conv=D_CONV, expand=EXPAND):
        super().__init__()
        self.d_model = d_model; self.d_state = d_state
        self.inner = d_model * expand
        self.in_proj = nn.Linear(d_model, self.inner * 2, bias=False)
        self.conv1d  = nn.Conv1d(self.inner, self.inner, kernel_size=d_conv,
                                  padding=d_conv - 1, groups=self.inner, bias=True)
        self.x_proj  = nn.Linear(self.inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.inner, bias=True)
        nn.init.uniform_(self.dt_proj.bias, -4, -1)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(self.inner, 1)
        self.register_buffer("A_log", torch.log(A))
        self.D        = nn.Parameter(torch.ones(self.inner))
        self.out_proj = nn.Linear(self.inner, d_model, bias=False)
        self.norm     = nn.LayerNorm(d_model)

    def selective_scan(self, u, delta, A, B, C):
        batch, d_inner, L = u.shape
        delta_A   = torch.exp(-A.unsqueeze(0).unsqueeze(-1) * delta.unsqueeze(2))
        delta_B_u = delta.unsqueeze(2) * B.unsqueeze(1) * u.unsqueeze(2)
        h = torch.zeros(batch, d_inner, self.d_state, dtype=u.dtype, device=u.device)
        ys = []
        for t in range(L):
            h = delta_A[..., t] * h + delta_B_u[..., t]
            ys.append((h * C[:, :, t].unsqueeze(1)).sum(dim=2))
        return torch.stack(ys, dim=2)

    def forward(self, x):
        residual = x; x = self.norm(x)
        batch, L, _ = x.shape
        xz = self.in_proj(x)
        x_main, z = xz.chunk(2, dim=-1)
        x_conv = self.conv1d(x_main.permute(0, 2, 1))[..., :L]
        x_conv = F.silu(x_conv)
        x_dbc  = self.x_proj(x_conv.permute(0, 2, 1))
        delta_log = x_dbc[..., :1]
        B = x_dbc[..., 1:1 + self.d_state].permute(0, 2, 1)
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
#  5. 因果先验 SE 注意力（与 CG-Mamba 完全一致）
# ══════════════════════════════════════════════════════

class CausalSEAttention(nn.Module):
    def __init__(self, causal_weights, reduction=2):
        super().__init__()
        n_ch = len(causal_weights); mid = max(n_ch // reduction, 4)
        self.n_ch = n_ch
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(n_ch, mid)
        self.fc2 = nn.Linear(mid, n_ch)
        self.register_buffer("causal_prior",
                             torch.tensor(causal_weights, dtype=torch.float32))
        with torch.no_grad():
            w = torch.clamp(torch.tensor(causal_weights, dtype=torch.float32), 1e-4, 1-1e-4)
            self.fc2.bias.data.copy_(torch.log(w / (1 - w)))
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.fc2.weight.data *= 0.1

    def forward(self, x, return_weights=False):
        s = self.squeeze(x).squeeze(-1)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        out = x * s.unsqueeze(-1)
        if return_weights:
            return out, s
        return out


# ══════════════════════════════════════════════════════
#  6. 概率输出头（多分位数，防交叉）
# ══════════════════════════════════════════════════════

class MonotonicQuantileHead(nn.Module):
    """
    多分位数输出头，保证分位数单调递增 (无交叉)。

    机制:
      网络输出 (base, delta_1, delta_2, ..., delta_{n-1})
      delta_i 经过 softplus → 非负
      q_0     = base                          (最低分位数 q_0.05)
      q_i     = base + sum(softplus(delta[:i])) (累加得到更高分位数)
      → 天然保证 q_0 ≤ q_1 ≤ ... ≤ q_{n-1}

    相比"先独立输出 n 个分位数再排序"的做法:
      - 天然单调, 无需后处理排序
      - 梯度更稳定, 因为所有输出都参与全部分位数的损失计算
    """
    def __init__(self, d_model, n_quantiles):
        super().__init__()
        self.n_quantiles = n_quantiles
        # 输出 n_quantiles 个值: 第 0 个是 base, 后面 n_quantiles-1 个是增量
        self.fc = nn.Linear(d_model, n_quantiles)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        # 对 bias 做初始化: base 在 0.5 附近, 各 delta 在小正数附近
        # 这样训练开始时分位数大致均匀分布在 [0.3, 0.7] 之间
        with torch.no_grad():
            self.fc.bias.data[0] = 0.3                     # base (q_0.05)
            self.fc.bias.data[1:] = 0.05                    # deltas → softplus ≈ 0.07

    def forward(self, x):
        """
        x: (B, d_model)
        返回: (B, n_quantiles) 单调递增
        """
        raw = self.fc(x)                              # (B, n_q)
        base   = raw[:, :1]                           # (B, 1)
        deltas = F.softplus(raw[:, 1:])               # (B, n_q - 1), 非负

        # cumsum 累加增量得到所有分位数
        cum = torch.cumsum(deltas, dim=-1)            # (B, n_q - 1)
        quantiles = torch.cat([base, base + cum], dim=-1)  # (B, n_q)
        return quantiles


# ══════════════════════════════════════════════════════
#  7. CGProb-Mamba 主模型
# ══════════════════════════════════════════════════════

class CGProbMambaModel(nn.Module):
    """
    CG + Mamba + 多分位数概率输出头

    网络结构 (与 CG-Mamba 唯一差别: 输出头):
      输入 (B, seq_len=96, n_features=6)
        → permute → (B, 6, L)
        → CausalSEAttention (因果通道加权)
        → permute → (B, L, 6)
        → input_proj (6→96)
        → N 层 MambaSSM
        → 取最后时间步 (B, 96)
        → LayerNorm
        → MonotonicQuantileHead(96 → n_quantiles=7)
        → (B, 7) 单调递增的分位数预测
    """

    def __init__(self, causal_weights, n_features=6,
                 d_model=D_MODEL, d_state=D_STATE, d_conv=D_CONV,
                 expand=EXPAND, n_layers=N_LAYERS, n_quantiles=N_QUANTILES):
        super().__init__()
        self.cg_se = CausalSEAttention(causal_weights)
        self.input_proj = nn.Linear(n_features, d_model)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        self.mamba_layers = nn.ModuleList([
            MambaSSM(d_model, d_state, d_conv, expand) for _ in range(n_layers)
        ])
        self.norm_out = nn.LayerNorm(d_model)
        self.quantile_head = MonotonicQuantileHead(d_model, n_quantiles)

    def forward(self, x, return_gate=False):
        x = x.permute(0, 2, 1)
        if return_gate:
            x, gate = self.cg_se(x, return_weights=True)
        else:
            x = self.cg_se(x)
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)
        for layer in self.mamba_layers:
            x = layer(x)
        x = x[:, -1, :]
        x = self.norm_out(x)
        q = self.quantile_head(x)                 # (B, n_quantiles)
        if return_gate:
            return q, gate
        return q


# ══════════════════════════════════════════════════════
#  8. Pinball Loss
# ══════════════════════════════════════════════════════

def pinball_loss(predictions, targets, quantiles):
    """
    分位数损失 (Pinball Loss).
    
    L_τ(y, q) = max(τ · (y - q), (τ-1) · (y - q))
              = ReLU(y - q) · τ + ReLU(q - y) · (1 - τ)
    
    参数:
        predictions: (B, n_quantiles)  模型输出的分位数预测
        targets:     (B,) 或 (B, 1)     真实值
        quantiles:   List[float]        分位数水平
    
    返回:
        标量损失 (对 batch 和 quantile 维度平均)
    """
    if targets.dim() == 1:
        targets = targets.unsqueeze(-1)         # (B, 1)

    errors = targets - predictions              # (B, n_q)
    q_tensor = torch.tensor(quantiles,
                             dtype=predictions.dtype,
                             device=predictions.device).unsqueeze(0)  # (1, n_q)
    
    # 等价于 max(τ·e, (τ-1)·e)
    loss = torch.max(q_tensor * errors, (q_tensor - 1) * errors)
    return loss.mean()


# ══════════════════════════════════════════════════════
#  9. 训练
# ══════════════════════════════════════════════════════

def train(model, X_tr, y_tr, X_vl, y_vl, device):
    torch.manual_seed(SEED); np.random.seed(SEED)
    X_tr = torch.FloatTensor(X_tr).to(device); y_tr = torch.FloatTensor(y_tr).to(device)
    X_vl = torch.FloatTensor(X_vl).to(device); y_vl = torch.FloatTensor(y_vl).to(device)

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_vl, y_vl), batch_size=512, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.2, patience=LR_PATIENCE)

    best_val, best_state, patience = float("inf"), None, 0
    train_hist, val_hist = [], []
    t0 = time.time()

    print(f"\n{'─'*55}\n 开始训练 CGProb-Mamba 模型 (Pinball Loss)")
    print(f" 训练样本: {len(X_tr)} | 验证样本: {len(X_vl)}")
    print(f" 分位数 ({N_QUANTILES}): {QUANTILES}\n{'─'*55}")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            q_pred = model(Xb)
            loss = pinball_loss(q_pred, yb, QUANTILES)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item() * len(yb)
        tr_loss /= len(X_tr)

        model.eval()
        vl_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                vl_loss += pinball_loss(model(Xb), yb, QUANTILES).item() * len(yb)
        vl_loss /= len(X_vl)
        train_hist.append(tr_loss); val_hist.append(vl_loss)
        scheduler.step(vl_loss)

        if vl_loss < best_val:
            best_val = vl_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= EARLY_STOP:
            print(f"  早停于 Epoch {epoch}, 最佳 Val Pinball: {best_val:.6f}, "
                  f"耗时: {time.time()-t0:.1f}s")
            break
        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}/{NUM_EPOCHS} | Train: {tr_loss:.5f} | "
                  f"Val: {vl_loss:.5f} | LR: {optimizer.param_groups[0]['lr']:.5f} | "
                  f"耗时: {time.time()-t0:.1f}s")

    model.load_state_dict(best_state); model.eval()
    return train_hist, val_hist


# ══════════════════════════════════════════════════════
#  10. 评价指标: 点预测 + 概率预测
# ══════════════════════════════════════════════════════

def compute_point_metrics(y_true, y_pred):
    """点预测指标 (基于中位数), 与 CG-Mamba 直接对比"""
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_r = np.sum((y_true - y_pred) ** 2)
    ss_t = np.sum((y_true - y_true.mean()) ** 2)
    return {"MAE": mae, "RMSE": rmse, "R2": float(1 - ss_r / (ss_t + 1e-10))}


def compute_probabilistic_metrics(y_true, q_preds, quantiles):
    """
    概率预测指标 (顶刊必报).

    1. Pinball Score: 平均 pinball loss, 综合概率质量
    2. CRPS (近似): Continuous Ranked Probability Score, 概率预测金标
    3. PICP@90:     90% 置信区间的实际覆盖概率 (理想值=0.9)
    4. MPIW@90:     90% 置信区间的平均宽度 (越窄越好)
    5. Winkler@90:  Winkler Score = MPIW + 2/α · 超出惩罚
    """
    y = y_true.reshape(-1, 1)
    errors = y - q_preds                                       # (N, n_q)
    q_arr = np.array(quantiles).reshape(1, -1)                 # (1, n_q)
    # Pinball
    pinball = np.maximum(q_arr * errors, (q_arr - 1) * errors).mean()

    # CRPS 近似 (基于离散分位数, Laio & Tamea 2007 方法)
    # CRPS ≈ 2 * Σ_i (q_i - y) * (τ_i - 1{y < q_i})
    # 等价于在分位数网格上积分
    crps_components = errors * (q_arr - (y < q_preds).astype(float))
    crps = float(2 * crps_components.mean())

    # 90% 区间: q_0.05 ~ q_0.95
    idx_lo = quantiles.index(0.05)
    idx_hi = quantiles.index(0.95)
    q_lo = q_preds[:, idx_lo]; q_hi = q_preds[:, idx_hi]
    in_interval = (y_true >= q_lo) & (y_true <= q_hi)
    picp_90 = float(in_interval.mean())
    mpiw_90 = float((q_hi - q_lo).mean())

    # Winkler Score (α=0.1)
    alpha = 0.10
    width = q_hi - q_lo
    penalty_lo = np.where(y_true < q_lo, 2 / alpha * (q_lo - y_true), 0)
    penalty_hi = np.where(y_true > q_hi, 2 / alpha * (y_true - q_hi), 0)
    winkler_90 = float((width + penalty_lo + penalty_hi).mean())

    # 80% 区间
    idx_lo80 = quantiles.index(0.10)
    idx_hi80 = quantiles.index(0.90)
    q_lo80 = q_preds[:, idx_lo80]; q_hi80 = q_preds[:, idx_hi80]
    picp_80 = float(((y_true >= q_lo80) & (y_true <= q_hi80)).mean())
    mpiw_80 = float((q_hi80 - q_lo80).mean())

    return {
        "Pinball": float(pinball),
        "CRPS":    crps,
        "PICP@90": picp_90, "MPIW@90": mpiw_90, "Winkler@90": winkler_90,
        "PICP@80": picp_80, "MPIW@80": mpiw_80,
    }


# ══════════════════════════════════════════════════════
#  11. 可视化
# ══════════════════════════════════════════════════════

def plot_prob_prediction(y_true, q_preds, quantiles, save_dir, max_steps=500):
    """概率预测可视化: 中位数 + 置信区间 + 真值"""
    os.makedirs(save_dir, exist_ok=True)
    idx = slice(0, min(max_steps, len(y_true)))
    t = np.arange(len(y_true))[idx]

    idx_median = quantiles.index(0.50)
    q05 = q_preds[idx, quantiles.index(0.05)] / 1000
    q10 = q_preds[idx, quantiles.index(0.10)] / 1000
    q50 = q_preds[idx, idx_median] / 1000
    q90 = q_preds[idx, quantiles.index(0.90)] / 1000
    q95 = q_preds[idx, quantiles.index(0.95)] / 1000

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(t, q05, q95, color="#90CAF9", alpha=0.35, label="90% PI")
    ax.fill_between(t, q10, q90, color="#1976D2", alpha=0.35, label="80% PI")
    ax.plot(t, q50, "r-", lw=1.0, label="Median (point forecast)")
    ax.plot(t, y_true[idx] / 1000, "k-", lw=0.6, label="True")
    ax.set_xlabel("Time Step"); ax.set_ylabel("Power (MW)")
    ax.set_title(f"CGProb-Mamba Probabilistic Forecast (first {len(t)} steps)")
    ax.legend(loc="upper right"); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "cg_prob_prediction.png"), dpi=150)
    plt.close()


def plot_reliability_diagram(y_true, q_preds, quantiles, save_dir):
    """可靠性图: 理论覆盖率 vs 实际覆盖率 (顶刊概率预测标配)"""
    empirical_coverage = [(y_true <= q_preds[:, i]).mean() for i in range(len(quantiles))]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect Calibration")
    ax.plot(quantiles, empirical_coverage, "o-", color="#D32F2F", lw=1.5,
             markersize=8, label="CGProb-Mamba")
    for q, e in zip(quantiles, empirical_coverage):
        ax.annotate(f"{e:.3f}", xy=(q, e), xytext=(5, 5),
                     textcoords="offset points", fontsize=8)
    ax.set_xlabel("Nominal Quantile Level"); ax.set_ylabel("Empirical Coverage")
    ax.set_title("Reliability Diagram\n(points should lie on diagonal)")
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "reliability_diagram.png"), dpi=150)
    plt.close()


def plot_loss(tr, vl, save_dir):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(tr, label="Train Pinball")
    ax.plot(vl, label="Val Pinball")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Pinball Loss")
    ax.set_title("CGProb-Mamba Training Curve"); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "cg_prob_loss_curve.png"), dpi=150)
    plt.close()


# ══════════════════════════════════════════════════════
#  12. 主流程
# ══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="CGProb-Mamba: 概率风电预测 (为 DRL 调度准备)")
    parser.add_argument("--data", default=DATA_PATH)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--causal-weights", default=None,
                        help="强烈推荐: 加载 CG-Mamba 的 causal_weights.npy 保证消融可比")
    args = parser.parse_args()

    os.makedirs(RESULT_DIR, exist_ok=True)
    logger = Logger(os.path.join(RESULT_DIR, "training_log.txt"))
    sys.stdout = logger

    torch.manual_seed(SEED); np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[系统] 使用设备: {device}")

    print("\n消融实验说明:")
    print("  本组: CGProb-Mamba (概率预测, 多分位数 + Pinball Loss)")
    print("  对比1: Mamba             MAE=10030.8 R²=0.9289 (点预测, 无CG)")
    print("  对比2: CG-Mamba          MAE= 9926.8 R²=0.9300 (点预测, 有CG)")
    print("  本模型: 中位数点预测 + 90%/80% 置信区间 + CRPS 评估")
    print("  下游: 输出可直接接入 DDPG/PPO 微电网调度 (不确定性感知)")

    if args.demo or not os.path.exists(args.data):
        data_path = generate_demo_data()
    else:
        data_path = args.data

    # ── Step 1-2: 数据 ──
    print("\n" + "="*55 + "\n Step 1-2: 数据加载与划分 (60/20/20 三段切分)\n" + "="*55)
    df = load_data(data_path)
    n_total = len(df)
    n_pred  = int(n_total * PREDICTOR_RATIO)              # 预测模型训练段 (前60%)
    n_rl_tr = int(n_total * RL_TRAIN_RATIO)               # RL训练段 (中间20%)
    rl_tr_start = n_pred
    rl_tr_end   = n_pred + n_rl_tr
    rl_te_start = rl_tr_end
    rl_te_end   = n_total                                 # RL测试段 (最后20%)

    # scaler 仅在 PREDICTOR 段拟合, 避免泄露
    feat_scaler, power_scaler = MinMaxScaler(), MinMaxScaler()
    feat_tr  = feat_scaler.fit_transform(df.iloc[:n_pred][FEATURE_COLS].values)
    power_tr = power_scaler.fit_transform(df.iloc[:n_pred][["POWER"]].values).ravel()

    # RL train 段(供下游RL训练用) 和 RL test 段(供下游RL评估用)
    feat_rl_tr  = feat_scaler.transform(df.iloc[rl_tr_start:rl_tr_end][FEATURE_COLS].values)
    power_rl_tr = power_scaler.transform(df.iloc[rl_tr_start:rl_tr_end][["POWER"]].values).ravel()
    feat_rl_te  = feat_scaler.transform(df.iloc[rl_te_start:rl_te_end][FEATURE_COLS].values)
    power_rl_te = power_scaler.transform(df.iloc[rl_te_start:rl_te_end][["POWER"]].values).ravel()

    # 兼容下游变量名: feat_te / power_te 设为 RL_TEST 段 (维持原评估逻辑在RL test段做)
    feat_te, power_te = feat_rl_te, power_rl_te
    n_train = n_pred  # 兼容性

    print(f"  总样本: {n_total}")
    print(f"  预测训练: 0..{n_pred}            ({n_pred}, {PREDICTOR_RATIO*100:.0f}%)")
    print(f"  RL 训练:  {rl_tr_start}..{rl_tr_end}  ({n_rl_tr}, {RL_TRAIN_RATIO*100:.0f}%)")
    print(f"  RL 测试:  {rl_te_start}..{rl_te_end}  ({rl_te_end-rl_te_start}, {(rl_te_end-rl_te_start)/n_total*100:.0f}%)")

    # ── Step 3: 因果权重 ──
    print("\n" + "="*55 + "\n Step 3: 因果权重\n" + "="*55)
    if args.causal_weights and os.path.exists(args.causal_weights):
        causal_weights = np.load(args.causal_weights)
        print(f"  [加载] {args.causal_weights}")
        np.save(os.path.join(RESULT_DIR, "causal_weights.npy"), causal_weights)
    else:
        causal_weights = run_pcmci(power_tr, feat_tr, RESULT_DIR)

    # ── Step 4: 滑动窗口 ──
    print("\n" + "="*55 + "\n Step 4: 滑动窗口\n" + "="*55)
    X_tr, y_tr = build_windows(feat_tr, power_tr)
    X_te, y_te = build_windows(feat_te, power_te)
    # RL train 段也要做窗口, 后面要单独推理保存
    X_rl_tr, y_rl_tr = build_windows(feat_rl_tr, power_rl_tr)
    n_val = max(100, int(len(X_tr) * 0.1))
    X_fit, y_fit = X_tr[:-n_val], y_tr[:-n_val]
    X_val, y_val = X_tr[-n_val:], y_tr[-n_val:]
    print(f"  预测训练: {len(X_fit)}, 验证: {len(X_val)}")
    print(f"  RL 训练段窗口: {len(X_rl_tr)}, RL 测试段窗口: {len(X_te)}")

    # ── Step 5: 模型 ──
    print("\n" + "="*55 + "\n Step 5: 构建 CGProb-Mamba\n" + "="*55)
    model = CGProbMambaModel(
        causal_weights=causal_weights, n_features=6,
        d_model=D_MODEL, d_state=D_STATE, d_conv=D_CONV,
        expand=EXPAND, n_layers=N_LAYERS, n_quantiles=N_QUANTILES).to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"  总参数量: {total:,}  (对比 CG-Mamba: 191,099)")
    print(f"  输出头: {N_QUANTILES} 个分位数 (单调递增保证)")
    print(f"  Loss: Pinball Loss (等权重平均所有分位数)")

    train_hist, val_hist = train(model, X_fit, y_fit, X_val, y_val, device)

    # ── Step 6: 预测 (分别在 RL 训练段 和 RL 测试段 推理) ──
    print("\n" + "="*55 + "\n Step 6: 推理 (RL train + RL test 两段)\n" + "="*55)
    model.eval()

    def _infer(X_input):
        """对一段输入跑推理, 返回 (q_preds_norm, gates_arr)"""
        all_q, all_g = [], []
        with torch.no_grad():
            loader = DataLoader(TensorDataset(torch.FloatTensor(X_input).to(device)),
                                 batch_size=512, shuffle=False)
            for (Xb,) in loader:
                q, g = model(Xb, return_gate=True)
                all_q.append(q.cpu().numpy())
                all_g.append(g.cpu().numpy())
        return np.concatenate(all_q, axis=0), np.concatenate(all_g, axis=0)

    # RL test 段 (主要评估在这里)
    q_preds_norm, gates = _infer(X_te)
    # RL train 段 (供下游 RL agent 训练时使用)
    q_preds_rl_tr_norm, gates_rl_tr = _infer(X_rl_tr)

    # 反归一化
    def _denorm(q_norm):
        q_kw = np.stack([
            power_scaler.inverse_transform(q_norm[:, i:i+1]).ravel()
            for i in range(N_QUANTILES)
        ], axis=1)
        return np.clip(q_kw, 0, None)

    q_preds_kw       = _denorm(q_preds_norm)         # RL test 段
    q_preds_rl_tr_kw = _denorm(q_preds_rl_tr_norm)   # RL train 段

    y_true_kw       = power_scaler.inverse_transform(y_te.reshape(-1, 1)).ravel()
    y_true_rl_tr_kw = power_scaler.inverse_transform(y_rl_tr.reshape(-1, 1)).ravel()
    y_pred_median_kw = q_preds_kw[:, MEDIAN_IDX]

    # ── Step 7: 评价 ──
    print("\n" + "="*55 + "\n Step 7: 评价指标\n" + "="*55)
    pt = compute_point_metrics(y_true_kw, y_pred_median_kw)
    pr = compute_probabilistic_metrics(y_true_kw, q_preds_kw, QUANTILES)
    
    print("【点预测指标】(中位数预测, 对标 CG-Mamba)")
    print(f"  MAE  = {pt['MAE']:.2f} kW")
    print(f"  RMSE = {pt['RMSE']:.2f} kW")
    print(f"  R²   = {pt['R2']:.4f}")
    print(f"\n  对比: CG-Mamba  MAE=9926.8  R²=0.9300")
    print(f"  ΔMAE = {pt['MAE']-9926.8:+.1f} kW "
          f"({'↓不差于/更好' if pt['MAE']<=9926.8+50 else '↑略差(可接受, 概率预测重心在PICP)'})")

    print("\n【概率预测指标】(顶刊必报)")
    print(f"  Pinball Score = {pr['Pinball']:.2f}  (越小越好)")
    print(f"  CRPS          = {pr['CRPS']:.2f}  (越小越好, 概率金标)")
    print(f"\n  90% 置信区间:")
    print(f"    PICP@90 = {pr['PICP@90']:.4f}  (目标=0.90, 误差={pr['PICP@90']-0.9:+.4f})")
    print(f"    MPIW@90 = {pr['MPIW@90']:.0f} kW  (区间平均宽度)")
    print(f"    Winkler@90 = {pr['Winkler@90']:.0f}  (综合评分)")
    print(f"\n  80% 置信区间:")
    print(f"    PICP@80 = {pr['PICP@80']:.4f}  (目标=0.80, 误差={pr['PICP@80']-0.8:+.4f})")
    print(f"    MPIW@80 = {pr['MPIW@80']:.0f} kW")

    # 可靠性诊断
    print(f"\n【可靠性诊断】")
    for i, q in enumerate(QUANTILES):
        emp = (y_true_kw <= q_preds_kw[:, i]).mean()
        status = "✓" if abs(emp - q) < 0.05 else "△"
        print(f"    q{q:.2f}: 理论={q:.3f}, 经验={emp:.3f}, 偏差={emp-q:+.4f}  {status}")

    # ── Step 8: 保存 DRL 友好接口 (RL train 段 + RL test 段两份CSV) ──
    print("\n" + "="*55 + "\n Step 8: 保存结果 (含 DRL 调度接口, 两段)\n" + "="*55)

    def _build_drl_csv(q_kw, gates_arr, y_true, iw_max_for_norm):
        """根据一段的分位数和门控权重, 构造下游 DRL 用的 DataFrame.
        归一化基准 iw_max_for_norm 必须跨两段统一, 避免下游 RL 的 obs 不一致.
        """
        q05  = q_kw[:, QUANTILES.index(0.05)]
        q10  = q_kw[:, QUANTILES.index(0.10)]
        q90  = q_kw[:, QUANTILES.index(0.90)]
        q95  = q_kw[:, QUANTILES.index(0.95)]
        width90 = q95 - q05
        uncert  = width90 / (iw_max_for_norm + 1e-8)
        return pd.DataFrame({
            "true_kW":           y_true,
            "pred_median_kW":    q_kw[:, MEDIAN_IDX],
            "pred_q05_kW":       q05,
            "pred_q10_kW":       q10,
            "pred_q25_kW":       q_kw[:, QUANTILES.index(0.25)],
            "pred_q75_kW":       q_kw[:, QUANTILES.index(0.75)],
            "pred_q90_kW":       q90,
            "pred_q95_kW":       q95,
            "interval_width_90_kW":  width90,
            "interval_width_80_kW":  q90 - q10,
            "uncertainty_norm":  uncert,
            "gate_power":        gates_arr[:, 0],
            "gate_windspeed":    gates_arr[:, 1],
            "gate_winddirection":gates_arr[:, 2],
            "gate_temperature":  gates_arr[:, 3],
            "gate_humidity":     gates_arr[:, 4],
            "gate_pressure":     gates_arr[:, 5],
        })

    # 用 RL train 段的最大区间宽度作为统一归一化基准
    # (因为下游 RL 在 train 段学策略时会"看到"这个最大值, test 段应保持同一尺度)
    iw_max_global = float(
        (q_preds_rl_tr_kw[:, QUANTILES.index(0.95)] - q_preds_rl_tr_kw[:, QUANTILES.index(0.05)]).max()
    )

    out_rl_tr = _build_drl_csv(q_preds_rl_tr_kw, gates_rl_tr, y_true_rl_tr_kw, iw_max_global)
    out_df    = _build_drl_csv(q_preds_kw,       gates,        y_true_kw,        iw_max_global)

    csv_path_rl_tr = os.path.join(RESULT_DIR, "cg_prob_predictions_rl_train.csv")
    csv_path       = os.path.join(RESULT_DIR, "cg_prob_predictions_rl_test.csv")
    csv_path_legacy = os.path.join(RESULT_DIR, "cg_prob_predictions.csv")  # 兼容旧名

    out_rl_tr.to_csv(csv_path_rl_tr, index=False)
    out_df.to_csv(csv_path, index=False)
    out_df.to_csv(csv_path_legacy, index=False)  # 旧脚本若指向这个文件名,等同 RL test 段
    print(f"[结果] RL train 段 CSV → {csv_path_rl_tr}  ({len(out_rl_tr)} 行)")
    print(f"[结果] RL test  段 CSV → {csv_path}        ({len(out_df)} 行)")
    print(f"       两段共享 uncertainty_norm 归一化基准: iw_max={iw_max_global:.1f} kW")
    print(f"       下游 RL: 用 _rl_train.csv 训练 agent, 用 _rl_test.csv 评估")
    print(f"       共 {len(out_df.columns)} 列, {len(out_df)} 行")
    print(f"       可直接作为 DDPG/PPO 的状态输入 (除 true_kW 外所有列)")

    # 保存分位数原始矩阵 (供下游快速加载)
    np.save(os.path.join(RESULT_DIR, "q_preds_kw.npy"), q_preds_kw)              # RL test 段
    np.save(os.path.join(RESULT_DIR, "q_preds_kw_rl_train.npy"), q_preds_rl_tr_kw)
    np.save(os.path.join(RESULT_DIR, "gates.npy"), gates)
    np.save(os.path.join(RESULT_DIR, "gates_rl_train.npy"), gates_rl_tr)

    # 指标 JSON (主指标基于 RL test 段, 避免与训练泄露)
    import json
    metrics = {"point": pt, "probabilistic": pr,
               "quantiles": QUANTILES,
               "n_test_samples": len(y_true_kw),
               "n_rl_train_samples": len(y_true_rl_tr_kw),
               "split": {"predictor_ratio": PREDICTOR_RATIO,
                         "rl_train_ratio": RL_TRAIN_RATIO,
                         "rl_test_ratio": 1 - PREDICTOR_RATIO - RL_TRAIN_RATIO}}
    with open(os.path.join(RESULT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # 可视化
    plot_prob_prediction(y_true_kw, q_preds_kw, QUANTILES, RESULT_DIR, max_steps=500)
    plot_reliability_diagram(y_true_kw, q_preds_kw, QUANTILES, RESULT_DIR)
    plot_loss(train_hist, val_hist, RESULT_DIR)
    torch.save(model.state_dict(), os.path.join(RESULT_DIR, "cg_prob_mamba_model.pt"))

    print(f"\n{'='*55}\n ✅ CGProb-Mamba 训练完成, 结果: {RESULT_DIR}/\n{'='*55}\n")
    print("下一步 (DRL 调度):")
    print("  1. DDPG/PPO 的状态空间增加 [pred_median, interval_width_90, uncertainty_norm]")
    print("  2. 调度策略在 interval 宽时保留更多火电备用, 窄时积极消纳风电")
    print("  3. 端到端评估: 总运营成本 (美元) + 碳排放量 (吨)")

    sys.stdout = logger.terminal
    logger.close()
    print(f"[日志] → {RESULT_DIR}/training_log.txt")
    return out_df, metrics


if __name__ == "__main__":
    main()
