"""
Mamba 基线模型（纯 PyTorch 实现，无需 CUDA 自定义算子）
对应论文消融试验：替换 LSTM 基线中的 LSTM 层为 Mamba SSM

Mamba 核心原理：
  选择性状态空间模型（Selective State Space Model）
  离散化 SSM：
    h_t = Ā·h_{t-1} + B̄·x_t   （状态更新）
    y_t = C·h_t                  （输出投影）
  其中 Δ（步长）、B、C 都是输入相关的（selective），
  A 固定为对角负矩阵（HiPPO 初始化）

运行方式:
    python mamba_baseline_aligned.py --data 风电机组数据集.xls
    python mamba_baseline_aligned.py --demo
"""

import os
import sys
import argparse
import math
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


# ══════════════════════════════════════════════════════
#  0. 全局配置
# ══════════════════════════════════════════════════════

DATA_PATH    = "风电机组数据集.xls"
RESULT_DIR   = "results_mamba"

FEATURE_COLS = ["WINDSPEED", "WINDDIRECTION", "TEMPERATURE", "HUMIDITY", "PRESSURE"]
TARGET_COL   = "YD15"

SEQ_LEN      = 96
TRAIN_RATIO  = 0.6   # 60% 用于预测模型训练
RL_TRAIN_RATIO = 0.2  # 20% 作为 RL agent 训练时使用 (预测模型不见)
                      # 剩下 20% 自动作为 RL test 段

# Mamba 超参数
# ── 消融公平对齐（与 CEEMDAN+Mamba/MSCNN+Mamba/CG-MSCNN+Mamba 完全一致）───
# 原配置: d_model=64, n_layers=2, LR=0.01, EARLY_STOP=20
# 这种轻量配置在 Mamba baseline 上 R²=0.9277，但与其他模型不可比。
# 本次用下游模型的统一配置重跑：d_model=96, n_layers=3, LR=0.005, EARLY_STOP=30
# ─────────────────────────────────────────────────────────────────────
D_MODEL      = 96      # 原 64 → 96
D_STATE      = 16      # SSM 状态维度 N
D_CONV       = 4       # 局部卷积核长度
EXPAND       = 2       # 内部扩展因子（inner_dim = D_MODEL * EXPAND）
N_LAYERS     = 3       # 原 2 → 3

BATCH_SIZE   = 64
NUM_EPOCHS   = 100
LR           = 0.005   # 原 0.01 → 0.005
LR_PATIENCE  = 8       # 原 10 → 8
EARLY_STOP   = 30      # 原 20 → 30
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
    power = np.clip(0.5 * wind**3 * np.random.uniform(0.8, 1.2, n), 0, 200)
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
        candidates = [c for c in df.columns if any(k in c.upper() for k in ["POWER","YD15"])]
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
#  3. 滑动窗口构建
# ══════════════════════════════════════════════════════

def build_windows(feat: np.ndarray, power: np.ndarray, seq_len: int = SEQ_LEN):
    """
    输入: feat(T,5), power(T,)
    输出: X(N, seq_len, 6), y(N,)
      6 = 1个历史功率 + 5个气象特征
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
#  4. 纯 PyTorch Mamba 实现
# ══════════════════════════════════════════════════════

class MambaSSM(nn.Module):
    """
    单层 Mamba SSM Block（纯 PyTorch，无 CUDA 自定义算子）

    输入:  (batch, seq_len, d_model)
    输出:  (batch, seq_len, d_model)

    内部流程：
      x → 线性投影 → inner_dim
      ├── 主路径：局部卷积(d_conv) → SiLU → SSM 选择性扫描
      └── 门控路径：线性 → SiLU
      ↓  逐元素相乘（门控）
      → 线性投影回 d_model
    """

    def __init__(self, d_model: int, d_state: int = D_STATE,
                 d_conv: int = D_CONV, expand: int = EXPAND):
        super().__init__()
        self.d_model  = d_model
        self.d_state  = d_state
        self.d_conv   = d_conv
        self.inner    = d_model * expand   # inner_dim

        # 输入投影：一次性生成主路径 + 门控路径
        self.in_proj  = nn.Linear(d_model, self.inner * 2, bias=False)

        # 局部深度卷积（捕获短程时序特征，类似 CNN 的 patch）
        self.conv1d   = nn.Conv1d(
            in_channels  = self.inner,
            out_channels = self.inner,
            kernel_size  = d_conv,
            padding      = d_conv - 1,
            groups       = self.inner,   # 深度可分离
            bias         = True,
        )

        # SSM 参数投影（selective：Δ, B, C 都是输入相关的）
        self.x_proj   = nn.Linear(self.inner, d_state * 2 + 1, bias=False)
        # d_state*2: B 和 C；+1: Δ（步长对数）

        # Δ 步长偏置（初始化为小正值）
        self.dt_proj  = nn.Linear(1, self.inner, bias=True)
        nn.init.uniform_(self.dt_proj.bias, -4, -1)   # log 空间初始化

        # A：固定对角负矩阵（HiPPO 初始化）
        # A[n] = -(n+1)，确保稳定的状态衰减
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)  # (1, N)
        A = A.repeat(self.inner, 1)   # (inner, N)
        self.register_buffer("A_log", torch.log(A))   # 存 log(A)，确保 A 始终为正

        # D：跳跃连接（残差标量）
        self.D         = nn.Parameter(torch.ones(self.inner))

        # 输出投影
        self.out_proj  = nn.Linear(self.inner, d_model, bias=False)

        # LayerNorm
        self.norm      = nn.LayerNorm(d_model)

    def selective_scan(self, u: torch.Tensor,
                       delta: torch.Tensor,
                       A: torch.Tensor,
                       B: torch.Tensor,
                       C: torch.Tensor) -> torch.Tensor:
        """
        选择性状态空间扫描（顺序 for 循环实现，等价于 CUDA 版本）

        参数:
            u:     (batch, inner, seq_len)   输入序列
            delta: (batch, inner, seq_len)   步长 Δ（已经过 softplus）
            A:     (inner, d_state)          固定衰减矩阵（正值）
            B:     (batch, d_state, seq_len) 输入矩阵（selective）
            C:     (batch, d_state, seq_len) 输出矩阵（selective）

        返回:
            y:     (batch, inner, seq_len)
        """
        batch, d_inner, L = u.shape
        d_state = A.shape[1]

        # 离散化：零阶保持（ZOH）
        # Ā = exp(-A * Δ)，shape: (batch, inner, d_state, seq_len)
        delta_A = torch.exp(
            -A.unsqueeze(0).unsqueeze(-1) *
            delta.unsqueeze(2)
        )  # (batch, inner, d_state, L)

        # B̄ = Δ * B，shape: (batch, inner, d_state, seq_len)
        delta_B_u = (
            delta.unsqueeze(2) *
            B.unsqueeze(1) *
            u.unsqueeze(2)
        )  # (batch, inner, d_state, L)

        # 顺序扫描
        h = torch.zeros(batch, d_inner, d_state,
                        dtype=u.dtype, device=u.device)
        ys = []
        for t in range(L):
            h = delta_A[..., t] * h + delta_B_u[..., t]
            # y_t = C_t · h_t，逐时刻点积
            y_t = (h * C[:, :, t].unsqueeze(1)).sum(dim=2)   # (batch, inner)
            ys.append(y_t)

        y = torch.stack(ys, dim=2)   # (batch, inner, L)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        """
        residual = x
        x = self.norm(x)

        batch, L, _ = x.shape

        # ── 输入投影 ──────────────────────────────────────────
        xz = self.in_proj(x)                  # (batch, L, inner*2)
        x_main, z = xz.chunk(2, dim=-1)       # 各 (batch, L, inner)

        # ── 局部卷积 ──────────────────────────────────────────
        x_conv = x_main.permute(0, 2, 1)      # (batch, inner, L)
        x_conv = self.conv1d(x_conv)[..., :L] # causal: 截断到原长
        x_conv = F.silu(x_conv)               # (batch, inner, L)

        # ── SSM 参数投影（selective）────────────────────────────
        # x_proj 的输入先转回 (batch, L, inner)
        x_proj_in = x_conv.permute(0, 2, 1)   # (batch, L, inner)
        x_dbc = self.x_proj(x_proj_in)        # (batch, L, d_state*2+1)

        # 拆分 Δ, B, C
        delta_log = x_dbc[..., :1]            # (batch, L, 1)
        B = x_dbc[..., 1: 1 + self.d_state]   # (batch, L, d_state)
        C = x_dbc[..., 1 + self.d_state:]     # (batch, L, d_state)

        # Δ：通过 dt_proj 扩展到 inner_dim，再 softplus 保证正值
        delta = self.dt_proj(delta_log)        # (batch, L, inner)
        delta = F.softplus(delta)
        delta = delta.permute(0, 2, 1)        # (batch, inner, L)

        B = B.permute(0, 2, 1)                # (batch, d_state, L)
        C = C.permute(0, 2, 1)                # (batch, d_state, L)

        # A：从 log 空间恢复正值
        A = torch.exp(self.A_log).to(x.dtype) # (inner, d_state)

        # ── 选择性扫描 (优先使用 mamba-ssm CUDA kernel) ────────
        if USE_MAMBA_SSM_KERNEL and x_conv.is_cuda:
            # mamba-ssm 期望 A < 0; D 由库内部加, 不需要外面再加
            y = selective_scan_fn(
                x_conv, delta, -A, B, C, D=self.D,
                z=None, delta_bias=None, delta_softplus=False
            )
        else:
            y = self.selective_scan(x_conv, delta, A, B, C)
            y = y + self.D.unsqueeze(0).unsqueeze(-1) * x_conv

        # ── 门控融合 ──────────────────────────────────────────
        y = y * F.silu(z.permute(0, 2, 1))   # (batch, inner, L)

        # ── 输出投影 ──────────────────────────────────────────
        y = y.permute(0, 2, 1)                # (batch, L, inner)
        y = self.out_proj(y)                  # (batch, L, d_model)

        return y + residual


# ══════════════════════════════════════════════════════
#  5. Mamba 回归模型
# ══════════════════════════════════════════════════════

class MambaModel(nn.Module):
    """
    Mamba 时序回归模型

    网络结构:
      输入(batch, seq_len, n_features)
        → 线性投影到 d_model
        → N 层 MambaSSM Block（每层含残差）
        → 取最后一步输出
        → LayerNorm → Linear(1)
        → 输出(batch,)
    """

    def __init__(self, n_features: int = 6,
                 d_model:    int = D_MODEL,
                 d_state:    int = D_STATE,
                 d_conv:     int = D_CONV,
                 expand:     int = EXPAND,
                 n_layers:   int = N_LAYERS):
        super().__init__()

        # 输入投影：将 n_features 映射到 d_model
        self.input_proj = nn.Linear(n_features, d_model)

        # 堆叠 Mamba Blocks
        self.layers = nn.ModuleList([
            MambaSSM(d_model, d_state, d_conv, expand)
            for _ in range(n_layers)
        ])

        # 输出头
        self.norm_out = nn.LayerNorm(d_model)
        self.fc_out   = nn.Linear(d_model, 1)

        # 权重初始化
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, n_features)
        """
        # 投影到模型维度
        x = self.input_proj(x)          # (batch, seq_len, d_model)

        # 逐层 Mamba
        for layer in self.layers:
            x = layer(x)                # (batch, seq_len, d_model)

        # 取最后时间步
        x = x[:, -1, :]                 # (batch, d_model)
        x = self.norm_out(x)
        out = self.fc_out(x).squeeze(-1) # (batch,)
        return out


# ══════════════════════════════════════════════════════
#  6. 训练函数
# ══════════════════════════════════════════════════════

def train(model, X_tr, y_tr, X_vl, y_vl, device):
    import time
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

    print(f"\n{'─'*50}")
    print(f" 开始训练 Mamba 模型")
    print(f" 训练样本: {len(X_tr)} | 验证样本: {len(X_vl)}")
    print(f"{'─'*50}")

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
            best_val   = vl_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1

        if patience_cnt >= EARLY_STOP:
            print(f"  早停于 Epoch {epoch}, 最佳验证损失: {best_val:.6f}")
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
#  7. 评价指标
# ══════════════════════════════════════════════════════

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_r = np.sum((y_true - y_pred) ** 2)
    ss_t = np.sum((y_true - y_true.mean()) ** 2)
    r2   = float(1 - ss_r / (ss_t + 1e-10))
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


# ══════════════════════════════════════════════════════
#  8. 绘图（纵坐标 MW，横坐标全测试集）
# ══════════════════════════════════════════════════════

def plot_prediction(y_true, y_pred, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    x = np.arange(len(y_true))
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(x, y_true / 1000, "b-",  lw=0.7, label="True Power")
    ax.plot(x, y_pred / 1000, "r--", lw=0.7, label="Predicted Power")
    ax.set_xlabel("Time Step"); ax.set_ylabel("Power (MW)")
    ax.set_title("Mamba Prediction"); ax.legend()
    plt.tight_layout()
    path = os.path.join(save_dir, "mamba_prediction.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"[图表] 预测结果图 → {path}")


def plot_loss(train_hist, val_hist, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_hist, label="Train Loss")
    ax.plot(val_hist,   label="Val Loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
    ax.set_title("Mamba Training Curve"); ax.legend()
    plt.tight_layout()
    path = os.path.join(save_dir, "mamba_loss_curve.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"[图表] 损失曲线图 → {path}")


# ══════════════════════════════════════════════════════
#  9. 主流程
# ══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Mamba 基线模型")
    parser.add_argument("--data", default=DATA_PATH)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    os.makedirs(RESULT_DIR, exist_ok=True)
    logger = Logger(os.path.join(RESULT_DIR, "training_log.txt"))
    sys.stdout = logger

    torch.manual_seed(SEED); np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[系统] 使用设备: {device}")

    if args.demo or not os.path.exists(args.data):
        print("[提示] 未找到数据文件，切换演示模式")
        data_path = generate_demo_data()
    else:
        data_path = args.data

    # ── Step 1: 加载数据 ──────────────────────────────
    print("\n" + "="*50)
    print(" Step 1: 加载与清洗数据")
    print("="*50)
    df = load_data(data_path)

    # ── Step 2: 划分 + 归一化 ─────────────────────────
    print("\n" + "="*50)
    print(" Step 2: 数据划分与归一化")
    print("="*50)
    n_total = len(df)
    n_train = int(n_total * TRAIN_RATIO)                    # 60% 预测训练
    n_rl_tr = int(n_total * RL_TRAIN_RATIO)                 # 20% RL 训练段
    rl_tr_start, rl_tr_end = n_train, n_train + n_rl_tr
    rl_te_start, rl_te_end = rl_tr_end, n_total
    print(f"  总样本: {n_total} (60/20/20 三段)")
    print(f"  预测训练: 0..{n_train}")
    print(f"  RL 训练:  {rl_tr_start}..{rl_tr_end}  ({n_rl_tr})")
    print(f"  RL 测试:  {rl_te_start}..{rl_te_end}  ({rl_te_end-rl_te_start})")

    feat_scaler  = MinMaxScaler()
    power_scaler = MinMaxScaler()

    feat_tr  = feat_scaler.fit_transform(df.iloc[:n_train][FEATURE_COLS].values)
    power_tr = power_scaler.fit_transform(df.iloc[:n_train][["POWER"]].values).ravel()
    feat_rl_tr  = feat_scaler.transform(df.iloc[rl_tr_start:rl_tr_end][FEATURE_COLS].values)
    power_rl_tr = power_scaler.transform(df.iloc[rl_tr_start:rl_tr_end][["POWER"]].values).ravel()
    feat_te  = feat_scaler.transform(df.iloc[rl_te_start:rl_te_end][FEATURE_COLS].values)
    power_te = power_scaler.transform(df.iloc[rl_te_start:rl_te_end][["POWER"]].values).ravel()

    # ── Step 3: 滑动窗口 ──────────────────────────────
    print("\n" + "="*50)
    print(" Step 3: 构建96步滑动窗口样本")
    print("="*50)
    X_tr, y_tr = build_windows(feat_tr, power_tr)
    X_te, y_te = build_windows(feat_te, power_te)
    X_rl_tr, y_rl_tr = build_windows(feat_rl_tr, power_rl_tr)
    print(f"  预测训练: X={X_tr.shape}, y={y_tr.shape}")
    print(f"  RL 训练:  X={X_rl_tr.shape}, y={y_rl_tr.shape}")
    print(f"  RL 测试:  X={X_te.shape}, y={y_te.shape}")

    n_val = max(100, int(len(X_tr) * 0.1))
    X_fit, y_fit = X_tr[:-n_val], y_tr[:-n_val]
    X_val, y_val = X_tr[-n_val:], y_tr[-n_val:]
    print(f"  训练子集: {len(X_fit)} | 验证子集: {len(X_val)}")

    # ── Step 4: 构建并训练模型 ────────────────────────
    print("\n" + "="*50)
    print(" Step 4: 构建 Mamba 模型")
    print("="*50)
    n_features = X_tr.shape[2]
    model = MambaModel(
        n_features = n_features,
        d_model    = D_MODEL,
        d_state    = D_STATE,
        d_conv     = D_CONV,
        expand     = EXPAND,
        n_layers   = N_LAYERS,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {total_params:,}")
    print(f"  d_model={D_MODEL}, d_state={D_STATE}, "
          f"d_conv={D_CONV}, expand={EXPAND}, n_layers={N_LAYERS}")

    train_hist, val_hist = train(model, X_fit, y_fit, X_val, y_val, device)

    # ── Step 5: 预测 (RL train + RL test 两段) ──
    print("\n" + "="*50)
    print(" Step 5: 推理 (RL train + RL test 两段)")
    print("="*50)
    model.eval()

    def _infer(X_input):
        with torch.no_grad():
            loader = DataLoader(
                TensorDataset(torch.FloatTensor(X_input).to(device)),
                batch_size=512, shuffle=False)
            return np.concatenate(
                [model(Xb).cpu().numpy() for (Xb,) in loader])

    preds_norm        = _infer(X_te)
    preds_norm_rl_tr  = _infer(X_rl_tr)

    y_pred_kw       = np.clip(power_scaler.inverse_transform(preds_norm.reshape(-1, 1)).ravel(), 0, None)
    y_pred_kw_rl_tr = np.clip(power_scaler.inverse_transform(preds_norm_rl_tr.reshape(-1, 1)).ravel(), 0, None)
    y_true_kw       = power_scaler.inverse_transform(y_te.reshape(-1, 1)).ravel()
    y_true_kw_rl_tr = power_scaler.inverse_transform(y_rl_tr.reshape(-1, 1)).ravel()

    # ── Step 6: 评价指标 ──────────────────────────────
    print("\n" + "="*50)
    print(" Step 6: 评价指标（反归一化后，单位 kW）")
    print("="*50)
    m = compute_metrics(y_true_kw, y_pred_kw)
    print(f"  MAE  = {m['MAE']:.4f} kW  ({m['MAE']/1000:.4f} MW)")
    print(f"  RMSE = {m['RMSE']:.4f} kW  ({m['RMSE']/1000:.4f} MW)")
    print(f"  R²   = {m['R2']:.4f}")

    # ── Step 7: 保存结果 (两段) ──────────────────────────────
    pd.DataFrame({"true_kW": y_true_kw, "pred_kW": y_pred_kw}).to_csv(
        os.path.join(RESULT_DIR, "mamba_predictions.csv"), index=False)
    pd.DataFrame({"true_kW": y_true_kw, "pred_kW": y_pred_kw}).to_csv(
        os.path.join(RESULT_DIR, "mamba_predictions_rl_test.csv"), index=False)
    pd.DataFrame({"true_kW": y_true_kw_rl_tr, "pred_kW": y_pred_kw_rl_tr}).to_csv(
        os.path.join(RESULT_DIR, "mamba_predictions_rl_train.csv"), index=False)
    print(f"\n[结果] CSV(RL test) → mamba_predictions_rl_test.csv  ({len(y_true_kw)} 行)")
    print(f"[结果] CSV(RL train) → mamba_predictions_rl_train.csv  ({len(y_true_kw_rl_tr)} 行)")

    plot_prediction(y_true_kw, y_pred_kw, RESULT_DIR)
    plot_loss(train_hist, val_hist, RESULT_DIR)

    torch.save(model.state_dict(), os.path.join(RESULT_DIR, "mamba_model.pt"))
    print(f"[结果] 模型权重 → {RESULT_DIR}/mamba_model.pt")

    print(f"\n{'='*50}")
    print(f" ✅ Mamba 训练完成，结果保存于: {RESULT_DIR}/")
    print(f"{'='*50}\n")

    sys.stdout = logger.terminal
    logger.close()
    print(f"[日志] 训练记录已保存: {RESULT_DIR}/training_log.txt")

    return y_true_kw, y_pred_kw, m


if __name__ == "__main__":
    main()
