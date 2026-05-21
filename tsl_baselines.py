"""
Time-Series-Library Baselines (Standalone Implementation)
═══════════════════════════════════════════════════════════════════
独立实现 5 个 SOTA 时序预测基线，不依赖 TSL 仓库。
数据管道和训练协议与 CG-Mamba 完全一致，保证公平对比。

实现的模型 (都来自官方仓库对标实现):
  1. DLinear      (ICLR 2023) — 线性基线
  2. PatchTST     (ICLR 2023) — Patch + Channel-Independent Transformer
  3. iTransformer (ICLR 2024) — 倒置 Transformer
  4. TimeMixer    (ICLR 2024) — 多尺度混合
  5. Autoformer   (NeurIPS 2021) — 自相关分解

实验协议 (与 CG-Mamba 完全对齐):
  - 数据: 相同 xls 清洗 (POWER≥0, 线性插值), 13023 条
  - 划分: 8:2 (训练集再切 10% 做 val), 测试集 = 最后 2509 条滑窗样本
  - 归一化: MinMaxScaler (feat 和 power 分别拟合训练集)
  - 滑窗: SEQ_LEN=96, pred_len=1, 通道 = [POWER] + 5气象
  - 训练: batch=64, lr=0.005, epoch≤100, patience=30, seed=42
  - 指标: 反归一化到 kW 空间后计算 MAE/RMSE/R²

运行:
    python tsl_baselines.py --data 15_processed.csv --model all

输出:
    results_tsl_baselines/
        {MODEL}/training_log.txt
        {MODEL}/{model}_predictions.csv       # 与 CG-Mamba 相同格式
        {MODEL}/{model}_model.pt
        comparison_table.csv                  # 汇总
        comparison_table.md                   # 论文级 Markdown 表
"""

import os
import sys
import math
import time
import argparse
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════
#  0. 全局配置 (与 CG-Mamba / Mamba 完全对齐)
# ══════════════════════════════════════════════════════

DATA_PATH    = "风电机组数据集.xls"
RESULT_ROOT  = "results_tsl_baselines"

FEATURE_COLS = ["WINDSPEED", "WINDDIRECTION", "TEMPERATURE", "HUMIDITY", "PRESSURE"]
TARGET_COL   = "POWER"

SEQ_LEN      = 96
PRED_LEN     = 1
LABEL_LEN    = 48        # encoder-decoder 模型 (Autoformer) 需要
TRAIN_RATIO  = 0.6   # 60% 用于预测模型训练
RL_TRAIN_RATIO = 0.2  # 20% 作为 RL agent 训练时使用 (预测模型不见)
                      # 剩下 20% 自动作为 RL test 段

# 通道数: [POWER, WS, WD, TEMP, HUM, PRESSURE] = 6
N_FEATURES   = 6
TARGET_IDX   = 0         # 在 6 维通道中, POWER 位置 (滑窗构造里放第一位)

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
#  1. 数据加载 (与 CG-Mamba 完全一致)
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
        "POWER": np.clip(power * 1000 + np.random.normal(0, 2000, n), 0, None),
    })
    path = "demo_data.csv"; df.to_csv(path, index=False)
    print(f"[演示] 合成数据 {n} 条 → {path}")
    return path


def load_data(path):
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith(".xls"):
        df = pd.read_excel(path, engine="xlrd")
    else:
        df = pd.read_excel(path, engine="openpyxl")
    print(f"[加载] 原始行数: {len(df)}, 列名: {list(df.columns)}")
    # 兼容 YD15 列名
    if TARGET_COL not in df.columns:
        candidates = [c for c in df.columns if any(k in c.upper() for k in ["POWER", "YD15"])]
        if not candidates:
            raise KeyError(f"找不到功率列")
        df.rename(columns={candidates[0]: TARGET_COL}, inplace=True)
        print(f"[加载] 自动识别功率列: {candidates[0]}")
    col_upper = {c.upper(): c for c in df.columns}
    for feat in FEATURE_COLS:
        if feat not in df.columns and feat.upper() in col_upper:
            df.rename(columns={col_upper[feat.upper()]: feat}, inplace=True)
    df = df[df[TARGET_COL] >= 0].copy()
    df[FEATURE_COLS + [TARGET_COL]] = (
        df[FEATURE_COLS + [TARGET_COL]]
        .replace([float("inf"), float("-inf")], float("nan"))
        .interpolate(method="linear", limit_direction="both"))
    df.dropna(subset=FEATURE_COLS + [TARGET_COL], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"[清洗] 清洗后行数: {len(df)}")
    return df


def build_windows(feat, power, seq_len=SEQ_LEN):
    X, y = [], []
    for i in range(len(power) - seq_len):
        X.append(np.concatenate([power[i:i+seq_len].reshape(-1, 1),
                                  feat[i:i+seq_len]], axis=1))
        y.append(power[i + seq_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ══════════════════════════════════════════════════════
#  2. 模型 1: DLinear (ICLR 2023)
# ══════════════════════════════════════════════════════

class SeriesDecomposition(nn.Module):
    """滑动平均分解 (Autoformer/DLinear/TimeMixer 共用)"""
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        # x: (B, L, C)
        pad = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, pad, 1)
        end   = x[:, -1:, :].repeat(1, self.kernel_size - 1 - pad, 1)
        x_padded = torch.cat([front, x, end], dim=1)
        trend = self.avg(x_padded.permute(0, 2, 1)).permute(0, 2, 1)
        seasonal = x - trend
        return seasonal, trend


class DLinear(nn.Module):
    """
    DLinear: "Are Transformers Effective for Time Series Forecasting?" (ICLR 2023)
    架构: 分解 + 两个独立线性层
    """
    name = "DLinear"

    def __init__(self, seq_len=SEQ_LEN, pred_len=PRED_LEN, n_features=N_FEATURES,
                 target_idx=TARGET_IDX, kernel_size=25, individual=False):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.target_idx = target_idx
        self.individual = individual
        self.decomp = SeriesDecomposition(kernel_size)

        if individual:
            self.linear_seasonal = nn.ModuleList([nn.Linear(seq_len, pred_len)
                                                   for _ in range(n_features)])
            self.linear_trend    = nn.ModuleList([nn.Linear(seq_len, pred_len)
                                                   for _ in range(n_features)])
        else:
            self.linear_seasonal = nn.Linear(seq_len, pred_len)
            self.linear_trend    = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # x: (B, L, C)
        seasonal, trend = self.decomp(x)  # (B, L, C)
        seasonal, trend = seasonal.permute(0, 2, 1), trend.permute(0, 2, 1)  # (B, C, L)
        if self.individual:
            s_out = torch.stack([self.linear_seasonal[i](seasonal[:, i]) for i in range(self.n_features)], dim=1)
            t_out = torch.stack([self.linear_trend[i](trend[:, i]) for i in range(self.n_features)], dim=1)
        else:
            s_out = self.linear_seasonal(seasonal)
            t_out = self.linear_trend(trend)
        out = s_out + t_out  # (B, C, pred_len)
        out = out.permute(0, 2, 1)  # (B, pred_len, C)
        # 提取目标通道最后一步
        return out[:, -1, self.target_idx]  # (B,)


# ══════════════════════════════════════════════════════
#  3. 模型 2: PatchTST (ICLR 2023)
# ══════════════════════════════════════════════════════

class RevIN(nn.Module):
    """
    Reversible Instance Normalization.
    论文: "RevIN: Reversible Instance Normalization for Accurate Time-Series
           Forecasting against Distribution Shift" (ICLR 2022)
    参考: github.com/ts-kim/RevIN

    核心机制 (和 PatchTST/iTransformer 原版完全一致):
      mode='norm':
        mean_i = mean(x, dim=time)
        std_i  = std(x, dim=time) + eps
        x_norm = (x - mean_i) / std_i
        x_out  = x_norm * affine_weight + affine_bias     # 可学习的 affine

      mode='denorm':
        x_denorm = (x - affine_bias) / (affine_weight + eps)
        x_out    = x_denorm * std_i + mean_i

    关键点:
      1. mean 和 std 在 forward('norm') 时保存, 供 forward('denorm') 使用
      2. affine_weight 初始化为 1.0, affine_bias 初始化为 0.0
         → 等价于 "learnable standardization"
      3. 每个样本 x per-channel 独立计算统计量 (不是 batch 维度)
    """

    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            # 这两个参数是修复的关键所在
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias   = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode: str):
        """
        x: (B, L, C)
        mode: 'norm' or 'denorm'
        """
        if mode == "norm":
            self._compute_stats(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError(f"RevIN mode={mode}")
        return x

    def _compute_stats(self, x):
        """在 time 维度 (dim=1) 上计算每个样本每个通道的统计量"""
        dim = 1   # time axis
        self.mean = torch.mean(x, dim=dim, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim, keepdim=True, unbiased=False)
                                + self.eps).detach()

    def _normalize(self, x):
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev + self.mean
        return x


# ══════════════════════════════════════════════════════
#  修复版 PatchTST
# ══════════════════════════════════════════════════════

class PatchTST(nn.Module):
    """
    PatchTST (ICLR 2023) — 修复版.

    与之前版本的区别:
      旧: per-sample mean/std 归一化, 无 affine, 所有通道共享头
      新:
        - 使用 RevIN(n_features) 做 per-channel 的可学习归一化
        - Channel Independence: 每个通道独立过 backbone
        - Flatten head: 每个通道独立投影到 pred_len

    Backbone 结构:
      input (B, L, C)
      → RevIN norm
      → permute (B, C, L)
      → unfold patches: (B, C, n_patches, patch_len)
      → reshape (B*C, n_patches, patch_len)
      → patch embed: (B*C, n_patches, d_model)
      → + positional embedding
      → Transformer encoder
      → flatten: (B*C, n_patches * d_model)
      → head: (B*C, pred_len)
      → reshape (B, C, pred_len) → (B, pred_len, C)
      → RevIN denorm
      → 取 target 通道最后一步
    """
    name = "PatchTST"

    def __init__(self, seq_len=SEQ_LEN, pred_len=PRED_LEN, n_features=N_FEATURES,
                 target_idx=TARGET_IDX, patch_len=16, stride=8,
                 d_model=128, n_heads=4, n_layers=3, d_ff=256, dropout=0.2):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.target_idx = target_idx
        self.patch_len = patch_len
        self.stride = stride

        # ⭐ 关键修复: 使用标准 RevIN
        self.revin = RevIN(num_features=n_features, affine=True)

        # Patch 数计算: padding 后 unfold
        # 标准 PatchTST 用的是 padding = stride, 让最后一个 patch 覆盖到序列末
        self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
        self.n_patches = int((seq_len - patch_len) / stride + 2)  # +1 for padded

        # Patch embedding
        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer encoder (channel-independent, 所有通道共享权重)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation="gelu",
            norm_first=True   # Pre-LayerNorm, 原版 PatchTST 使用
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)

        # Flatten head
        self.head = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.n_patches * d_model, pred_len)
        )

    def forward(self, x):
        # x: (B, L, C)
        B, L, C = x.shape

        # ⭐ RevIN normalize
        x = self.revin(x, mode="norm")            # (B, L, C)

        # Channel Independence: (B, L, C) → (B, C, L)
        x = x.permute(0, 2, 1)

        # Padding + Patching: (B, C, L) → (B, C, L + stride) → (B, C, n_patches, patch_len)
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # x: (B, C, n_patches, patch_len)

        # Reshape 把通道维度并入 batch, 让所有通道共享 encoder
        n_patches = x.shape[2]
        x = x.reshape(B * C, n_patches, self.patch_len)

        # Patch embedding + positional
        x = self.patch_embed(x) + self.pos_embed[:, :n_patches, :]
        x = self.dropout(x)

        # Transformer encoder
        x = self.encoder(x)                       # (B*C, n_patches, d_model)

        # Head → pred_len
        x = self.head(x)                          # (B*C, pred_len)

        # Back to (B, C, pred_len) → (B, pred_len, C)
        x = x.reshape(B, C, self.pred_len).permute(0, 2, 1)

        # ⭐ RevIN denormalize
        x = self.revin(x, mode="denorm")          # (B, pred_len, C)

        # 取 target 通道最后一步
        return x[:, -1, self.target_idx]          # (B,)


# ══════════════════════════════════════════════════════
#  修复版 iTransformer
# ══════════════════════════════════════════════════════

class iTransformer(nn.Module):
    """
    iTransformer (ICLR 2024) — 修复版.

    与之前版本的区别:
      旧: 手动 mean/std 归一化后直接 embed, 没有 affine, 反归一化简单乘回
      新:
        - 使用 RevIN(n_features) 做 per-channel 可学习归一化
        - 倒置: 每个变量的整条归一化后时序 → 一个 d_model token
        - Self-attention 作用于变量 token
        - Projection 回 pred_len, RevIN denorm

    核心洞见 (原论文):
      Transformer 用于 time tokens 效果不如用于 variate tokens
      因为不同时间步在物理上往往无关, 但不同变量间有稳定的相互作用模式
    """
    name = "iTransformer"

    def __init__(self, seq_len=SEQ_LEN, pred_len=PRED_LEN, n_features=N_FEATURES,
                 target_idx=TARGET_IDX, d_model=128, n_heads=4, n_layers=3,
                 d_ff=256, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.target_idx = target_idx

        # ⭐ 关键修复: 使用标准 RevIN
        self.revin = RevIN(num_features=n_features, affine=True)

        # Variate-level embedding: 整条归一化后时序 → d_model token
        self.embed = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        # Encoder over variate tokens
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation="gelu",
            norm_first=True  # 原版 iTransformer 用 Pre-LN
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        # Projection: d_model → pred_len
        self.projection = nn.Linear(d_model, pred_len)

    def forward(self, x):
        # x: (B, L, C)

        # ⭐ RevIN normalize (per-sample per-channel affine)
        x = self.revin(x, mode="norm")       # (B, L, C)

        # 倒置: (B, L, C) → (B, C, L), 每个变量当作一个 token
        x = x.permute(0, 2, 1)               # (B, C, L)
        x = self.dropout(self.embed(x))      # (B, C, d_model)

        # Self-attention on variate tokens
        x = self.encoder(x)                  # (B, C, d_model)
        x = self.norm(x)

        # Projection to pred_len
        x = self.projection(x)               # (B, C, pred_len)
        x = x.permute(0, 2, 1)               # (B, pred_len, C)

        # ⭐ RevIN denormalize
        x = self.revin(x, mode="denorm")

        # 取 target 通道最后一步
        return x[:, -1, self.target_idx]     # (B,)

class PastDecomposableMixing(nn.Module):
    """TimeMixer 的核心: 分解 + 多尺度时间混合"""
    def __init__(self, seq_lens, d_model, d_ff, dropout):
        super().__init__()
        self.n_scales = len(seq_lens)
        self.decomp = SeriesDecomposition(kernel_size=25)
        # 季节性混合 (bottom-up, 从短尺度到长尺度)
        self.season_mixing = nn.ModuleList([
            nn.Sequential(
                nn.Linear(seq_lens[i], seq_lens[i+1]),
                nn.GELU(),
                nn.Linear(seq_lens[i+1], seq_lens[i+1]),
            ) for i in range(self.n_scales - 1)
        ])
        # 趋势混合 (top-down, 从长尺度到短尺度)
        self.trend_mixing = nn.ModuleList([
            nn.Sequential(
                nn.Linear(seq_lens[i+1], seq_lens[i]),
                nn.GELU(),
                nn.Linear(seq_lens[i], seq_lens[i]),
            ) for i in range(self.n_scales - 1)
        ])
        self.norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(self.n_scales)])

    def forward(self, x_list):
        # x_list: List[(B, L_i, D)], 每个尺度一个
        season_list, trend_list = [], []
        for x in x_list:
            s, t = self.decomp(x)
            season_list.append(s); trend_list.append(t)
        # Bottom-up 季节混合
        out_season = [season_list[0]]
        for i in range(self.n_scales - 1):
            mixed = self.season_mixing[i](out_season[i].permute(0, 2, 1)).permute(0, 2, 1)
            out_season.append(mixed + season_list[i+1])
        # Top-down 趋势混合
        out_trend = [None] * self.n_scales
        out_trend[-1] = trend_list[-1]
        for i in range(self.n_scales - 2, -1, -1):
            mixed = self.trend_mixing[i](out_trend[i+1].permute(0, 2, 1)).permute(0, 2, 1)
            out_trend[i] = mixed + trend_list[i]
        # 合并并归一化
        return [self.norm[i](out_season[i] + out_trend[i]) for i in range(self.n_scales)]


class TimeMixer(nn.Module):
    """
    TimeMixer: "Decomposable Multiscale Mixing for Time Series Forecasting" (ICLR 2024)
    核心: 多尺度下采样 + 分解 + 跨尺度混合
    """
    name = "TimeMixer"

    def __init__(self, seq_len=SEQ_LEN, pred_len=PRED_LEN, n_features=N_FEATURES,
                 target_idx=TARGET_IDX, d_model=128, d_ff=256, e_layers=2,
                 down_sampling_layers=2, down_sampling_window=2, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.target_idx = target_idx
        self.down_sampling_window = down_sampling_window
        self.down_sampling_layers = down_sampling_layers
        # 计算各尺度的序列长度
        self.seq_lens = [seq_len]
        for _ in range(down_sampling_layers):
            self.seq_lens.append(self.seq_lens[-1] // down_sampling_window)
        self.n_scales = len(self.seq_lens)
        # Embedding
        self.embed = nn.Linear(n_features, d_model)
        self.dropout = nn.Dropout(dropout)
        # 下采样
        self.down_pool = nn.AvgPool1d(kernel_size=down_sampling_window, stride=down_sampling_window)
        # PDM blocks
        self.pdm_blocks = nn.ModuleList([
            PastDecomposableMixing(self.seq_lens, d_model, d_ff, dropout)
            for _ in range(e_layers)
        ])
        # 预测头 (每个尺度一个, 加权融合)
        self.predict_heads = nn.ModuleList([
            nn.Linear(self.seq_lens[i], pred_len) for i in range(self.n_scales)
        ])
        self.out_proj = nn.Linear(d_model, n_features)

    def forward(self, x):
        # x: (B, L, C)
        # Per-instance 归一化
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-5
        x = (x - mean) / std
        # Embed
        x_e = self.dropout(self.embed(x))            # (B, L, d_model)
        # 构造多尺度: [L, L/2, L/4, ...]
        x_list = [x_e]
        cur = x_e
        for _ in range(self.down_sampling_layers):
            cur = self.down_pool(cur.permute(0, 2, 1)).permute(0, 2, 1)
            x_list.append(cur)
        # PDM blocks
        for block in self.pdm_blocks:
            x_list = block(x_list)
        # 每个尺度生成预测, 加权求和
        preds = []
        for i, (feat, head) in enumerate(zip(x_list, self.predict_heads)):
            # feat: (B, L_i, d_model) → (B, pred_len, n_features)
            p = head(feat.permute(0, 2, 1)).permute(0, 2, 1)   # (B, pred_len, d_model)
            p = self.out_proj(p)                                # (B, pred_len, n_features)
            preds.append(p)
        out = torch.stack(preds, dim=0).mean(dim=0)             # (B, pred_len, n_features)
        # 反归一化
        out = out * std + mean
        return out[:, -1, self.target_idx]                      # (B,)


# ══════════════════════════════════════════════════════
#  6. 模型 5: Autoformer (NeurIPS 2021)
# ══════════════════════════════════════════════════════

class AutoCorrelation(nn.Module):
    """
    Autoformer 的核心 Auto-Correlation: 基于 FFT 做周期发现 + 时间延迟聚合
    """
    def __init__(self, factor=3, attention_dropout=0.1):
        super().__init__()
        self.factor = factor
        self.dropout = nn.Dropout(attention_dropout)

    def time_delay_agg(self, values, corr):
        # values: (B, H, C, L), corr: (B, H, C, L)
        B, H, C, L = values.shape
        # 选 top-k 相关位置
        top_k = int(self.factor * math.log(L))
        top_k = max(1, min(L, top_k))
        mean_corr = corr.mean(dim=(1, 2))            # (B, L)
        _, index = torch.topk(mean_corr, top_k, dim=-1)    # (B, top_k)
        weights = F.softmax(mean_corr.gather(-1, index), dim=-1)  # (B, top_k)
        t = torch.arange(L, device=values.device, dtype=index.dtype)  # (L,)
        gather_idx = (t.view(1, 1, 1, 1, L) + index.view(B, top_k, 1, 1, 1)) % L  # (B, top_k, 1, 1, L)
        gather_idx = gather_idx.expand(B, top_k, H, C, L)  # (B, top_k, H, C, L)
        shifted = torch.gather(values.unsqueeze(1).expand(B, top_k, H, C, L), dim=-1, index=gather_idx)
        out = (shifted * weights.view(B, top_k, 1, 1, 1)).sum(dim=1)
        return out

    def forward(self, q, k, v):
        # q/k/v: (B, L, H, E)
        B, L, H, E = q.shape
        # FFT-based correlation
        q_f = torch.fft.rfft(q.permute(0, 2, 3, 1), dim=-1)
        k_f = torch.fft.rfft(k.permute(0, 2, 3, 1), dim=-1)
        corr_f = q_f * torch.conj(k_f)
        corr = torch.fft.irfft(corr_f, n=L, dim=-1)      # (B, H, E, L)
        # Time delay aggregation
        v_reshape = v.permute(0, 2, 3, 1)                 # (B, H, E, L)
        out = self.time_delay_agg(v_reshape, corr)        # (B, H, E, L)
        out = out.permute(0, 3, 1, 2)                     # (B, L, H, E)
        return self.dropout(out)


class AutoformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, factor=3):
        super().__init__()
        self.d_head = d_model // n_heads
        self.n_heads = n_heads
        self.d_model = d_model
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.auto_corr = AutoCorrelation(factor=factor, attention_dropout=dropout)
        self.decomp1 = SeriesDecomposition(kernel_size=25)
        self.decomp2 = SeriesDecomposition(kernel_size=25)
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, D = x.shape
        q = self.q_proj(x).reshape(B, L, self.n_heads, self.d_head)
        k = self.k_proj(x).reshape(B, L, self.n_heads, self.d_head)
        v = self.v_proj(x).reshape(B, L, self.n_heads, self.d_head)
        attn = self.auto_corr(q, k, v).reshape(B, L, D)
        x = x + self.dropout(self.out_proj(attn))
        x, _ = self.decomp1(x)     # 只取季节分量
        # FFN
        y = self.conv2(F.gelu(self.conv1(x.permute(0, 2, 1)))).permute(0, 2, 1)
        x = x + self.dropout(y)
        x, _ = self.decomp2(x)
        return x


class Autoformer(nn.Module):
    """
    Autoformer (NeurIPS 2021): 分解 + Auto-Correlation
    (简化版: 仅 encoder, 单步预测场景够用)
    """
    name = "Autoformer"

    def __init__(self, seq_len=SEQ_LEN, pred_len=PRED_LEN, n_features=N_FEATURES,
                 target_idx=TARGET_IDX, d_model=128, n_heads=8, e_layers=2,
                 d_ff=256, dropout=0.1, factor=3):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.target_idx = target_idx
        self.embed = nn.Linear(n_features, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.encoder_layers = nn.ModuleList([
            AutoformerEncoderLayer(d_model, n_heads, d_ff, dropout, factor)
            for _ in range(e_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)                # 只预测 POWER

    def forward(self, x):
        # x: (B, L, C)
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-5
        x = (x - mean) / std
        x = self.embed(x) + self.pos_embed               # (B, L, d_model)
        for layer in self.encoder_layers:
            x = layer(x)
        x = self.norm(x)
        out = self.head(x[:, -1, :])                     # (B, 1) 最后一步 → POWER
        # 反归一化 (只对 POWER)
        power_mean = mean[:, 0, self.target_idx:self.target_idx+1]    # (B, 1)
        power_std  = std[:, 0, self.target_idx:self.target_idx+1]
        out = out * power_std + power_mean
        return out.squeeze(-1)                           # (B,)


# ══════════════════════════════════════════════════════
#  7. 模型注册
# ══════════════════════════════════════════════════════

MODEL_REGISTRY = {
    "DLinear":      DLinear,
    "PatchTST":     PatchTST,
    "iTransformer": iTransformer,
    "TimeMixer":    TimeMixer,
    "Autoformer":   Autoformer,
}


# ══════════════════════════════════════════════════════
#  8. 训练
# ══════════════════════════════════════════════════════

def train_model(model, X_tr, y_tr, X_vl, y_vl, device, save_dir):
    torch.manual_seed(SEED); np.random.seed(SEED)
    X_tr = torch.FloatTensor(X_tr).to(device); y_tr = torch.FloatTensor(y_tr).to(device)
    X_vl = torch.FloatTensor(X_vl).to(device); y_vl = torch.FloatTensor(y_vl).to(device)

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_vl, y_vl), batch_size=512, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.2, patience=LR_PATIENCE)
    criterion = nn.MSELoss()

    best_val, best_state, patience = float("inf"), None, 0
    train_hist, val_hist = [], []
    t0 = time.time()

    print(f"\n{'─'*55}\n 训练 {model.name} (samples: {len(X_tr)}, val: {len(X_vl)})")
    print(f"{'─'*55}")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
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
            best_val = vl_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= EARLY_STOP:
            print(f"  早停于 Epoch {epoch}, 最佳 Val: {best_val:.6f}, 耗时: {time.time()-t0:.1f}s")
            break
        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}/{NUM_EPOCHS} | Train: {tr_loss:.5f} | Val: {vl_loss:.5f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.5f} | 耗时: {time.time()-t0:.1f}s")

    model.load_state_dict(best_state)
    model.eval()
    return train_hist, val_hist


def compute_metrics(y_true, y_pred):
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_r = np.sum((y_true - y_pred) ** 2)
    ss_t = np.sum((y_true - y_true.mean()) ** 2)
    return {"MAE": mae, "RMSE": rmse, "R2": float(1 - ss_r / (ss_t + 1e-10))}


# ══════════════════════════════════════════════════════
#  9. 单模型实验流程
# ══════════════════════════════════════════════════════

def run_single_model(model_name, df, device):
    """训练 + 预测单个模型, 返回指标"""
    result_dir = os.path.join(RESULT_ROOT, model_name)
    os.makedirs(result_dir, exist_ok=True)

    # 重定向日志
    old_stdout = sys.stdout
    logger = Logger(os.path.join(result_dir, "training_log.txt"))
    sys.stdout = logger

    try:
        print(f"[系统] 设备: {device}, 模型: {model_name}")
        # 数据划分 (60/20/20 三段)
        n_total = len(df); n_train = int(n_total * TRAIN_RATIO)
        n_rl_tr = int(n_total * RL_TRAIN_RATIO)
        rl_tr_start, rl_tr_end = n_train, n_train + n_rl_tr
        rl_te_start, rl_te_end = rl_tr_end, n_total
        feat_scaler, power_scaler = MinMaxScaler(), MinMaxScaler()
        feat_tr  = feat_scaler.fit_transform(df.iloc[:n_train][FEATURE_COLS].values)
        power_tr = power_scaler.fit_transform(df.iloc[:n_train][[TARGET_COL]].values).ravel()
        feat_rl_tr  = feat_scaler.transform(df.iloc[rl_tr_start:rl_tr_end][FEATURE_COLS].values)
        power_rl_tr = power_scaler.transform(df.iloc[rl_tr_start:rl_tr_end][[TARGET_COL]].values).ravel()
        feat_te  = feat_scaler.transform(df.iloc[rl_te_start:rl_te_end][FEATURE_COLS].values)
        power_te = power_scaler.transform(df.iloc[rl_te_start:rl_te_end][[TARGET_COL]].values).ravel()
        print(f"[划分] predictor={n_train}, rl_train={n_rl_tr}, rl_test={rl_te_end-rl_te_start}")

        # 滑窗
        X_tr, y_tr = build_windows(feat_tr, power_tr)
        X_te, y_te = build_windows(feat_te, power_te)
        X_rl_tr, y_rl_tr = build_windows(feat_rl_tr, power_rl_tr)
        n_val = max(100, int(len(X_tr) * 0.1))
        X_fit, y_fit = X_tr[:-n_val], y_tr[:-n_val]
        X_val, y_val = X_tr[-n_val:], y_tr[-n_val:]
        print(f"[滑窗] predictor_train={len(X_fit)}, val={len(X_val)}, "
              f"rl_train={len(X_rl_tr)}, rl_test={len(X_te)}")
        print(f"[通道] {N_FEATURES} 维 [POWER, WS, WD, TEMP, HUM, PRESSURE]")

        # 构建模型
        torch.manual_seed(SEED); np.random.seed(SEED)
        model_cls = MODEL_REGISTRY[model_name]
        model = model_cls().to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[模型] {model_name} 参数量: {n_params:,}")

        # 训练
        train_hist, val_hist = train_model(model, X_fit, y_fit, X_val, y_val, device, result_dir)

        # 推理 (两段)
        model.eval()
        def _infer(X_input):
            with torch.no_grad():
                loader = DataLoader(TensorDataset(torch.FloatTensor(X_input).to(device)),
                                     batch_size=512, shuffle=False)
                return np.concatenate([model(Xb).cpu().numpy() for (Xb,) in loader])
        preds_norm        = _infer(X_te)
        preds_norm_rl_tr  = _infer(X_rl_tr)

        # 反归一化
        y_pred_kw       = np.clip(power_scaler.inverse_transform(preds_norm.reshape(-1, 1)).ravel(), 0, None)
        y_pred_kw_rl_tr = np.clip(power_scaler.inverse_transform(preds_norm_rl_tr.reshape(-1, 1)).ravel(), 0, None)
        y_true_kw       = power_scaler.inverse_transform(y_te.reshape(-1, 1)).ravel()
        y_true_kw_rl_tr = power_scaler.inverse_transform(y_rl_tr.reshape(-1, 1)).ravel()

        # 指标 (基于 RL test 段)
        m = compute_metrics(y_true_kw, y_pred_kw)
        print(f"\nMAE={m['MAE']:.1f} kW, RMSE={m['RMSE']:.1f} kW, R²={m['R2']:.4f}")

        # 保存预测 (兼容旧名 + 显式两段)
        name_lower = model_name.lower()
        pd.DataFrame({"true_kW": y_true_kw, "pred_kW": y_pred_kw}).to_csv(
            os.path.join(result_dir, f"{name_lower}_predictions.csv"), index=False)
        pd.DataFrame({"true_kW": y_true_kw, "pred_kW": y_pred_kw}).to_csv(
            os.path.join(result_dir, f"{name_lower}_predictions_rl_test.csv"), index=False)
        pd.DataFrame({"true_kW": y_true_kw_rl_tr, "pred_kW": y_pred_kw_rl_tr}).to_csv(
            os.path.join(result_dir, f"{name_lower}_predictions_rl_train.csv"), index=False)
        torch.save(model.state_dict(), os.path.join(result_dir, f"{model_name.lower()}_model.pt"))
        print(f"[输出] 预测 & 模型权重保存到 {result_dir}/")

        m["Model"] = model_name
        m["Params"] = n_params
        m["n_test"] = len(y_true_kw)
        return m

    finally:
        sys.stdout = old_stdout
        logger.close()


# ══════════════════════════════════════════════════════
#  10. 汇总对比 (和 CG-Mamba 一起)
# ══════════════════════════════════════════════════════

def generate_comparison(results, your_models_csv=None):
    """生成 CSV 和 Markdown 对比表"""
    rows = []
    for r in results:
        rows.append({
            "Model":  r["Model"],
            "Source": "TSL",
            "Params": r.get("Params", 0),
            "MAE":    r["MAE"],
            "RMSE":   r["RMSE"],
            "R2":     r["R2"],
        })
    # 加入你的模型 (如果提供了 CSV 路径)
    if your_models_csv:
        for name, path in your_models_csv.items():
            if os.path.exists(path):
                df = pd.read_csv(path)
                true_col = next((c for c in df.columns if "true" in c.lower()), None)
                pred_col = next((c for c in ["pred_median_kW", "pred_kW"] if c in df.columns), None)
                if true_col and pred_col:
                    t, p = df[true_col].values, df[pred_col].values
                    m = compute_metrics(t, p)
                    rows.append({
                        "Model": name, "Source": "Ours", "Params": 0,
                        "MAE": m["MAE"], "RMSE": m["RMSE"], "R2": m["R2"],
                    })
                    print(f"  + 加载 {name}: MAE={m['MAE']:.1f}")

    df_cmp = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    df_cmp.insert(0, "Rank", range(1, len(df_cmp)+1))

    # CSV
    csv_path = os.path.join(RESULT_ROOT, "comparison_table.csv")
    df_cmp.to_csv(csv_path, index=False)

    # Markdown
    md_path = os.path.join(RESULT_ROOT, "comparison_table.md")
    lines = ["# Wind Power Forecasting: Baseline Comparison",
             "",
             "| Rank | Model | Source | Params | MAE (kW) | RMSE (kW) | R² |",
             "|:----:|:------|:------:|-------:|---------:|----------:|---:|"]
    for _, r in df_cmp.iterrows():
        mark = " **🥇**" if r["Rank"] == 1 else ""
        params_str = f"{int(r['Params']):,}" if r["Params"] > 0 else "—"
        lines.append(f"| {r['Rank']} | {r['Model']}{mark} | {r['Source']} | "
                     f"{params_str} | {r['MAE']:.1f} | {r['RMSE']:.1f} | {r['R2']:.4f} |")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # 终端输出
    print("\n" + "=" * 75)
    print(" 最终对比表 (按 MAE 排序)")
    print("=" * 75)
    print(f"{'Rank':>4} {'Model':<18} {'Source':<8} {'Params':>10} "
          f"{'MAE (kW)':>10} {'RMSE (kW)':>10} {'R²':>8}")
    print("-" * 75)
    for _, r in df_cmp.iterrows():
        p = f"{int(r['Params']):,}" if r["Params"] > 0 else "—"
        print(f"{int(r['Rank']):>4} {r['Model']:<18} {r['Source']:<8} {p:>10} "
              f"{r['MAE']:>10.1f} {r['RMSE']:>10.1f} {r['R2']:>8.4f}")
    print(f"\n  CSV → {csv_path}")
    print(f"  Markdown → {md_path}")


# ══════════════════════════════════════════════════════
#  11. 主入口
# ══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="TSL Baselines Standalone (与 CG-Mamba 对齐协议)")
    parser.add_argument("--data", default=DATA_PATH)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--model", default="all",
                        choices=["all"] + list(MODEL_REGISTRY.keys()),
                        help="指定单个模型或 all")
    parser.add_argument("--mamba-csv", default=None,
                        help="Mamba 预测 CSV 路径 (加入对比表)")
    parser.add_argument("--cg-mamba-csv", default=None,
                        help="CG-Mamba 预测 CSV 路径")
    parser.add_argument("--cg-prob-csv", default=None,
                        help="CGProb-Mamba 预测 CSV 路径")
    args = parser.parse_args()

    os.makedirs(RESULT_ROOT, exist_ok=True)

    # 数据
    if args.demo or not os.path.exists(args.data):
        print("[提示] 演示模式")
        data_path = generate_demo_data()
    else:
        data_path = args.data
    df = load_data(data_path)

    # 设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[系统] 设备: {device}")

    # 运行模型
    models_to_run = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]
    print(f"[计划] 将运行 {len(models_to_run)} 个模型: {models_to_run}")

    all_results = []
    for i, name in enumerate(models_to_run, 1):
        print(f"\n{'='*75}")
        print(f" [{i}/{len(models_to_run)}] 开始训练: {name}")
        print(f"{'='*75}")
        t0 = time.time()
        try:
            m = run_single_model(name, df, device)
            all_results.append(m)
            print(f" ✅ {name} 完成, 耗时 {(time.time()-t0)/60:.1f} 分钟")
            print(f"    MAE={m['MAE']:.1f}, RMSE={m['RMSE']:.1f}, R²={m['R2']:.4f}")
        except Exception as e:
            print(f" ❌ {name} 失败: {e}")
            import traceback; traceback.print_exc()

    # 对比
    if all_results:
        your_csv = {
            "Mamba (ours)":        args.mamba_csv,
            "CG-Mamba (ours)":     args.cg_mamba_csv,
            "CGProb-Mamba (ours)": args.cg_prob_csv,
        }
        your_csv = {k: v for k, v in your_csv.items() if v}
        generate_comparison(all_results, your_csv)


if __name__ == "__main__":
    main()
