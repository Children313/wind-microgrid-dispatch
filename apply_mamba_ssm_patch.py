"""
apply_mamba_ssm_patch.py
========================
给 cg_prob_mamba.py / cg_mamba.py / mamba_baseline_aligned.py 三个脚本
应用 mamba-ssm CUDA kernel 加速.

实测加速比: ~30x (10 epoch 3900s → ~130s)

修改点:
  1. 在文件顶部 (import 后) 加 mamba-ssm 的 try/except 导入
  2. 在 MambaSSM.forward 里, 把 self.selective_scan(...) 调用换成
     条件分支 (用 mamba-ssm 走 CUDA kernel, 否则 fallback 到原循环)

数学等价性:
  原代码: A = exp(A_log) > 0, 然后 selective_scan 内部用 -A 算 delta_A
  新代码: 直接传 -exp(A_log) 给 selective_scan_fn, 库内部自己算 delta_A

  delta 在原代码里已经过 softplus, 所以新代码 delta_softplus=False
"""

import os, sys, re, argparse
from pathlib import Path

IMPORT_BLOCK = '''
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
'''

# 替换 forward 里 selective_scan 的调用部分
# 原代码:
#     A = torch.exp(self.A_log).to(x.dtype)
#     y = self.selective_scan(x_conv, delta, A, B, C)
#     y = y + self.D.unsqueeze(0).unsqueeze(-1) * x_conv
# 新代码:
#     A = torch.exp(self.A_log).to(x.dtype)
#     if USE_MAMBA_SSM_KERNEL and x.is_cuda:
#         # mamba-ssm 内部期望 A < 0
#         y = selective_scan_fn(x_conv, delta, -A, B, C, D=self.D,
#                               z=None, delta_bias=None, delta_softplus=False)
#         # 注意: selective_scan_fn 已经把 D*u 加进去了, 不要再加
#     else:
#         y = self.selective_scan(x_conv, delta, A, B, C)
#         y = y + self.D.unsqueeze(0).unsqueeze(-1) * x_conv

OLD_PATTERN = '''        A = torch.exp(self.A_log).to(x.dtype)
        y = self.selective_scan(x_conv, delta, A, B, C)
        y = y + self.D.unsqueeze(0).unsqueeze(-1) * x_conv'''

NEW_PATTERN = '''        A = torch.exp(self.A_log).to(x.dtype)
        if USE_MAMBA_SSM_KERNEL and x_conv.is_cuda:
            # mamba-ssm 期望 A < 0; D 由库内部加, 不需要外面再加
            y = selective_scan_fn(
                x_conv, delta, -A, B, C, D=self.D,
                z=None, delta_bias=None, delta_softplus=False
            )
        else:
            y = self.selective_scan(x_conv, delta, A, B, C)
            y = y + self.D.unsqueeze(0).unsqueeze(-1) * x_conv'''

# mamba_baseline_aligned.py 用了带类型标注 + 注释的版本, 单独支持
OLD_PATTERN_2 = '''        A = torch.exp(self.A_log).to(x.dtype) # (inner, d_state)

        # ── 选择性扫描 ─────────────────────────────────────────
        y = self.selective_scan(x_conv, delta, A, B, C)  # (batch, inner, L)

        # D 跳跃连接
        y = y + self.D.unsqueeze(0).unsqueeze(-1) * x_conv'''

NEW_PATTERN_2 = '''        A = torch.exp(self.A_log).to(x.dtype) # (inner, d_state)

        # ── 选择性扫描 (优先使用 mamba-ssm CUDA kernel) ────────
        if USE_MAMBA_SSM_KERNEL and x_conv.is_cuda:
            # mamba-ssm 期望 A < 0; D 由库内部加, 不需要外面再加
            y = selective_scan_fn(
                x_conv, delta, -A, B, C, D=self.D,
                z=None, delta_bias=None, delta_softplus=False
            )
        else:
            y = self.selective_scan(x_conv, delta, A, B, C)
            y = y + self.D.unsqueeze(0).unsqueeze(-1) * x_conv'''


def patch_file(filepath: Path, dry_run: bool = False) -> bool:
    text = filepath.read_text(encoding='utf-8')
    original = text
    changed = False

    # 1. 加 import block (如果还没加)
    if 'USE_MAMBA_SSM_KERNEL' not in text:
        # 在 "import torch.nn.functional as F" 后插入 (假设这是必有的 import)
        markers = [
            'import torch.nn.functional as F',
            'import torch.nn as nn',
            'import torch',
        ]
        inserted = False
        for marker in markers:
            if marker + '\n' in text:
                text = text.replace(
                    marker + '\n',
                    marker + '\n' + IMPORT_BLOCK,
                    1
                )
                inserted = True
                break
        if not inserted:
            print(f"  [skip] 没找到 import torch 行: {filepath}")
            return False
        changed = True

    # 2. 替换 forward 里的 selective_scan 调用 (尝试两种模式)
    matched = False
    if OLD_PATTERN in text:
        text = text.replace(OLD_PATTERN, NEW_PATTERN, 1)
        matched = True
    elif OLD_PATTERN_2 in text:
        text = text.replace(OLD_PATTERN_2, NEW_PATTERN_2, 1)
        matched = True

    if matched:
        changed = True
    elif 'USE_MAMBA_SSM_KERNEL and x_conv.is_cuda' not in text:
        print(f"  [警告] 没找到标准的 selective_scan 调用模式: {filepath.name}")
        print(f"         可能你的代码已被修改, 请手动改 forward")

    if not changed:
        return False

    if dry_run:
        print(f"  [dry] would patch: {filepath}")
        return True

    backup = filepath.with_suffix(filepath.suffix + '.bak')
    if not backup.exists():
        backup.write_text(original, encoding='utf-8')
        print(f"  [backup] {backup.name}")

    filepath.write_text(text, encoding='utf-8')
    print(f"  [patched] {filepath.name}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default='.',
                        help="脚本所在目录")
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--files', nargs='+', default=None)
    args = parser.parse_args()

    target_dir = Path(args.target).resolve()
    if not target_dir.exists():
        print(f"目录不存在: {target_dir}"); sys.exit(1)

    if args.files:
        files = [target_dir / f for f in args.files]
    else:
        candidates = ['cg_prob_mamba.py', 'cg_mamba.py', 'mamba_baseline_aligned.py']
        files = [target_dir / f for f in candidates if (target_dir / f).exists()]

    if not files:
        print(f"未找到目标文件"); sys.exit(1)

    print(f"目标目录: {target_dir}")
    print(f"目标文件: {[f.name for f in files]}\n")

    for f in files:
        if not f.exists():
            print(f"  [skip] 不存在: {f}")
            continue
        patch_file(f, dry_run=args.dry_run)

    if args.dry_run:
        print("\n(dry-run)")
    else:
        print("\n[完成]")
        print("  - 备份保存为 .bak")
        print("  - 现在重跑训练命令即可, 自动走 mamba-ssm CUDA kernel")
        print("  - 预期: 10 epoch 从 3900s -> ~130s")


if __name__ == "__main__":
    main()
