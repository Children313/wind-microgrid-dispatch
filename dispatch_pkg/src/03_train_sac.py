"""
03_train_sac.py
===============
6 组对照 SAC 训练 (你的 GPU 自动检测):

  1. baseline           无预测信号           obs=19
  2. point_dlinear      DLinear 点预测       obs=20
  3. point_patchtst     PatchTST 点预测      obs=20
  4. point_cgmamba      CG-Mamba 点预测      obs=20
  5. prob_cgprob        CG-Prob-Mamba 概率   obs=22  ⭐主推
  6. oracle             作弊上界 (true t+1)   obs=22

数据使用:
  - 训练时读 data/rl_train.csv + 对应预测 CSV 的 train 段
  - 评估时读 data/rl_eval.csv + 对应预测 CSV 的 eval 段
  - 训练段和评估段在时间上严格不重叠 (无数据泄露)

使用:
  # 单组训练
  python 03_train_sac.py --agent prob_cgprob --steps 50000 --seed 42

  # 全部 6 组 (需要约 50000步x6组 时间)
  python 03_train_sac.py --agent all --steps 50000 --seed 42

输出:
  results/<agent>_seed<S>/
    train_log.csv     训练曲线
    model.zip         模型权重
"""

import warnings; warnings.filterwarnings('ignore')
import os, sys, time, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.resolve()))
from importlib import import_module
# 优先使用 v2 (论文级环境), 找不到则 fallback 到 v1
try:
    mge = import_module("02_microgrid_env_v2")
    print("[env] 使用 02_microgrid_env_v2 (论文级, 复现冯文韬 2023)")
except ImportError:
    mge = import_module("02_microgrid_env")
    print("[env] 使用 02_microgrid_env v1 (简化版)")

from stable_baselines3 import SAC, TD3, DDPG
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise

ALGOS = {"SAC": SAC, "TD3": TD3, "DDPG": DDPG}


# ====================================================================
#  Reward Logger
# ====================================================================
class RewardLogger(BaseCallback):
    def __init__(self, name, every=2500):
        super().__init__()
        self.name, self.every = name, every
        self.history, self.recent = [], []

    def _on_step(self):
        if 'rewards' in self.locals:
            r = self.locals['rewards']
            r_val = float(r[0]) if hasattr(r, '__len__') else float(r)
            self.recent.append(r_val)
        if self.num_timesteps % self.every == 0 and self.recent:
            window = self.recent[-self.every:]
            mean_r = float(np.mean(window))
            self.history.append((self.num_timesteps, mean_r))
            print(f"  [{self.name:18s}] step {self.num_timesteps:>7d}  "
                  f"reward_avg(last {len(window)}): {mean_r:>12,.1f}")
        return True


# ====================================================================
#  通用 Trainer
# ====================================================================
def make_model(algo_name, env, seed, device, total_steps):
    Algo = ALGOS[algo_name]
    common = dict(
        learning_rate=3e-4,
        buffer_size=min(total_steps, 100_000),
        batch_size=256,
        tau=0.005, gamma=0.99,
        learning_starts=1000,
        train_freq=1, gradient_steps=1,
        policy_kwargs=dict(net_arch=[128, 128]),
        verbose=0, seed=seed, device=device,
    )
    if algo_name == "SAC":
        return Algo("MlpPolicy", env, ent_coef='auto', **common)
    elif algo_name == "TD3":
        n = env.action_space.shape[-1]
        common['action_noise'] = NormalActionNoise(np.zeros(n), 0.1*np.ones(n))
        return Algo("MlpPolicy", env, **common)
    else:
        n = env.action_space.shape[-1]
        common['action_noise'] = NormalActionNoise(np.zeros(n), 0.2*np.ones(n))
        return Algo("MlpPolicy", env, **common)


def train_one(agent, total_steps, algo, seed, device, out_root):
    print(f"\n{'='*60}\n  Training: {agent} ({algo}, {total_steps} steps, seed={seed})\n{'='*60}")
    out_dir = os.path.join(out_root, f"{agent}_seed{seed}")
    os.makedirs(out_dir, exist_ok=True)

    # 训练时用 train 段数据
    env_train = mge.make_env(agent, mode='train')
    print(f"  obs_space: {env_train.observation_space.shape}, "
          f"action_space: {env_train.action_space.shape}")

    torch.manual_seed(seed); np.random.seed(seed)
    model = make_model(algo, env_train, seed, device, total_steps)

    cb = RewardLogger(agent, every=max(2500, total_steps//20))
    t0 = time.time()
    model.learn(total_timesteps=total_steps, callback=cb, log_interval=100)
    elapsed = time.time() - t0
    print(f"  耗时: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # 保存
    if cb.history:
        pd.DataFrame(cb.history, columns=["step", "mean_reward"]).to_csv(
            os.path.join(out_dir, "train_log.csv"), index=False)
    model.save(os.path.join(out_dir, "model"))
    print(f"  Saved → {out_dir}/model.zip")
    return model


# ====================================================================
#  Main
# ====================================================================
def print_device_info(device):
    print(f"\n[device] requested: {device}")
    print(f"[device] torch.__version__: {torch.__version__}")
    print(f"[device] cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[device] cuda device: {torch.cuda.get_device_name(0)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, default='all',
                        help=f"指定单组或 'all'. 选项: {mge.SUPPORTED_AGENTS} / 'all'")
    parser.add_argument('--steps', type=int, default=50000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--algo', type=str, default='SAC', choices=list(ALGOS.keys()))
    parser.add_argument('--device', type=str, default='auto',
                        help="auto / cuda / cpu")
    parser.add_argument('--out_root', type=str, default='./results')
    parser.add_argument('--skip', type=str, default='',
                        help="逗号分隔, 例如 'baseline,oracle'")
    args = parser.parse_args()

    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print_device_info(args.device)

    # 默认推荐顺序: 先快的 (baseline, point), 再慢的 (prob, oracle)
    DEFAULT_ORDER = [
        'baseline',
        'point_dlinear', 'point_patchtst', 'point_cgmamba',
        'prob_cgprob',
        'oracle',
    ]
    if args.agent == 'all':
        agents = DEFAULT_ORDER
    else:
        agents = [args.agent]

    skip = set(s.strip() for s in args.skip.split(',') if s.strip())
    agents = [a for a in agents if a not in skip]
    print(f"\n[plan] 训练顺序: {agents}")
    print(f"       每组 {args.steps} steps × seed={args.seed}")
    print(f"       预估总时长: ~{args.steps/3000 * len(agents):.1f} 分钟 (在 RTX 4060 Laptop 上)")

    for agent in agents:
        train_one(agent, args.steps, args.algo, args.seed, args.device, args.out_root)

    print(f"\n{'='*60}\n训练完成. 评估 + 出对比表/图:")
    print(f"  python 04_evaluate.py --out_root {args.out_root} --seed {args.seed}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
