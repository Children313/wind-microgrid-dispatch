# 预测脚本 v2 改动说明

## 核心变更

把原本 `TRAIN_RATIO = 0.8`（80/20 二段切分）改成 **60/20/20 三段切分**：

```
[============== 预测模型训练 ==============][== RL train ==][== RL test ==]
←——————————— 60% (PREDICTOR_RATIO) ———————→←- 20% (RL_TRAIN_RATIO) -→←- 20% -→
                 ↑                              ↑                       ↑
   预测模型在这段拟合 (内含 10% val)         RL agent 训练时          RL agent 评估时
                                               用这段的预测           用这段的预测
```

**为什么这样切**：

- 原来 80% 训练 + 20% 测试，下游 RL 把这 20% 测试段同时用作 RL 训练和 RL 评估，**存在数据泄露**
- 现在 60% 给预测器训练（预测器永远看不到后 40%），后 40% 切两半给 RL，**RL 训练段和评估段在时间上严格不重叠**

## 5 个脚本的改动方式相同

| 脚本 | 角色 | 用法 |
|---|---|---|
| `cg_prob_mamba.py` | 你主推的概率预测模型 | RL 状态扩展用，输出含分位数+不确定性的 CSV |
| `cg_mamba.py` | CG-Mamba 点预测 | PointAware 对照组用 |
| `mamba_baseline_aligned.py` | vanilla Mamba 点预测 | PointAware-Mamba 对照组用 |
| `tsl_baselines.py` | 5 个 SOTA baseline 合并版（DLinear/PatchTST/iTransformer/TimeMixer/Autoformer）| 论文 baseline 对比表 |

**注**：`tsl_baselines.py` 是把原来的 `tsl_baselines_standalone.py` 和 `tsl_fixed_models.py` 合并成一个，
其中 PatchTST 和 iTransformer 用了带 RevIN 的修复版（避免之前预测塌缩 / 尺度错位的问题）。

每个脚本都做了同样的 4 件事：

1. **常量改三段**：新增 `RL_TRAIN_RATIO=0.2`，`TRAIN_RATIO` 从 0.8 → 0.6
2. **数据划分改三段**：scaler 仅在前 60% 训练段拟合（避免泄露），中间 20% 和后 20% 都只用 transform
3. **滑动窗口加 RL train 段**：新增 `X_rl_tr, y_rl_tr = build_windows(feat_rl_tr, power_rl_tr)`
4. **推理+保存改两段**：分别在 RL train 段和 RL test 段做预测，输出**两个 CSV** 给下游 RL：
   - `*_predictions_rl_train.csv` → RL agent 训练时用
   - `*_predictions_rl_test.csv` → RL agent 评估时用
   - `*_predictions.csv`（旧文件名兼容，等同 RL test 段）

## CG-Prob-Mamba 特殊处理：归一化基准统一

`uncertainty_norm` 列的归一化基准用 **RL train 段** 的最大区间宽度，
两段共享同一个 `iw_max_global` 值。这保证下游 RL agent 在 train 段学到的 obs 尺度
和 eval 段评估时看到的尺度一致，不会出现"训练时 unc 都是 0.5，评估时变成 0.9"的尺度漂移。

## 你跑预测的命令

```bash
python cg_prob_mamba.py --data /path/to/15_processed.csv
python cg_mamba.py --data /path/to/15_processed.csv
python mamba_baseline_aligned.py --data /path/to/15_processed.csv

# tsl_baselines.py 一键跑全部 5 个
python tsl_baselines.py --data /path/to/15_processed.csv --model all
# 或单独跑某一个
python tsl_baselines.py --data /path/to/15_processed.csv --model PatchTST
```

## 数据量预估 (15_processed.csv 共 43680 行)

| 段 | 比例 | 原始行数 | 滑窗后 (SEQ_LEN=96) |
|---|---|---|---|
| 预测训练 | 60% | 26208 | 26112 |
| RL 训练 | 20% | 8736 | 8640 |
| RL 测试 | 20% | 8736 | 8640 |

每段约 91 天，覆盖完整季节循环（春-夏-秋-冬-春-夏，三个季节梯度）。

## 下游 RL 调度脚本如何对接

下次你重写 `02_microgrid_env.py` 时，`make_envs(mode="train")` 读
`_rl_train.csv`，`make_envs(mode="eval")` 读 `_rl_test.csv`。这件事我可以
在你跑完 7 个预测模型、拿到所有 `_rl_train.csv` 和 `_rl_test.csv` 之后再帮你改。

## 跑完后该看什么

1. **预测精度比之前 16383 行版本应该提升** —— 训练数据从 13107 → 26112，CG-Prob-Mamba 的 R² 应该从 0.93 进一步提升
2. **关键诊断**：检查 `_rl_train.csv` 和 `_rl_test.csv` 上的 MAE 是否相近。如果 RL test 段精度比 RL train 段差很多（比如 MAE 差 50% 以上），说明季节漂移严重，需要在论文里讨论
3. **CG-Prob-Mamba 的 PICP@90 是否仍然贴近 0.9** —— 校准能否在新数据集上保持
