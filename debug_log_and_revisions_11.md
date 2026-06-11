# Debug Log & Revisions 11 — GPR Kernel Reconstruction & y-Randomization Gate

> **日期:** 2026-05-12
> **依据:** `revision_12.md` + `run_diagnosis_report_2.md`
> **范围:** GPR 致命崩溃修复 + y-Randomization 暂时关闭

---

## 修改总览

| 任务 | 描述 | 状态 |
|---|---|---|
| 任务一 | GPR kernel 参数泄漏崩溃修复 + `n_restarts_optimizer` | DONE |
| 任务二 | 注释掉 y-Randomization（可手动恢复） | DONE |

---

## 任务一：GPR Kernel 重建 — 致命崩溃修复

### 问题回顾

GPR 在第 1 轮迭代即崩溃：

```
TypeError: GaussianProcessRegressor.__init__() got an unexpected keyword argument 'length_scale'
```

**根因:** Optuna `objective()` 中为构建内核对象调用了 `trial.suggest_float("length_scale", ...)`。`study.best_params` 返回所有 trial-suggest 参数（包括 `kernel`、`length_scale`、`noise_level`）。这些参数被合并进 `final_params` 后传入 `GaussianProcessRegressor()`，但 GPR 不接受它们。

**方案 A 的陷阱:** 仅 `pop` 掉这些参数会导致 `kernel` 键也丢失，GPR 退化使用 `1.0 * RBF(1.0)` 默认内核 → Optuna 所有调优工作白费。

### 完整闭环修复

在 `src/train_and_evaluate.py` 的 Optuna 结果处理区，Lasso 分支之后增加 GPR 分支：

1. **剥离** `kernel`（字符串）、`length_scale`、`noise_level` 三个动态参数
2. **重建** `kernel` 对象（`ConstantKernel * RBF/Matern + WhiteKernel`）
3. **注入** 重建后的 kernel 对象回 `best_params`

```python
elif model_class == GaussianProcessRegressor:
    from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
    best_params = dict(best_params)

    k_choice = best_params.pop("kernel", "RBF")
    l_scale = best_params.pop("length_scale", 1.0)
    n_level = best_params.pop("noise_level", 1e-5)

    if k_choice == "RBF":
        final_kernel = ConstantKernel(1.0) * RBF(length_scale=l_scale)
    elif k_choice == "Matern":
        final_kernel = ConstantKernel(1.0) * Matern(length_scale=l_scale)
    else:
        final_kernel = ConstantKernel(1.0) * RBF(length_scale=l_scale) + WhiteKernel(noise_level=n_level)

    best_params["kernel"] = final_kernel
```

### 配套优化

`src/fixed_params.py` 中 GPR 新增 `"n_restarts_optimizer": 5`，辅助 L-BFGS-B 求解器越过核函数边缘似然的局部极值点。

---

## 任务二：y-Randomization 暂时关闭

### 原因

y-Randomization 每个模型执行 100×5×5 = 2500 次模型拟合。对树集成模型极为耗时：
- RandomForest: ~17 分钟
- XGBoost: ~8 小时

在频繁的调试和验证阶段，这个开销不可接受。

### 实施

`src/iterative_optimization.py` 中的 y-Randomization 调用块被整体注释。所有代码保留原样，仅加了 `#` 前缀。

### 手动恢复

取消 `src/iterative_optimization.py` 第 ~570-590 行的注释即可。搜索 `y-Randomization (commented out` 定位。

---

## 修正后的代码使用指导

命令行无变化：

```bash
nohup uv run python main.py --n_trials 100 --mae_threshold 2.0 --min_features 3 --n_jobs -1 > train.log 2>&1 &
```

GPR 现在可以正常运行 12 轮迭代特征筛选。

### 启用 y-Randomization

编辑 `src/iterative_optimization.py`，找到：
```
# ── y-Randomization (commented out — uncomment to re-enable) ──
```
将下方 ~20 行取消注释即可。

---

## 遗留问题与建议

1. **GPR ConvergenceWarning**: `length_scale` 在搜索中频繁触达上下界。当前 `n_restarts_optimizer=5` 已提供初步缓解。如需进一步优化可扩大搜索范围或增加重启次数。

2. **GPR 运行时间**: GPR 拟合复杂度 O(N³)。N≈88 时每 fit ~0.1s，尚可接受。但若数据扩充到 200+，GPR 会成为显著瓶颈。

3. **y-Randomization 恢复策略**: 建议在最终确认所有模型参数和特征数后，单独跑一轮开启 y-Randomization 的正式运行作为论文佐证。调试阶段保持关闭。

---

## 变更文件清单

```
修改的文件:
  src/train_and_evaluate.py     (任务一: GPR kernel 重建逻辑 — 20 行新增)
  src/fixed_params.py           (任务一: GPR n_restarts_optimizer=5)
  src/iterative_optimization.py (任务二: y-Randomization 块注释掉)
```
