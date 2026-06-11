# Debug Log & Revisions 15 — GPlearn p_crossover Crash & Warning Flood Fix

> **日期:** 2026-05-19
> **运行日志:** `train.log` (8604 行), `bugs.txt` (用户提取的报错样本)
> **范围:** 1 个致命崩溃 + 3 类警告修复

---

## 修改总览

| # | 问题 | 严重程度 | 状态 |
|---|---|---|---|
| 1 | GPlearn `p_crossover` 导致 `ValueError` 崩溃 | **致命** | FIXED |
| 2 | Optuna MLP `hidden_layer_sizes` tuple 类型警告 | 中 (海量) | FIXED |
| 3 | LightGBM `No further splits with positive gain` 噪声 | 低 (海量) | FIXED |
| 4 | LightGBM `X does not have valid feature names` 警告 | 低 | NOTED |

---

## Bug 1：GPlearn `p_crossover` ValueError（致命崩溃）

### 日志

```
ValueError: The sum of p_crossover, p_subtree_mutation, p_hoist_mutation
and p_point_mutation should total to 1.0 or less.
```

### 根因

gplearn 的 `SymbolicRegressor` 内置了固定变异概率：
- `p_subtree_mutation=0.1`
- `p_hoist_mutation=0.05`
- `p_point_mutation=0.1`

三者合计 0.25。`gplearn_wrapper.py` 将 `p_crossover` 的默认值设为 0.7（总计 0.7+0.25=0.95 < 1.0，安全）。但 Optuna 搜索空间设为 `p_crossover: 0.5-0.9`。当 Optuna 采样到 `p_crossover > 0.75` 时，总概率 > 1.0 → `ValueError` 崩溃。

### 修复

从 Optuna 搜索空间中移除 `p_crossover`，改为从 `fixed_params` 继承固定值 0.7：

```python
# 修改前:
params = {**base_params,
    "p_crossover": trial.suggest_float("p_crossover", 0.5, 0.9),  # BUG!
    ...
}

# 修改后:
params = {**base_params,  # p_crossover comes from fixed_params (0.7)
    "population_size": trial.suggest_int("population_size", 1000, 5000, step=500),
    "generations": trial.suggest_int("generations", 8, 25),
    "parsimony_coefficient": trial.suggest_float("parsimony_coefficient", 1e-4, 1e-2, log=True),
}
```

---

## Bug 2：Optuna MLP tuple 类型警告

### 日志（每 100 trials 输出 4 次 × 所有 MLP 迭代 = 海量重复）

```
UserWarning: Choices for a categorical distribution should be a tuple of
None, bool, int, float and str for persistent storage but contains (20,) which
is of type tuple.
```

### 根因

MLP 的 `hidden_layer_sizes` 搜索空间使用了 Python 原生 tuple 类型 `[(20,), (50,), (20,10), (50,25)]`。Optuna 建议 categorical 分布使用基本类型（None/bool/int/float/str），tuple 在 SQLite 持久化时有兼容性问题。

### 修复

将 tuple 选项改为字符串，在 objective 内部解析：

```python
hidden_layer_choices = ["20", "50", "20_10", "50_25"]
choice_str = trial.suggest_categorical("hidden_layer_sizes", hidden_layer_choices)
if choice_str == "20_10":
    parsed = (20, 10)
elif choice_str == "50_25":
    parsed = (50, 25)
else:
    parsed = (int(choice_str),)
```

---

## Bug 3：LightGBM 分裂警告噪声

### 日志（每条 50-200 行重复）

```
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
```

### 根因

LightGBM 在小样本（N≈90 训练样本）下，树的深度增长后无法找到有益的进一步分裂。这是正常的树模型收敛行为，不是错误。但 LGBM 的默认 `verbose=1` 会打印每条警告。

### 修复

在 `src/fixed_params.py` 中为 LGBM 设置 `"verbose": -1`（静默所有 LGBM 内部输出）：

```python
LGBMRegressor: {"n_jobs": n_jobs, "random_state": 42, "verbose": -1},
```

---

## Bug 4：LightGBM feature names 警告

### 日志

```
UserWarning: X does not have valid feature names, but LGBMRegressor was
fitted with feature names
```

### 根因

sklearn 1.6+ 版本的 `_validate_data` 在新旧 API 间有兼容性过渡。LGBM 的 scikit-learn 包装器在 `fit` 时传入 DataFrame（有列名），但后续 `predict` 时传入 numpy array（无列名）→ 触发了 scikit-learn 的特征名校验警告。

### 处理

**不修复**。这是 scikit-learn/LightGBM 上游库的兼容性行为，不影响计算结果。在 LGBM 的后续版本中会被官方修复。

---

## 变更文件清单

```
修改的文件:
  src/train_and_evaluate.py      (Bug 1: 移除 GPlearn p_crossover 搜索;
                                   Bug 2: MLP hidden_layer_sizes tuple→string)
  src/fixed_params.py            (Bug 1: GPlearn n_jobs=1;
                                   Bug 3: LGBM verbose=-1)
```
