# Debug Log & Revisions 6 — Architecture Closure

> **日期:** 2026-05-11
> **依据:** `revision_7.md` 终极三刀
> **范围:** 配置中心贯通 + 评估中心提取 + 死代码清理

---

## 修改总览

| 任务 | 描述 | 状态 |
|---|---|---|
| 刀法 1 | 创建 `src/evaluation.py` 统一评估中心 | DONE |
| 刀法 2 | `y_randomization.py` 复用统一评估中心 | DONE |
| 刀法 3 | `objective()` 使用 `get_fixed_params()` 作为底座 | DONE |
| 刀法 4 | `leave_one_out_validation` 签名净化 + `visualization.py` 死代码清理 | DONE |

---

## 逐项修正详情

### 刀法 1+2：统一评估中心 `src/evaluation.py` + 消除代码拷贝

**痛点:** `train_and_evaluate.py` 和 `y_randomization.py` 各自维护了一份功能完全相同的 `repeated_kfold_evaluate` 实现（各 ~55 行），严重违反 DRY 原则。

**修复方案:** 新建 `src/evaluation.py`，暴露唯一接口 `repeated_kfold_evaluate()`。两个调用方均从此导入。

**变更:**

| 文件 | 修改前 | 修改后 |
|---|---|---|
| `src/train_and_evaluate.py` | 内含 55 行本地 `repeated_kfold_evaluate` 函数 | `from src.evaluation import repeated_kfold_evaluate` (1 行) |
| `src/y_randomization.py` | 内含 55 行本地 `_evaluate_with_repeated_kfold` 函数 | `from src.evaluation import repeated_kfold_evaluate`；删除死函数 |
| `src/evaluation.py` | (不存在) | 新建，93 行 |

**返回值兼容性:** `evaluation.py` 同时返回主流程键名(`rkf_mae_mean`等)和 y-randomization 兼容别名(`mae_mean`等)，单次调用同时满足双方需求。

---

### 刀法 3：`objective()` 贯通配置中心

**痛点:** `objective()` 内部每个模型的固定参数直接写死。如果修改 `fixed_params.py`（如将 MLP 的 `max_iter` 从 2000 改为 5000），Optuna 调参仍使用旧值，导致调参与部署配置割裂。

**修复方案:** 每个模型的 `params` 从 `get_fixed_params()` 拉取底座，仅在底座之上覆盖 Optuna 动态参数：

```python
# 修改前（写死）:
params = {"C": trial.suggest_float("C", ...), "kernel": "rbf", "tol": 1e-3, ...}

# 修改后（从配置中心拉取底座）:
base_params = get_fixed_params(model_class, n_jobs).copy()
params = {**base_params, "C": trial.suggest_float("C", ...)}
```

所有 16 个模型均改为 `{**base_params, <Optuna动态参数>}` 模式。当前工程中修改 `fixed_params.py` 任一值，Optuna 搜索空间**自动同步**。

**代码量:** `objective()` 从 ~180 行缩减到 ~120 行（约 33% 减少）。

---

### 刀法 4：签名净化与死代码清理

#### 4a: `leave_one_out_validation` 签名净化

移除闲置且误导的 `scaler_X` 参数。该参数自上一轮重构（每折独立创建 MinMaxScaler）后已完全不参与内部逻辑。

**变更:**
- `src/leave_one_out_validation.py`: 签名 `(best_model, scaler_X, X_model, y)` → `(best_model, X_model, y)`
- `src/iterative_optimization.py`: 2 处调用同步更新
- `src/visualization.py`: 1 处调用同步更新

#### 4b: `visualization.py` 死函数清理

删除两个已无任何调用方的函数：
- `plot_r2_on_100_random_samples` (~35 行)：混用 `StandardScaler`、不对 y 做缩放
- `plot_r2_distribution` (~27 行)：量纲断层

同时清理相关冗余导入：`StandardScaler`、`train_test_split`、`leave_one_out_validation`。

---

## 修正后的代码使用指导

### 新增模块

| 模块 | 职责 | 导入方式 |
|---|---|---|
| `src/fixed_params.py` | 全项目固定参数唯一来源 | `from src.fixed_params import get_fixed_params` |
| `src/evaluation.py` | 全项目 5×5 RepeatedKFold 唯一评估中心 | `from src.evaluation import repeated_kfold_evaluate` |

### 命令行参数

无变化。

```bash
nohup uv run python main.py --n_trials 100 --mae_threshold 2.0 --min_features 3 --n_jobs -1 > train.log 2>&1 &
```

---

## 遗留问题与建议

1. **`src/validation_process.py`**: 该模块用于生成 sub_H×sub_B 验证空间组合，当前未被主动调用但功能独立有效。如需完整的 Stage 6 外部验证流程，可保留；否则可归档。

2. **`CLAUDE.md` 已更新**: 反映了新模块(`fixed_params.py`, `evaluation.py`)和归档目录(`archive/`)。

3. **文档中 `model_feature_filter.py` 引用**: README、README_CN、user_manual 中仍有对已归档模块的少量引用。这些历史文档引用的更新建议放在下一轮文档同步中批量处理。

---

## 变更文件清单

```
新建的文件:
  src/evaluation.py                              (统一评估中心: 93 行)

修改的文件:
  src/train_and_evaluate.py                      (移除本地 repeated_kfold_evaluate → 导入 evaluation.py;
                                                   objective() 全部 16 模型改用 get_fixed_params() 底座: 180→120 行)
  src/y_randomization.py                         (删除 _evaluate_with_repeated_kfold → 导入 evaluation.py)
  src/leave_one_out_validation.py                (移除闲置 scaler_X 参数)
  src/iterative_optimization.py                  (2 处 leave_one_out_validation 调用更新)
  src/visualization.py                           (删除 plot_r2_on_100_random_samples + plot_r2_distribution;
                                                   清理 StandardScaler/train_test_split/leave_one_out_validation 导入)
  CLAUDE.md                                      (架构描述更新)
```
