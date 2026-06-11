# Debug Log & Revisions 5 — Architecture Finalization

> **日期:** 2026-05-11
> **依据:** `revision_6.md` 终极重构任务
> **范围:** 配置中心提取 + 废弃模块归档 + 代码去重

---

## 修改总览

| 任务 | 描述 | 状态 |
|---|---|---|
| 任务一 | 提取 `src/fixed_params.py` 作为单一事实来源 | DONE |
| 任务二 | 精简 `hyperparameter_optimization_and_training.py` | DONE |
| 任务三 | 归档 4 个废弃模块至 `archive/` | DONE |
| 附带 | 移除 `main.py` 中未使用的 import | DONE |

---

## 逐项修正详情

### 任务一：创建单一事实来源配置中心 `src/fixed_params.py`

**痛点:** `train_and_evaluate.py` 和 `hyperparameter_optimization_and_training.py` 各自维护了一份 16 模型 × 多行参数的 `FIXED_PARAMS_MAP`，完全重复。

**解决方案:** 新建 `src/fixed_params.py`，暴露唯一接口 `get_fixed_params(model_class, n_jobs=-1)`。

**受影响文件及修改:**

| 文件 | 修改前 | 修改后 |
|---|---|---|
| `src/train_and_evaluate.py` | 内联 16 模型 × 多行参数字典 (~40 行) | `from src.fixed_params import get_fixed_params` → 1 行 |
| `src/hyperparameter_optimization_and_training.py` | 内联 16 模型 × 多行参数字典 (~45 行) + 重复 import 列表 | `from src.fixed_params import get_fixed_params` → 1 行，代码从 68 行缩减到 48 行 |

**效果:** 未来调整任何模型的固定参数（如加深树深度、改变收敛容忍度），只需修改 `src/fixed_params.py` 一处，全项目自动同步。

**`get_fixed_params()` 接口:**

```python
from src.fixed_params import get_fixed_params

# 在任何需要实例化完整模型的地方:
final_params = {**get_fixed_params(model_class, n_jobs), **best_params}
model = model_class(**final_params)
```

---

### 任务二：精简 `hyperparameter_optimization_and_training.py`

**修改内容:**

1. **导入共享配置:** `from src.fixed_params import get_fixed_params` 替代内联字典
2. **透传 random_state:** 不再硬编码 `random_state=42`
3. **移除冗余 import:** 不再需要 12 个 sklearn/xgboost/lightgbm 类引用（它们现在只在 `fixed_params.py` 中导入）

**修改前** (68 行, 含冗长 import 和 45 行字典):

```python
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.svm import SVR
# ... 10 more imports ...

FIXED_PARAMS_MAP = {
    Lasso: {"max_iter": 10000, ...},
    ElasticNet: {...},
    # ... 14 more entries ...
}
```

**修改后** (48 行):

```python
from src.fixed_params import get_fixed_params

final_params = {**get_fixed_params(model_class, n_jobs), **best_params}
best_model = model_class(**final_params)
```

---

### 任务三：废弃模块归档

共归档 **4 个模块** 至 `archive/` 目录。每个归档文件均在首行注入 `raise DeprecationWarning(...)` 阻止误调用。

#### 已归档文件清单

| 原路径 | 归档路径 | 归档原因 |
|---|---|---|
| `src/feature_filter.py` | `archive/feature_filter.py` | 穷举组合搜索，功能已被 SHAP-RFECV 完全替代；存在数据标签错位 Bug（y 全量 vs X_train 子集不对齐） |
| `example/model_feature_filter.py` | `archive/model_feature_filter.py` | 调用 feature_filter 的示例脚本，一同废弃 |
| `src/feature_importance_analysis.py` | `archive/feature_importance_analysis.py` | 旧的 SHAP 分析模块，功能已嵌入 `src/feature_selection.py` |
| `src/feature_correlation_analysis.py` | `archive/feature_correlation_analysis.py` | 旧的相关性分析模块，功能已嵌入 `src/feature_selection.py`（shap_rfecv_select_worst_feature 内计算相关系数矩阵） |

#### 未归档但部分废弃的文件

| 文件 | 状态 | 原因 |
|---|---|---|
| `src/visualization.py` | **保留**（`plot_scatter` 仍被 `iterative_optimization.py` 和 `main.py` 活跃使用） | 仅 `plot_r2_on_100_random_samples` 函数未被调用；从 `main.py` 中移除了该函数的 import |
| `src/evaluate_and_plot.py` | **保留** | 仍被 `iterative_optimization.py` 调用 |

#### DeprecationWarning 格式

每个归档文件顶部注入以下代码块：

```python
# ==============================================================================
# DEPRECATED: This module has been archived.
# [具体废弃原因]
# DO NOT import or call from active workflows.
# ==============================================================================
raise DeprecationWarning("This legacy module has been archived. ...")
```

---

## 修正后的代码使用指导

### 新增模块

```python
from src.fixed_params import get_fixed_params

# 获取某模型的固定参数
fixed = get_fixed_params(SVR, n_jobs=-1)
# → {'kernel': 'rbf', 'tol': 1e-3, 'max_iter': 10000, 'cache_size': 1000}
```

### 命令行参数

无变化。

### 废弃模块迁移

如果外部脚本或 notebook 之前导入了被归档的模块，需要改为：

| 旧用法 | 新用法 |
|---|---|
| `from src.feature_filter import feature_filter` | 不再可用；使用 SHAP-RFECV（已在 `iterative_optimization.py` 中自动执行） |
| `from src.feature_importance_analysis import feature_importance_analysis` | 不再可用；使用 `src/feature_selection.py` 中的 `shap_rfecv_select_worst_feature()` |
| `python example/model_feature_filter.py` | 不再可用；使用 `python main.py --min_features 3` 自动执行 SHAP-RFECV |

---

## 遗留问题与建议

1. **`visualization.py` 中的 `plot_r2_on_100_random_samples` 函数:** 该函数已无任何调用方，属于 dead code。建议后续版本中移除该函数及相关 import（`StandardScaler`, `LeaveOneOut`），进一步精简该文件。当前保留以防止破坏性变更。

2. **`leave_one_out_validation` 的 `scaler_X` 参数:** 该参数已完全不被内部使用（每折独立创建 scaler），仅为向后兼容保留。建议后续清理函数签名和所有调用点。

3. **`src/validation_process.py`:** 该模块用于生成 sub_H×sub_B 组合验证空间，当前未被主动调用但功能独立有效。如果不再需要外部验证数据集生成功能，也可归档。

4. **`CLAUDE.md`、`README.md`、`README_CN.md`、`user_manual.md`:** 这些文档中仍有对 `example/model_feature_filter.py` 的引用。文档更新不在本次代码修改范围内，建议后续统一更新。

---

## 变更文件清单

```
新增的文件:
  src/fixed_params.py                          (单一事实来源配置中心)

修改的文件:
  src/train_and_evaluate.py                    (移除内联 FIXED_PARAMS_MAP → 导入 shared)
  src/hyperparameter_optimization_and_training.py  (精简: 68行→48行, 导入 shared)
  main.py                                      (移除未使用的 plot_r2_on_100_random_samples import)

归档的文件 (移至 archive/):
  src/feature_filter.py                        → archive/feature_filter.py
  example/model_feature_filter.py              → archive/model_feature_filter.py
  src/feature_importance_analysis.py           → archive/feature_importance_analysis.py
  src/feature_correlation_analysis.py          → archive/feature_correlation_analysis.py
```
