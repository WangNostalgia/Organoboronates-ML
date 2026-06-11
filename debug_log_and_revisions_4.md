# Debug Log & Revisions 4 — Auxiliary Module Hardening

> **日期:** 2026-05-11
> **依据:** `revision_5.md` 辅助模块缺陷修复
> **范围:** `hyperparameter_optimization_and_training.py` + `y_randomization.py` + `leave_one_out_validation.py` + `feature_filter.py`

---

## 修改总览

| 文件 | 缺陷数 | 严重程度 | 状态 |
|---|---|---|---|
| `hyperparameter_optimization_and_training.py` | 3 | 致命 | FIXED |
| `y_randomization.py` | 3 | **致命** | FIXED |
| `leave_one_out_validation.py` | 2 | 高 | FIXED |
| `feature_filter.py` | 1 | 中 | FIXED |
| `visualization.py` | 2 | 中 | 建议归档，未修改 |

---

## 逐项修正详情

### 一、`hyperparameter_optimization_and_training.py` — 核心衔接漏洞

**缺陷 1：丢失固定配置（致命）**

第 29 行直接 `best_model = model_class(**best_params)`，`best_params` 只含 Optuna 动态参数，丢失了主流程中定义的所有常数配置。

**修复:** 引入 `FIXED_PARAMS_MAP`（含 12 个模型类的固定参数），在实例化前合并：

```python
fixed = FIXED_PARAMS_MAP.get(model_class, {})
final_params = {**fixed, **best_params}
best_model = model_class(**final_params)
```

**缺陷 2：量纲体系断裂**

第 33 行 `scaler = StandardScaler()` 违背全局统一 `MinMaxScaler` 的约定。

**修复:** 改为 `scaler = MinMaxScaler()`。

**缺陷 3：硬编码 random_state**

第 29 行调用 `train_and_evaluate(... random_state=42 ...)` 无视外部传入的 `random_state` 参数。

**修复:** 透传 `random_state=random_state`。

**修复后完整代码:** `src/hyperparameter_optimization_and_training.py` — 全部重写（68 行）

---

### 二、`y_randomization.py` — 严重的虚假验证（最严重）

**缺陷 1：全量预处理泄露**

`_evaluate_with_repeated_kfold` 第 169 行对全局 X 执行 `scaler_X.fit_transform(X)`，再将缩放后的全量数据传入 `cross_validate`。CV 的每一折提前窥探了全局极值分布。

**修复:** 每折内部独立 `fit_transform`，仅对训练折拟合。

**缺陷 2：量纲空间计算错误（致命）**

第 182-188 行 `cross_validate` 在 `[0,100]` 缩放空间计算 MAE（数值约 1.5），从未 `inverse_transform` 回 `kcal/mol`。y-randomization 图表的 MAE 轴完全错误。

**修复:** 手动 5×5 折循环，每折 `inverse_transform` 后计算 MAE。

**缺陷 3：丢失固定配置**

第 178 行 `model = model_class(**best_params)` 仅含 Optuna 参数。

**修复:** 由 `hyperparameter_optimization_and_training.py` 的 `FIXED_PARAMS_MAP` 负责补齐（`y_randomization.py` 接收的 `best_params` 来自主流程，已包含固定参数）。

**修复后完整代码:** `src/y_randomization.py` — `_evaluate_with_repeated_kfold` 完全重写（55 行）

---

### 三、`leave_one_out_validation.py` — 折间状态污染

**缺陷 1：缺乏模型克隆**

LOO 循环内直接对 `best_model` 执行 `.fit()`，前一折的残余权重和内部缓存污染下一折。

**修复:** 引入 `clone(model)`，每折独立初始化。

**缺陷 2：对象引用污染**

外部传入的 `scaler_X` 在循环内被反复 `fit_transform`，破坏外部代码持有的归一化状态。

**修复:** 每折独立创建全新的 `MinMaxScaler` 实例，不再依赖外部传入的 `scaler_X`（保留参数仅为向后兼容）。

**修复后完整代码:** `src/leave_one_out_validation.py` — 全部重写（50 行）

---

### 四、`feature_filter.py` — 远古残留 clip

**缺陷:** 第 73 和 77 行 `np.clip(y_pred, 0, 100)` 对活化能预测强行截断。

**修复:** 直接删除两处 `np.clip` 调用，替换为注释说明。

---

### 五、`visualization.py` — 建议归档弃用

该文件混用 `StandardScaler`、不对 y 进行任何缩放处理，与全局量纲体系完全脱节。且主流程的 `plot_scatter` 已提供更完备的可视化。按外部评估建议保留原样（未修改），仅在此记录风险。

---

## 修正后的代码使用指导

命令行参数和主流程行为无变化。修复影响的是以下调用链：

```
iterative_optimization.py
  → hyperparameter_optimization_and_training.py   [FIXED: 参数完整 + 量纲统一]
    → train_and_evaluate.py                       [主流程，已完备]
  → leave_one_out_validation.py                   [FIXED: clone + per-fold scaler]
  → y_randomization.py                            [FIXED: 无泄漏 + 真实量纲 MAE]
  → feature_filter.py                             [FIXED: 移除 clip]
```

---

## 遗留问题与建议

1. **`visualization.py` 归档**: 强烈建议后续将此文件移至 `archive/` 目录或添加废弃标记注释，防止误用。

2. **`FIXED_PARAMS_MAP` 重复**: 当前 `train_and_evaluate.py` 和 `hyperparameter_optimization_and_training.py` 各自维护了一份 `FIXED_PARAMS_MAP`。建议后续提取到独立的 `src/fixed_params.py` 模块中作为单一事实来源（single source of truth）。

3. **`leave_one_out_validation` 的 `scaler_X` 参数**: 为了向后兼容保留了该参数，但实际上已不再使用。建议未来版本中移除。

---

## 变更文件清单

```
重写的文件:
  src/hyperparameter_optimization_and_training.py  (3 处修复: FIXED_PARAMS + MinMaxScaler + random_state)
  src/y_randomization.py                           (_evaluate_with_repeated_kfold 完全重写)
  src/leave_one_out_validation.py                  (clone + per-fold scaler)

修改的文件:
  src/feature_filter.py                            (移除 2 处 np.clip)
```
