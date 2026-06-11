# Debug Log & Revisions 7 — Final Technical Debt Resolution

> **日期:** 2026-05-11
> **依据:** `revision_8.md` 零散模块终极修复
> **范围:** 日志隔离 + 采样安全 + Pearson R 修正 + 算力黑洞清理

---

## 修改总览

| 修复项 | 文件 | 严重程度 | 状态 |
|---|---|---|---|
| 日志断层 | `src/logger_config.py` | 高危 | FIXED |
| 采样越界崩溃 | `src/feature_selection.py` | **致命** | FIXED |
| Pearson R NaN | `src/visualization.py` | 高 | FIXED |
| NoneType 迭代崩溃 | `src/validation_process.py` | 中 | FIXED |
| 算力黑洞 | `src/evaluate_and_plot.py` → `archive/` | 高 | ARCHIVED |

---

## 逐项修正详情

### 修复 1：日志断层 — 根日志器隔离 (`logger_config.py`)

**缺陷:** `setup_logger` 直接操作根日志器 `logging.getLogger()`，在多模型循环中每次调用都 `handlers.clear()`，导致前一模型的日志流被截断。

**修复:** 全面放弃根句柄操作，改为创建独立命名的子日志器 (`"AI4S_Optimization_{model_dir}"`)，设置 `propagate=False` 阻断向根日志器冒泡。

**效果:** 14 个模型串行训练的日志流互不干扰，再无静默丢失。

---

### 修复 2：采样越界崩溃 — 安全上限保护 (`feature_selection.py`)

**缺陷:** `n_samples = max(20, ...)` 在数据极小时（如最后一轮 SHAP-RFECV 仅剩 5 个特征、验证折仅 10 个样本）强行将采样数拔高到 20，随后 `np.random.choice(10, 20, replace=False)` 直接抛出 `ValueError` 崩溃。

**修复:** 在所有 KernelExplainer 路径中加入安全钳：

```python
n_samp_desired = max(20, min(int(len(X) * 0.3), 100))
n_samples = max(1, min(n_samp_desired, len(X_scaled)))  # 绝不越界
```

同时用局部 `rng = np.random.RandomState(42)` 替代全局 `np.random.seed(42)`，消除对 Optuna 采样器和树模型随机性的干扰。

**修复位置:** `src/feature_selection.py` — 简单路径 + 多折共识路径，共 2 处

---

### 修复 3：Pearson R 负数 NaN 崩溃 (`visualization.py`)

**缺陷:** `np.sqrt(R²)` 在非线性模型（SVR/RF/XGBoost）中数学上不等于 Pearson r，且在 $R^² < 0$ 时触发 `RuntimeWarning: invalid value encountered in sqrt` 并输出 `NaN`。

**修复:** 用 `np.corrcoef` 直接计算真实 Pearson 相关系数：

```python
def _pearson_r(y_true, y_pred):
    if len(y_true) > 1:
        return np.corrcoef(y_true, y_pred)[0, 1]
    return 0.0
```

`calculate_metrics` 新增 `r_train` 和 `r_test` 键；`add_plot_labels` 和 `add_plot_labels_standard` 改为使用 `metrics['r_train']` 和 `metrics['r_test']`。

---

### 修复 4：NoneType 迭代崩溃 (`validation_process.py`)

**缺陷:** `validation_data_produce` 签名默认 `H_feature_cols=None`，内部直接 `for col in H_feature_cols:`，不传参立即 `TypeError`。

**修复:** 函数头部加入一行防御：

```python
H_feature_cols = H_feature_cols or []
B_feature_cols = B_feature_cols or []
```

---

### 修复 5：算力黑洞清理 — 归档 `evaluate_and_plot.py`

**缺陷评估:**
- 两个函数生成图表后既不 `savefig` 也不 `show` → 100% 无效算力开销
- 不调用 `plt.close()` → 密集迭代中内存泄漏
- 文本标注硬编码绝对物理坐标 `plt.text(100, -5)` → 数据区间改变时文字越界
- 使用 `IPython.display.display` → 生产环境 `nohup` 下无意义

**处置:**
1. 从 `iterative_optimization.py` 中移除 `evaluate_and_plot()` 调用和 import（主流程已有完备的 `plot_scatter` 负责所有可视化）
2. 将 `src/evaluate_and_plot.py` 移入 `archive/`，首行注入 `raise DeprecationWarning`

---

## 修正后的代码使用指导

命令行参数无变化。日志系统行为变化：

| 行为 | 修改前 | 修改后 |
|---|---|---|
| 多模型日志 | 每切换模型清空根日志器，前期日志丢失 | 各模型独立命名日志器，完整保留 |

---

## 遗留问题与建议

1. **`validation_process.py` 的静默误差**: 特征提取用 `.iloc[0]` 假设同名分子特征绝对一致。如果存在浮点数截断导致的微小差异，当前代码会静默忽略。建议后续版本中加入跨行数值一致性检查（`assert all close`）。

2. **`logger_config.py` 的 `unique_name`**: 对 `basename(model_dir)` 的依赖意味着必须为每个模型创建不同目录。如果未来某次运行中多个模型共用一个目录，唯一性会失效。当前架构下每个模型有独立子目录，不会触发此问题。

3. **`archive/` 目录**: `evaluate_and_plot.py` 是第 5 个归档的废弃模块。

---

## 变更文件清单

```
修改的文件:
  src/logger_config.py             (根日志器隔离: propagate=False + 命名句柄)
  src/feature_selection.py         (2 处采样越界防御 + 局部 RandomState)
  src/visualization.py             (Pearson R 替换 np.sqrt(R²) + 新增 _pearson_r)
  src/validation_process.py        (NoneType 缺省值防御)
  src/iterative_optimization.py    (移除 evaluate_and_plot 调用和 import)

归档的文件 (移至 archive/):
  src/evaluate_and_plot.py         → archive/evaluate_and_plot.py
```
