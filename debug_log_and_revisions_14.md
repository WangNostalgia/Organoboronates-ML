# Debug Log & Revisions 14 — GPLearn Import Bug & Font Warning Fix

> **日期:** 2026-05-19
> **运行日志:** `train.log` (8604 行)
> **范围:** 2 个 Bug 修复

---

## 修改总览

| # | 问题 | 严重程度 | 状态 |
|---|---|---|---|
| 1 | `NameError: name 'GPLearnRegressor' is not defined` | **致命** | FIXED |
| 2 | `WARNING: Font family 'Arial' not found` | 低 | FIXED |

---

## Bug 1：GPLearnRegressor 未导入（致命）

### 日志信息

```
[W 2026-05-19 02:21:14,807] Trial 0 failed with parameters: {}
NameError: name 'GPLearnRegressor' is not defined
```

### 崩溃位置

`src/train_and_evaluate.py` 第 158 行，`objective()` 函数内部。

### 根因

在将 `SISSORegressor` → `GPLearnRegressor` 重命名时，`src/train_and_evaluate.py` 中 `objective()` 函数引用了 `GPLearnRegressor` 类，但没有导入它。`objective()` 是嵌套函数，无法访问外部 import。

### 修复

在 `src/train_and_evaluate.py` 顶部添加：

```python
from src.gplearn_wrapper import GPLearnRegressor
```

---

## Bug 2：Arial 字体缺失警告

### 日志信息

```
WARNING - findfont: Font family 'Arial' not found.
```

### 根因

`src/visualization.py` 中所有 `plt.text()` 调用硬编码了 `fontname='Arial'`。Arial 是 Windows 系统字体，Linux 默认未安装。在 matplotlib 运行时，每次调用 `plt.text()` 都会触发字体回退警告。

### 修复

将所有 `fontname='Arial'` 替换为 `fontname='DejaVu Sans'`：

| 位置 | 修改 |
|---|---|
| `plot_scatter` → `add_plot_labels` | 9 处 `plt.text()` 字体参数 |
| `add_plot_labels_standard` | 9 处 `plt.text()` 字体参数 |
| `plot_scatter_standard` 的默认参数 | `fontname='Times New Roman'` → `fontname='DejaVu Sans'` |

**选择 DejaVu Sans 的原因：**
- 随 matplotlib 打包安装，**所有平台**（Windows/Linux/macOS）均可使用
- 清晰的无衬线字体，适合科学图表
- 与原有代码中 `iterative_optimization.py` 第 27 行 `plt.rcParams['font.family'] = 'DejaVu Sans'` 保持一致

文件顶部添加了说明注释：

```python
# All fonts use 'DejaVu Sans' (bundled with matplotlib) for cross-platform compatibility.
# 'Arial' and 'Times New Roman' are not installed by default on Linux.
```

---

## 变更文件清单

```
修改的文件:
  src/train_and_evaluate.py      (Bug 1: 添加 from src.gplearn_wrapper import GPLearnRegressor)
  src/visualization.py           (Bug 2: Arial → DejaVu Sans, Times New Roman → DejaVu Sans)
```
