# Debug Log & Revisions 12 — Seed Consistency & Standalone y-Randomization

> **日期:** 2026-05-12
> **依据:** `revision_13.md`
> **范围:** 种子一致性 + GPR 搜索扩容 + 独立 y-Randomization 脚本

---

## 修改总览

| # | 修改 | 文件 | 状态 |
|---|---|---|---|
| 1 | 修复硬编码 `random_state=42` → 透传参数 | `src/train_and_evaluate.py` | DONE |
| 2 | GPR `length_scale` 扩容 0.1~10 → 1e-3~1e3 | `src/train_and_evaluate.py` | DONE |
| 3 | 独立 y-Randomization 脚本 | `example/standalone_y_randomization.py` (新建) | DONE |
| — | `plot_scatter` 参数不存在 | `src/visualization.py` | 无需修改（已存在） |

---

## 逐项修正

### 修复 1：消除残留硬编码种子

**位置:** `src/train_and_evaluate.py` 第 395 行和第 414 行

**修改前:**
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42        # ← 硬编码
)
logger.info("random_state = 42")                 # ← 硬编码字符串
```

**修改后:**
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_state  # ← 透传调用方种子
)
logger.info(f"random_state = {random_state}")        # ← 动态输出
```

**效果:** legacy 报告指标的单次划分现在使用与其他所有 KFold 一致的种子（40），不再与优化过程割裂。

---

### 修复 2：GPR `length_scale` 搜索扩容

**修改前:** `length_scale: 0.1 ~ 10`（log-uniform）
**修改后:** `length_scale: 1e-3 ~ 1e3`（log-uniform）

同时重构了 GPR objective 的参数构建：将 `trial.suggest_float("length_scale", ...)` 提前到分支外部作为 `l_scale` 变量，三个内核分支（RBF/Matern/RBF+White）共用。避免了之前三个分支中各调用一次 `suggest_float` 导致代码膨胀和潜在的注释越界语法错误。

**效果:** GPR 在 MinMaxScaler 缩放后的特征空间中，核函数的长度尺度搜索不再频繁触达边界，ConvergenceWarning 预计大幅减少。

---

### 新增：独立 y-Randomization 脚本

**文件:** `example/standalone_y_randomization.py`

**设计思路:** 主流程中的 y-Randomization 已被注释（每模型 2500 fits，树模型数小时）。此脚本独立运行，在最终确认模型和特征数后执行一次完整的置换检验作为论文佐证。

**使用方法:**

```python
# 编辑脚本顶部的 CONFIGURATION 区域:
MODEL_DIR = 'models/SVR'     # 模型目录
N_FEATURES = 5                # 特征数
N_PERMS = 100                 # 排列次数（快速测试可设为 30）

# 运行:
python example/standalone_y_randomization.py
```

**输出示例:**
```
============================================================
  y-Randomization Results
============================================================
  Original MAE:  2.4500 ± 0.3100
  Original R²:   0.7700 ± 0.0500
  Random MAE:    5.1200 ± 0.4800
  Random R²:     -0.1500 ± 0.1200
  p-value (MAE): 0.0000
  Result:        PASSED ✓
```

**特性:**
- 自动从 `models/<ModelName>/` 加载指定特征数的 checkpoint
- 自动检测模型类（也可手动指定 `MANUAL_MODEL_CLASS`）
- 使用与主流程完全一致的 5×5 RepeatedKFold 评估（导入 `src.evaluation`）
- 生成 `models/y_randomization_<ModelName>.png` 分布直方图

---

## 关于 `plot_scatter` 参数不兼容声称的核查

外部评估声称 `visualization.py` 的 `plot_scatter` 未包含 `rkf_mae` 和 `rkf_r2` 参数。**经核查，这两个参数已在 revision 9 中正确添加**：

```python
def plot_scatter(..., r2_loo=None, rkf_mae=None, rkf_r2=None):
```

不需要修改。

---

## 变更文件清单

```
修改的文件:
  src/train_and_evaluate.py                (修复 1: 种子一致性; 修复 2: GPR length_scale 扩容)

新增的文件:
  example/standalone_y_randomization.py    (独立 y-Randomization 脚本)
```
