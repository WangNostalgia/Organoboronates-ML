# Debug Log & Revisions 9 — 1-SE Rule Fix & Manual Selection

> **日期:** 2026-05-11
> **依据:** `revision_10.md`
> **范围:** Fractional 1-SE + 手动选择脚本 + 散点图增强

---

## 修改总览

| 任务 | 描述 | 状态 |
|---|---|---|
| 任务一 | Fractional 1-SE 替代标准 1-SE | DONE |
| 任务二 | 手动特征选择 + 独立绘图脚本 | DONE |
| 任务三 | 散点图增加 RKfold MAE/R² | DONE |

---

## 任务一：Fractional 1-SE 法则

### 问题

标准 1-SE 法则在 XGBoost 的 Path Summary 上将模型推向了 3 特征（实质欠拟合）：

```
6 特征: MAE=2.7343 ± 0.4275  ← 绝对最小值但 std 异常暴增
3 特征: MAE=2.8900           ← 被选中（标准 1-SE: threshold=2.7343+0.4275=3.1618）
4 特征: MAE=2.7532 ± 0.3064  ← 真正的"甜点"（std 最低、MAE 仅比最小值高 0.02）
```

**根因:** 小样本（N=88）下 RepeatedKFold 的折间 std 天然偏大。绝对最小值点（6 特征）的 std 异常暴增（0.4275），用全量 std 作为惩罚跨度 → 容忍阈值被拉到 3.16 → min_features 直接触底。

### 修复

改为 **Fractional 1-SE**，`PENALTY_ALPHA = 0.25`：

```
threshold = min_mae + 0.25 × min_std
         = 2.7343 + 0.25 × 0.4275
         = 2.8411
```

| 特征数 | MAE | vs threshold(2.84) | 判定 |
|---|---|---|---|
| 3 | 2.8900 | > 2.84 | ❌ 不合格（成功拦截欠拟合） |
| 4 | 2.7532 | ≤ 2.84 | ✅ 达标 + 特征最少 → **选中** |
| 5 | 2.7738 | ≤ 2.84 | 达标但非最少 |

**修复位置:** `src/iterative_optimization.py` — 1-SE 决策块

**日志输出示例:**
```
★ Fractional 1-SE (α=0.25): absolute min at 6 feat (MAE=2.7343 ± 0.4275),
  threshold=2.8411, selecting 4 feat (MAE=2.7532)
```

---

## 任务二：手动特征选择 + 独立绘图脚本

### 新文件

`example/manual_selection_and_plot.py` — 可独立运行的完整脚本。

### 使用流程

**第 1 步:** 创建 CSV 指定每模型的手动特征数：

```csv
model_name,n_features
SVR,5
RandomForest,7
XGBoost,4
GradientBoosting,6
```

**第 2 步:** 运行脚本：

```bash
python example/manual_selection_and_plot.py
```

### 功能

| 功能 | 状态 |
|---|---|
| 读取 CSV 中指定的模型和特征数 | ✅ |
| 自动找到对应特征的 iteration checkpoint | ✅ |
| 生成 `final_scatter_*.png` + `*_outliers.csv` | ✅ |
| 预测外部数据 | 🔒 已实现但注释掉（待以后启用） |

### 输出

```
models/manual_selection_plots/
├── SVR/
│   ├── final_scatter_5feat.png
│   └── final_scatter_5feat_outliers.csv
├── XGBoost/
│   ├── final_scatter_4feat.png
│   └── final_scatter_4feat_outliers.csv
└── ...
```

---

## 任务三：散点图增加 RKfold MAE 和 RKfold R²

### 修改

`plot_scatter` 新增可选参数 `rkf_mae=None` 和 `rkf_r2=None`。当提供时，散点图右侧额外显示这两个来自 5×5 RepeatedKFold 的主要指标。

**修改文件:**
- `src/visualization.py`: `plot_scatter`, `add_plot_labels`, `add_plot_labels_standard` 均新增参数
- `src/iterative_optimization.py`: 调用 `plot_scatter` 时传入 `rkf_mae_opt_mean` 和 `rkf_r2_opt_mean`

**散点图右侧文本（修改后）:**

```
Pearson R_train: 0.9234
Pearson R_test:  0.8745
RMSE_test:       3.25
MAE_test:        2.13
MAE_mean:        2.31
RKfold MAE:      2.25    ← 新增
RKfold R²:       0.81    ← 新增
```

向后兼容：不传 `rkf_mae`/`rkf_r2` 时行为与原来完全一致。

---

## 修正后的代码使用指导

### 命令行参数

无变化。Fractional 1-SE 的 α 值硬编码为 0.25。如需调整，编辑 `iterative_optimization.py` 中的 `PENALTY_ALPHA`。

### 手动选择流程

```
训练完成 → 查看 Path Summary → 填写 manual_feature_selection.csv
         → python example/manual_selection_and_plot.py
         → 生成最终散点图 + outliers CSV
```

---

## 变更文件清单

```
修改的文件:
  src/iterative_optimization.py          (任务一: Fractional 1-SE + 任务三: 传递 rkf_mae/rkf_r2 到 plot_scatter)
  src/visualization.py                   (任务三: plot_scatter/add_plot_labels 新增 rkf_mae/rkf_r2 参数)

新增的文件:
  example/manual_selection_and_plot.py   (任务二: 手动选择+独立绘图完整脚本)
```
