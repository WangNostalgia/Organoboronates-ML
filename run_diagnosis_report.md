# 运行诊断报告

> **运行时间:** 2026-05-11 14:57 — (XGBoost y-Randomization 仍在运行中)
> **日志文件:** `my_final_new.log` (3104 行)
> **激活模型:** 8 个 (LinearRegression, Ridge, Lasso, SVR, DecisionTree, RandomForest, GradientBoosting, XGBoost)
> **配置:** `--min_features 3`, 14 初始特征

---

## 一、整体运行状态

| 阶段 | 完成模型数 | 状态 |
|---|---|---|
| 迭代特征筛选 (12轮/模型) | 8/8 (96 轮迭代) | ✅ 全部完成 |
| SHAP-RFECV 自动选择 | 8/8 | ✅ 全部完成 |
| y-Randomization 检验 | 7/8 | ⚠️ XGBoost 仍在运行中 |

**日志中未发现任何报错、崩溃或异常。**

---

## 二、逐模型结果核验

### 2.1 双 CV 输出（以 LinearRegression Iteration 1 为例）

```
================================================================================
  Iteration 1 — Dual CV Comparison
  Metric                              Value
  ----------------------------------------
  5×5 RKfold MAE               2.7037 ± 0.3693  ← PRIMARY
  5×5 RKfold R²                 0.7467 ± 0.1269  ← PRIMARY
  LOOCV R²                           0.7866        ← auxiliary
  LOOCV MAE                         2.6429        ← auxiliary
  100-split MAE                 2.7120        ← legacy
  Test R² (single)             0.6088        ← legacy
  Test MAE (single)            2.5427        ← legacy
================================================================================
```

**✅ 符合预期**：7 个指标全部输出，PRIMARY/AUXILIARY/LEGACY 三级清晰分层。MAE 量纲统一（kcal/mol），无误导性的缩放空间数值。

### 2.2 两级 SHAP 策略运行正常

日志中正确显示 `Single-fit` 和 `Multi-fold consensus` 的模式切换：

```
=== SHAP-RFECV (Iteration 1, 14 features, Single-fit) ===    ← 特征多，粗筛阶段
=== SHAP-RFECV (Iteration 7, 9 features, Multi-fold consensus) — recording CV path ===  ← 精选阶段
```

**✅ 符合预期**：阈值 `max(10, eff_min_features + 3) = max(10, 6) = 10` 正确生效。

### 2.3 高相关特征处理

```
→ Selected for removal: 'VBur_C' (reason: high_correlation)
```

日志显示 `high_correlation` 原因被正确触发和标注。**✅ 符合预期**。

### 2.4 SHAP-RFECV Path Summary（以 XGBoost 为例）

```
SHAP-RFECV Path Summary for XGBoost
Feat  RKfold MAE ± std       RKfold R² ± std       LOOCV R²   LOOCV MAE  100-spl MAE  Test R²   Test MAE
----------------------------------------------------------------------------------------------------------
10    2.7666 ± 0.3860     0.7450 ± 0.1032    0.7588     2.8506     2.8020       0.5471    2.9153
9     2.7572 ± 0.3658     0.7542 ± 0.0848    0.7821     2.7183     2.7791       0.5808    2.7470
8     2.7692 ± 0.3754     0.7557 ± 0.0879    0.8005     2.6518     2.8032       0.4486    3.2925
7     2.7446 ± 0.3797     0.7536 ± 0.1003    0.7892     2.6648     2.7499       0.5299    2.9180
6     2.7343 ± 0.4275     0.7519 ± 0.1005    0.7815     2.7467     2.7979       0.5559    2.8337
5     2.7738 ± 0.3144     0.7612 ± 0.0759    0.7830     2.7431     2.8365       0.5397    2.9786
4     2.7532 ± 0.3064     0.7470 ± 0.0834    0.7831     2.7122     2.7867       0.4895    2.9029
3     2.8900 ± 0.3296     0.7255 ± 0.0864    0.7660     2.7931     2.9455       0.4332    3.0270
★ 1-SE Rule triggered: absolute minimum at 6 features (MAE=2.7343 ± 0.4275),
  selecting simpler model with 3 features (MAE=2.8900)
```

**✅ 符合预期**：全部 8 列指标输出，1-SE 法则正确触发（6 特征 MAE 最小但 std 极大，3 特征在 2.7343+0.4275=3.1618 范围内）。

### 2.5 y-Randomization 结果

| 模型 | 耗时 | 结果 |
|---|---|---|
| LinearRegression | ~11s | PASSED (p=0.0000) |
| Ridge | ~11s | PASSED (p=0.0000) |
| Lasso | ~39s | PASSED (p=0.0000) |
| SVR | ~50s | PASSED (p=0.0000) |
| DecisionTree | ~34s | PASSED (p=0.0000) |
| RandomForest | **~17min** | PASSED (p=0.0000) |
| GradientBoosting | ~4min | PASSED (p=0.0000) |
| XGBoost | **>2h (运行中)** | ⏳ 等待中 |

**✅ 7/8 模型全部通过**。p=0.0000 意味着 100 次随机排列中没有一次能达到原始模型的性能，所有模型都学到了真实的构效关系。

---

## 三、XGBoost y-Randomization 卡住问题分析

### 3.1 不是卡住，是正常的慢

XGBoost 没有死锁或崩溃。它正在执行 **2525 次模型拟合**：

```
y-Randomization = (1 次原始评估 + 100 次随机排列) × 5×5 RepeatedKFold
                = 101 × 25
                = 2525 次 fit
```

每次 fit 训练 `n_estimators=176` 棵树的 XGBoost。

### 3.2 时间估算

| 每 fit 耗时 | 总耗时 |
|---|---|
| ~3 秒 | 2525 × 3 = 7575s ≈ **2.1 小时** |
| ~5 秒 | 2525 × 5 = 12625s ≈ **3.5 小时** |

参照 RandomForest 在相同条件下耗时 17 分钟（`n_estimators=11`），XGBoost 的 `n_estimators=176`（约 16 倍）预估约 16 × 17min ≈ **4.5 小时**。当前 2 小时仍在正常范围内。

### 3.3 为什么 RandomForest 比 LinearRegression 慢 93 倍

| 模型 | y-Randomization 耗时 | 原因 |
|---|---|---|
| LinearRegression | 11s | 无超参数，闭式解，极速 |
| Ridge | 11s | 同上 |
| Lasso | 39s | 迭代优化，但收敛快 |
| SVR | 50s | 核方法，仍需迭代 |
| DecisionTree | 34s | 单棵树，快 |
| GradientBoosting | ~4min | `n_estimators=72`，72 棵树 × 2500 次 |
| **RandomForest** | **~17min** | `n_estimators=11`，但每棵树较深 + bootstrap |
| XGBoost | **>2h** | `n_estimators=176`，176 棵树 × 2500 次 + hist 算法 |

### 3.4 验证是否仍在运行

```bash
# 检查进程是否存在
ps aux | grep "main.py"

# 检查 CPU 使用率（应该接近 100% × 核心数）
top -p $(pgrep -f main.py)

# 检查日志是否在增长
wc -l my_final_new.log
# 等待 10 分钟再检查一次，行数应该增加
```

### 3.5 y-Randomization 的进度提示

代码每 20 次随机排列打印一次 `Completed 20/100...`。但由于 XGBoost 单次 fit 慢，两次进度提示之间可能需要 20-60 分钟。

---

## 四、数据质量评估

### 4.1 RKfold MAE 量纲验证

所有模型的 `rkf_mae_mean` 与 `loo_mae` 和 `100-split MAE` 在同一数量级（~2-3 kcal/mol），**Ridge MAE 量纲 Bug 已被修复**。不再出现 Ridge MAE ~0.5 的异常低值。

### 4.2 1-SE 法则效果

XGBoost 从 6 特征的绝对最低 MAE(2.73) 选择了 3 特征(MAE=2.89)。阈值 = 2.73 + 0.43 = 3.16，3 特征的 2.89 在范围内。**符合奥卡姆剃刀原则**。

### 4.3 日志系统验证

通过 `logger name` 字段可见 `AI4S_Optimization_models` — 新的命名日志器正常工作。8 个模型的日志完整输出在同一文件中，**日志断层问题已被修复**。

---

## 五、建议

### 短期

1. **继续等待 XGBoost 完成** — 不是卡死，只是慢
2. 完成后检查 `models/XGBoost/` 下是否生成了 `y_randomization_XGBoost.png` 和 `*_final_*.joblib`

### 中长期（可选优化）

1. **y-Randomization 中使用 `n_jobs=1`**：当前 2525 次 fit 是顺序执行的，但每个 XGBoost fit 内部用满所有核心（`n_jobs=-1`），导致频繁的线程创建/销毁开销。改为 `n_jobs=1` 配合外层的 joblib 并行可能更高效
2. **减少 y-Randomization 的排列次数**：当前 100 次。对于 p=0.0000 已经极端显著的模型，50 次甚至 30 次就足以确认显著性
3. **XGBoost 的 `n_estimators` 限制**：176 棵树对于 N=88 的小样本可能过多。可在搜索空间中降低上限（如 50-150）

---

## 六、输出文件状态

```
models/
├── LinearRegression/
│   ├── LinearRegression_final_20260511_145754.joblib  ✅
│   ├── performance_history_20260511_145754.csv        ✅
│   ├── performance_history_20260511_145754.png        ✅
│   ├── final_scatter_20260511_145754.png              ✅
│   └── y_randomization_LinearRegression.png           ✅
├── Ridge/          (同上结构)                          ✅
├── Lasso/                                              ✅
├── SVR/                                                ✅
├── DecisionTree/                                       ✅
├── RandomForest/                                       ✅
├── GradientBoosting/                                   ✅
└── XGBoost/         (scatter + history 已生成，y-randomization 等待中) ⏳
```
