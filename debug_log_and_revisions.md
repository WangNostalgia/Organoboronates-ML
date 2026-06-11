# Debug Log & Revisions Report

> **日期:** 2026-05-10
> **来源运行:** `my_final_new_2.log`（2026-05-10 运行）
> **依据:** `modifications_explained.md` + 用户提出的 5 个疑惑

---

## 1. 实际运行结果与预期对比

对照 `modifications_explained.md` 中描述的每项改动，检查其是否成功实现。

| 修改项 | 预期表现 | 实际表现 | 状态 |
|---|---|---|---|
| SVR epsilon (0.001–0.5) | 搜索范围缩小 | ✅ 代码中已修改 | PASS |
| SVR gamma (0.1–10) | 搜索范围缩小 | ✅ 代码中已修改 | PASS |
| RidgeCV auto-α | 跳过 Optuna，自动选 α | ✅ 日志显示 Ridge 直接使用 RidgeCV（无 Optuna 迭代） | PASS |
| LassoCV auto-α | LassoCV 选 α，Optuna 只调 tol | ✅ 日志显示 LassoCV 路径 | PASS |
| 5×5 RepeatedKFold | 输出 MAE ± std, R² ± std | ⚠️ 输出存在但 **MAE 数值错误**（缩放问题） | BUG FIXED |
| SHAP-RFECV 特征选择 | 全部迭代使用 SHAP | ✅ 日志显示 `=== SHAP-RFECV` | PASS |
| SHAP-RFECV 自动选择 | 路径 <10 特征记录并自动选最优 | ✅ Path Summary 输出，找到 optimal | PASS |
| y-Randomization | 最优模型后执行 | ✅ 日志显示 `y-Randomization Test` | PASS |
| LOOCV 返回 MAE | R² + MAE 同时输出 | ✅ 日志中 LOOCV MAE 有数值 | PASS |
| 性能历史图 | 双面板 MAE+R² | ✅ 图表已生成 | PASS |
| **高相关处理** | SHAP-RFECV 中无共线性预筛 | ❌ 缺失：高相关特征对 (>0.8) 未经预处理 | FIXED |
| **日志重复输出** | 每迭代一次清晰表格 | ❌ 特征 <10 时出现双重输出（Dual CV 表格 + RFECV path 详细列表） | FIXED |
| **Metrics 文件不完整** | 迭代和最终 txt 含全部双CV数据 | ❌ 仅含 3-5 个基本指标 | FIXED |
| **rkf_mae_mean 异常偏低** | 与 loo_mae 同一尺度 | ❌ 0.37 vs 2.6 kcal/mol —— 约差 7 倍 | BUG FIXED |
| **rkf_r2_std 过大** | 标准差在合理范围 | ⚠️ 统计上N=88下正常，但需说明 | EXPLAINED |

---

## 2. 疑惑解答与修正详情

### 疑惑①：SHAP-RFECV 中缺少高相关特征处理

**原因分析:**
之前重构为"全部迭代使用 SHAP-RFECV"时，直接移除了 `feature_selection()` 函数调用（该函数有高相关预筛逻辑），而 `shap_rfecv_select_worst_feature()` 没有继承这个逻辑。导致高相关特征对（如 |r| > 0.8）不再被优先处理。

**修改位置:** `src/feature_selection.py:7-106` — 重写 `shap_rfecv_select_worst_feature()`

**修改内容:**
新增两阶段选择逻辑：

```
Phase 1 — 高相关预筛:
  计算 Pearson 相关系数矩阵
  for 每对特征:
    if |r| > 0.8:
      取重要性较低的那个作为候选
  如果存在候选 → 移除候选中最不重要的那个

Phase 2 — 低重要性:
  如果没有高相关对 → 移除全局 SHAP 重要性最低的特征
```

**修改前后对比:**

```python
# 修改前: 函数签名
def shap_rfecv_select_worst_feature(model, X, scaler_X, model_name):
    ...
    return worst_feature, ranking  # 2 个返回值

# 修改后: 函数签名
def shap_rfecv_select_worst_feature(model, X, scaler_X, model_name, corr_threshold=0.8):
    ...
    return worst_feature, ranking, removal_reason  # 3 个返回值
```

**影响评估:**
- 高相关特征对被优先处理 → 减少共线性对 SHAP 归因的干扰
- 每轮计算一次相关系数矩阵（O(n²)，n 为特征数）——在 n<15 的范围内开销可忽略
- 如果用户想禁用高相关预筛，设 `corr_threshold=1.0` 即可

---

### 疑惑②：日志中双 CV 结果重复输出

**原因分析:**
当特征 < 10 时，代码执行路径依次触发：
1. `Dual CV Comparison` 表格（5 个指标）
2. `RFECV path[N features]` 详细列表（7 个指标，与上表高度重叠）

两者在日志中相隔仅几行，造成信息冗余。

**修改位置:** `src/iterative_optimization.py:202-215`（表格扩展）、`src/iterative_optimization.py:268-273`（RFECV path 精简）

**修改内容:**
- **Dual CV Comparison 表格**：从 5 个指标扩展到全部 7 个指标（新增 LOOCV MAE 和 Test MAE），成为唯一的迭代指标输出
- **RFECV path 记录日志**：从 7 行详细列表精简为 1 行确认信息

**修改后日志示例:**
```
================================================================================
  Iteration 6 — Dual CV Comparison
  Metric                       Value
  ----------------------------------------
  5×5 RKfold MAE           2.3400 ± 0.1500  ← PRIMARY
  5×5 RKfold R²            0.7700 ± 0.0400  ← PRIMARY
  LOOCV R²                 0.8112        ← auxiliary
  LOOCV MAE                2.1800        ← auxiliary
  100-split MAE             2.3100        ← legacy
  Test R² (single)          0.8007        ← legacy
  Test MAE (single)         2.3700        ← legacy
================================================================================
  ✓ RFECV path recorded [9 features] — RKfold MAE=2.3400 (see table above for full metrics)
```

---

### 疑惑③：迭代和最终 metrics 文件内容不完整

**原因分析:**
`_iteration_N_<ts>_metrics.txt` 和 `_final_<ts>_metrics.txt` 的写入逻辑在本次修改后未同步更新，仍只输出基础指标（MAE Mean, R² Test, MAE Test）。

**修改位置:**
- `src/iterative_optimization.py` — 迭代 checkpoint 的 `_metrics.txt` 写入（约 line 319）
- `src/iterative_optimization.py` — 最终模型的 `_metrics.txt` 写入（约 line 430）

**修改后迭代 metrics 格式:**
```
Model: SVR
Iteration: 6
Features (9): reaction_energy, lumo_energy, VBur_C, pka, ...
Removed Features: homo_energy, electronegativity, ...

--- PRIMARY (5x5 RepeatedKFold) ---
RKfold MAE: 2.3400 +/- 0.1500
RKfold R²:  0.7700 +/- 0.0400

--- AUXILIARY (LOOCV) ---
LOO R²:  0.8112
LOO MAE: 2.1800

--- LEGACY ---
100-split MAE: 2.3100
Test R²:       0.8007
Test MAE:      2.3700

Best Parameters: {'C': 73.31, 'epsilon': 0.017, 'gamma': 8.04}
```

---

### 疑惑④：`rkf_mae_mean` 异常偏低 —— **关键 BUG**

**原因分析（根因）:**

`repeated_kfold_evaluate()` 使用 `cross_validate()` 搭配 `scoring='neg_mean_absolute_error'` 计算 MAE。问题在于：
- `scaler_y` 是 `StandardScaler()` 拟合的（`train_and_evaluate.py` line 346）
- `y_scaled` 经 StandardScaler 变换后均值为 0、标准差为 1
- `cross_validate` 在 `y_scaled` 上计算 MAE → 单位为"标准差"
- 原始活化能标准差约 6–7 kcal/mol
- 所以 `rkf_mae_mean ≈ 0.37` 相当于 `0.37 × 7 ≈ 2.6 kcal/mol`

**证据链:**
```
日志: Test MAE (真实尺度)    = 2.90 kcal/mol  ← 合理
日志: 100-split MAE (真实)   = 2.71 kcal/mol  ← 合理  
日志: 5×5 RKfold MAE         = 0.38           ← 约 7 倍偏差
CSV:  loo_mae                 = 3.31 kcal/mol  ← 合理
CSV:  rkf_mae_mean            = 0.43           ← 约 7-8 倍偏差
```

**修改位置:** `src/train_and_evaluate.py:45-74` — 重写 `repeated_kfold_evaluate()`

**修改方案:**
放弃 `cross_validate` + 内置 scorer，改为手动遍历每个 fold/repeat。在**逆变换回原始单位**后再计算 MAE：

```python
for train_idx, test_idx in rkf.split(X_scaled):
    # 每个 fold 独立做 target scaling
    fold_scaler_y = MinMaxScaler(feature_range=(0, 100))
    y_tr_s = fold_scaler_y.fit_transform(y_tr_orig).ravel()
    y_te_s = fold_scaler_y.transform(y_te_orig).ravel()

    m.fit(X_tr, y_tr_s)
    y_pred_orig = fold_scaler_y.inverse_transform(y_pred_s)  # ← 逆变换！

    mae = mean_absolute_error(y_te_orig, y_pred_orig)  # ← 在原始尺度上计算
```

**R² 不受影响:** R² 是尺度无关的归一化指标，修正前 R² 值已是正确的。

**预期效果:**
| 指标 | 修正前 (典型值) | 修正后 (预期) |
|---|---|---|
| rkf_mae_mean | ~0.37 | ~2.6 (与 loo_mae 同尺度) |
| rkf_mae_std  | ~0.05 | ~0.3-0.5 (与 MAE 的变异一致) |

---

### 疑惑⑤：`rkf_r2_std` 过大是否正常？

**原因分析:**

从 Performance History CSV 数据看（以 GradientBoosting 为例）：
```
rkf_r2_mean ≈ 0.76
rkf_r2_std  ≈ 0.11–0.13
```

`rkf_r2_std` 相对于 `rkf_r2_mean` 看起来很大（约 15%），但这**在统计上是正常的**，原因如下：

**1. 小样本效应 (N≈88):**
- 5-fold 使每折验证集约 18 个样本
- 在 18 个样本上计算 R²，其抽样方差天然较大
- 25 次评估（5×5）的标准差 0.11 意味着 95% CI 约 ±0.22 —— 在 R² 尺度上并不异常

**2. 修正 MAE 后 R² 判断力增强:**
- 修复 MAE 缩放问题后，用户可以同时参考 MAE（更稳定、更直观）和 R²（归一化、可跨数据集比较）
- 报告 R² std 实际上增加了透明度 —— 它诚实反映了模型在 25 个不同数据划分下的波动

**3. Lasso 的特殊情况:**
Lasso 的 `rkf_r2_mean = -0.05` 且所有迭代值完全相同，说明 Lasso 未正确收敛（alpha 路径问题，与 LassoCV 实现有关）。这是独立问题，建议单独排查。非 Lasso 模型未见此现象。

**建议:**
- **不需要修改代码** —— R² 的标准差计算正确
- 在最终报告中将 R² 呈现为 `mean ± std`（如 `0.76 ± 0.11`），并注明 25 次评估来源
- 考虑在性能历史图中用箱线图或误差带展示 R² 分布（当前已用 ±std 误差带）

---

### 追加修复⑥：Lasso alpha 丢失导致指标恒定

**诊断过程:**

执行了 `example/diagnose_lasso.py` 对 LassoCV 的选择行为进行独立测试。诊断结果：

```
LassoCV 选择 alpha = 0.0569 (在候选列表中间位置，非边界值)
非零系数: 8 / 15
Alpha 路径的 CV MSE 在 87 ~ 497 之间变化 → LassoCV 工作正常
```

LassoCV 本身没有问题。问题出在 `train_and_evaluate.py` 的 **alpha 传递链**。

**根因:**

1. `objective()` 函数中 LassoCV 正确选择了 alpha 并通过 `trial.set_user_attr("best_alpha", ...)` 存储
2. 但 Optuna 优化结束后，`study.best_params` **只包含 Optuna 调优的参数**（即 `tol`），不包含 user_attr
3. `best_model = Lasso(**best_params)` 创建模型时 **没有传入 alpha** → 使用 scikit-learn 默认值 `alpha=1.0`
4. `alpha=1.0` 是 LassoCV 选出的最优值 (0.057) 的约 **18 倍**，导致所有系数被极度压缩 → 模型退化

**证据:**
```
LassoCV 最优 alpha:  0.0569  (8 个非零系数，合理)
实际使用的 alpha:     1.0     (scikit-learn 默认值，严重过正则化)
→ 预测近似常数 → MAE 恒定 ~5.68，R² 恒定 ~-0.05
→ 特征变化不影响预测（因为系数全被压到 0 附近）
```

**修改位置:** `src/train_and_evaluate.py:403-420`

**修改内容:** Optuna 优化结束后，从 best_trial 的 user_attrs 中提取 LassoCV 选择的 alpha 并加入 best_params：

```python
if model_class == Lasso:
    best_alpha = study.best_trial.user_attrs.get("best_alpha", None)
    if best_alpha is not None:
        best_params = dict(best_params)
        best_params["alpha"] = best_alpha
```

### 追加修复⑦：SHAP-RFECV Path Summary 增加标准差列

**修改位置:** `src/iterative_optimization.py` — 控制台输出和 metrics.txt

**修改内容:** RKfold MAE 和 RKfold R² 列从只显示均值改为 `均值 ± 标准差` 格式：

```
修改前:  RKfold MAE       RKfold R²
         2.3400±0.150     0.7700±0.040

修改后:  RKfold MAE ± std         RKfold R² ± std
         2.3400 ± 0.1500          0.7700 ± 0.0400
```

---

## 3. 修正后的代码使用指导

### 命令行参数（无变化）

```bash
nohup uv run python main.py --n_trials 100 --mae_threshold 2.0 --min_features 3 --n_jobs -1 > train.log 2>&1 &
```

### 关键改进

| 改进 | 用户可见变化 |
|---|---|
| MAE 修复 | `rkf_mae_mean` 从 ~0.4 变为 ~2.5（与 loo_mae 同尺度） |
| 高相关处理 | 日志新增 `removal_reason: high_correlation` 或 `low_importance` |
| 日志去重 | 每迭代仅一个统一表格，RFECV 路径记录精简为一行 |
| Metrics 文件 | 迭代和最终 txt 均包含 PRIMARY/AUXILIARY/LEGACY 三组完整指标 |

### 函数签名变化

```python
# shap_rfecv_select_worst_feature 返回值变化
worst_feat, ranking, removal_reason = shap_rfecv_select_worst_feature(...)
# 新增第三个返回值: 'high_correlation' 或 'low_importance'

# repeated_kfold_evaluate 内部实现重写（签名不变）
# 返回的 'rkf_mae_mean' 现在在原始 kcal/mol 尺度
```

---

## 4. 遗留问题与建议

1. **~~Lasso 模型异常~~:** 已修复 — 根因是 Optuna `study.best_params` 不包含 LassoCV 内部选择的 alpha，导致默认 alpha=1.0 过度正则化。修复后 alpha 正确传递。

2. **~~R² std 报告方式~~:** 已添加 — SHAP-RFECV Path Summary 的控制台输出和 metrics.txt 中，RKfold MAE 和 R² 列均以 `均值 ± 标准差` 格式显示。

3. **DecisionTree 欠拟合:** 用户指示暂不注释，保持现状。

4. **小样本下的 RepeatedKFold:** 用户指示暂不更改，保持 5×5=25 次评估。

5. **Lasso alpha 搜索范围:** 诊断确认当前 `alphas=np.logspace(-4, 1, 50)` 范围合理（最优 alpha=0.057 在中间位置）。无需调整。

---

## 变更文件清单

```
修改的文件:
  src/train_and_evaluate.py              (Issue 4: repeated_kfold_evaluate 重写 — 修复 MAE 缩放)
  src/iterative_optimization.py          (Issue 2: 统一 Dual CV 表格 / Issue 3: 扩展 metrics .txt)
  src/feature_selection.py               (Issue 1: SHAP-RFECV 增加高相关预筛)
```
