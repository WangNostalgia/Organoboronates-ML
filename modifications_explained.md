# 代码修改说明文档 (Modifications Explained)

> **修改日期:** 2026-05-10
> **范围:** `src/` 下的核心训练管线
> **依据:** `model_analysis_report.md` 分析结论 + `methods_explanation.md` 方法论

---

## 修改总览

| 修改项 | 涉及文件 | 类型 | 影响 |
|---|---|---|---|
| SVR epsilon 范围缩小 | `src/train_and_evaluate.py` | 参数修改 | SVR 拟合精度提升 |
| SVR gamma 范围缩小 | `src/train_and_evaluate.py` | 参数修改 | 排除无意义的低 gamma 区域 |
| RidgeCV / LassoCV 自动选 α | `src/train_and_evaluate.py` | 逻辑替换 | Ridge 跳过 Optuna；Lasso α 由 CV 自动确定 |
| 5×5 RepeatedKFold 评估函数 | `src/train_and_evaluate.py` (新增) | 功能新增 | 每轮迭代输出主要评估指标 |
| 返回值扩展 (rkf_results) | `src/train_and_evaluate.py`, `src/hyperparameter_optimization_and_training.py`, `src/feature_filter.py` | 接口变更 | 沿调用链传递 RepeatedKFold 结果 |
| SHAP-RFECV 混合特征筛选 | `src/feature_selection.py` (新增), `src/iterative_optimization.py` | 逻辑新增 | 特征 ≥ 10 用原方法，< 10 用 SHAP-RFECV |
| y-Randomization 检验 | `src/y_randomization.py` (新建), `src/iterative_optimization.py` | 模块新增 | 每个模型完成后自动执行 |
| LOOCV 同时返回 MAE | `src/leave_one_out_validation.py`, 4个调用点 | 接口变更 | LOOCV 数据完整（R² + MAE） |
| 双重 CV 输出表 | `src/iterative_optimization.py` | 输出增强 | 每轮迭代对比 RepeatedKFold + LOOCV + legacy |
| 性能历史图重设计 | `src/iterative_optimization.py` | 输出增强 | 按指标类型分组（MAE面板 + R²面板） |
| 性能历史扩展 | `src/iterative_optimization.py` | 输出增强 | CSV 新增 rkf_*/loo_* 列 |
| main.py custom_min_features | `main.py` | 文档更新 | 示例值更新，SHAP-RFECV 说明 | |

---

## 逐项说明

### 修改 1：SVR epsilon 范围缩小

**位置:** `src/train_and_evaluate.py:82`

**修改前:**
```python
"epsilon": trial.suggest_float("epsilon", 1e-2, 10, log=True),
```

**修改后:**
```python
"epsilon": trial.suggest_float("epsilon", 1e-3, 0.5, log=True),
```

**目的:** 上次训练得出的最优 ε = 1.41 kcal/mol（在 1e-2 ~ 10 范围内搜索），过大导致 SVR 对 1.41 kcal/mol 以内的误差完全不敏感。化学精度要求通常 < 1 kcal/mol。缩小搜索范围到 0.001 ~ 0.5，使模型对预测误差更敏感，拟合更精确。

---

### 修改 2：RidgeCV / LassoCV 自动选择 α

**位置:** `src/train_and_evaluate.py:55-64` (参数定义), `src/train_and_evaluate.py:227-275` (模型训练), `src/train_and_evaluate.py:316-340` (Optuna 分支)

**修改内容:**

1. **Ridge 参数:** 移除 `trial.suggest_float("alpha", ...)`，改为空 `params = {}`
2. **Lasso 参数:** 移除 `trial.suggest_float("alpha", ...)`，保留 `tol` 由 Optuna 优化
3. **目标函数内部:** Ridge/Lasso 使用 `RidgeCV`/`LassoCV` 内置 LOO CV 自动选择最优 α
4. **Optuna 分支:** Ridge 跳过 Optuna 优化（无参数可调），直接调用 RidgeCV 获取最优 α

**RidgeCV 配置:**
```python
ridge_cv = RidgeCV(
    alphas=np.logspace(-3, 3, 50),      # 50 个候选 α: 0.001 ~ 1000
    cv=LeaveOneOut(),                     # LOO CV
    scoring='neg_mean_absolute_error'     # 以 MAE 为优化目标
)
```

**LassoCV 配置:**
```python
lasso_cv = LassoCV(
    alphas=np.logspace(-4, 1, 50),       # 50 个候选 α: 0.0001 ~ 10
    cv=LeaveOneOut(),
    max_iter=10000,
    tol=<Optuna 优化的 tol>,
    selection='random', random_state=42
)
```

**目的:** Ridge/Lasso 只有 α 一个关键超参数。传统做法是用 Optuna 搜索 α，但 RidgeCV/LassoCV 内置的 CV 搜索更高效精确，避免了 Optuna 的随机搜索开销。

---

### 修改 3：5×5 RepeatedKFold 作为主要评估方法

**位置:**
- `src/train_and_evaluate.py:22-73` (新增 `repeated_kfold_evaluate()` 函数)
- `src/train_and_evaluate.py:467-477` (调用并输出)
- `src/hyperparameter_optimization_and_training.py:29-31` (传递返回值)
- `src/iterative_optimization.py:155-156` (接收并记录)

**使用方法:**
```python
from src.train_and_evaluate import repeated_kfold_evaluate

rkf_results = repeated_kfold_evaluate(
    model, X, y, scaler_X, scaler_y,
    n_splits=5, n_repeats=5
)
# Returns: {'rkf_mae_mean': 2.34, 'rkf_mae_std': 0.15, ...}
```

**目的:** 单次 train/test split 和 LOOCV 在小样本（N≈88）下方差大。5×5 RepeatedKFold（25 次评估）提供更低方差的性能估计。输出均值 ± 标准差，标准差直接反映模型稳定性。

**为什么是 5-fold?** 见 `methods_explanation.md` 问题四的详细论证：N=88 时 k=5 使验证集约 18 个样本，恰好满足回归任务稳定评估的最低要求。

---

### 修改 4：完整 SHAP-RFECV 自动特征选择

**位置:**
- `src/feature_selection.py:7-72` (新增 `shap_rfecv_select_worst_feature()` 函数)
- `src/iterative_optimization.py:147-273` (特征选择循环)
- `src/iterative_optimization.py:348-430` (SHAP-RFECV 路径总结 + 自动选择最优特征数)

**工作流程:**

```
全部迭代统一使用 SHAP-RFECV:
  1. 每次迭代: SHAP 重要性 → 移除重要性最低的特征
  2. 特征 < 10 时: 同步记录 CV 指标到 shap_rfecv_path（模型快照 + RKfold MAE/R² + LOOCV R²）
  3. 循环直到 min_features（地板值）

循环结束后:
  4. 打印 SHAP-RFECV 路径总结表（所有 <10 特征数的 CV 指标）
  5. 自动选择: 取 RKfold MAE 最小的特征数作为最优
  6. 使用最优模型进行后续所有步骤（y-randomization、散点图、保存）
```

**自动选择示例输出:**
```
SHAP-RFECV Path Summary for SVR
Features   RKfold MAE     RKfold R²     LOOCV R²
--------------------------------------------------
9          2.3400         0.7700        0.8112
8          2.2800         0.7850        0.8201
7          2.1500         0.8100        0.8450    ← 最优!
6          2.3100         0.7800        0.8180
5          2.5200         0.7400        0.7900
4          2.8800         0.6700        0.7500
3          3.1500         0.6100        0.6900
--------------------------------------------------
★ Auto-selected optimal: 7 features (RKfold MAE=2.1500)
```

**min_features 的角色变化:** 原来是硬停止点（最终模型就是 min_features 个特征）；现在是**地板值**（循环在此停止，但自动选择可能回溯到更优的中间点）。建议设为 2-3 以让自动选择探索完整路径。

**未使用的旧函数:** `feature_importance_analysis` 和 `feature_correlation_analysis` 的 import 已从 `iterative_optimization.py` 中移除（不再需要）。函数本身保留在各自模块中，以备其他地方使用。

---

### 修改 5：y-Randomization 检验

**位置:**
- `src/y_randomization.py` (新建模块，151 行)
- `src/iterative_optimization.py:427-445` (调用)

**触发时机:** 每个模型完成迭代特征筛选、训出最终模型后立即执行。

**检验流程:**
```
1. 固定最终特征子集和最佳超参数
2. 计算原始模型的 5×5 RepeatedKFold 性能
3. 重复 100 次:
    随机打乱 y
    用相同超参数和特征重新训练
    用 5×5 RepeatedKFold 评估
    记录 MAE 和 R²
4. 对比原始性能 vs 随机分布
5. p-value = 随机模型 MAE ≤ 原始 MAE 的比例
6. p < 0.05 → 通过检验（模型学到了真实关系）
```

**输出文件:**
- `models/y_randomization_{ModelName}.png` — 分布直方图（红色虚线 = 原始模型性能）
- 控制台输出 p-value 和通过/失败判断

**目的:** 验证模型的性能不是来自偶然相关性。这是 QSAR/QSPR 建模的标准验证步骤。

---

### 修改 6：双重 CV 输出表

**位置:** `src/iterative_optimization.py:189-206`

**每轮迭代输出格式:**
```
================================================================================
  Iteration 1 — Dual CV Comparison
  Metric                      Value
  -----------------------------------
  5×5 RKfold MAE            2.3400 ± 0.1500  ← PRIMARY
  5×5 RKfold R²             0.7700 ± 0.0400  ← PRIMARY
  LOOCV R²                  0.8112           ← auxiliary
  100-split MAE              2.3100           ← legacy
  Test R² (single)           0.8007           ← legacy
================================================================================
```

**Decision hierarchy:**
- **模型选择** → 主要依据 5×5 RepeatedKFold 的 MAE 均值
- **模型稳定性** → 主要依据 5×5 RepeatedKFold 的 MAE 标准差
- **特征选择** → 主要依据 RepeatedKFold MAE 变化
- **文献对比** → 参考 LOOCV R²

---

### 修改 7：SVR gamma 搜索范围缩小

**位置:** `src/train_and_evaluate.py:141`

**修改前:**
```python
"gamma": trial.suggest_float("gamma", 1e-2, 10, log=True),
```

**修改后:**
```python
"gamma": trial.suggest_float("gamma", 0.1, 10, log=True),
```

**目的:** gamma 控制 RBF 核的"影响半径"。由于特征已通过 MinMaxScaler 缩放到 [0,1] 范围，gamma=0.01 时影响半径 ≈7.07，远超数据范围，RBF 核退化为近似线性核，失去了非线性拟合能力。下限设为 0.1 保证了有意义的非线性搜索空间。

---

### 修改 8：LOOCV 同时返回 MAE

**位置:** `src/leave_one_out_validation.py`（修改），调用点：`src/iterative_optimization.py`（2处）、`src/feature_filter.py`（1处）、`src/visualization.py`（1处）

**修改前:** 函数只返回 `r2_loo`（单一 float）
**修改后:** 函数返回 `(r2_loo, mae_loo)` 元组

**目的:** 原来的 `leave_one_out_validation()` 只计算 R²，导致 `performance_history['loo_mae']` 一直是 `nan`，无法在性能历史图中展示 LOOCV MAE。现在 LOOCV 数据完整：R² 和 MAE 都有。

**附加修正:** 移除了原来不合理的 `np.clip(y_pred, 0, 100)` — 这对活化能（kcal/mol）没有意义，活化能可能是任意正值。

---

### 修改 9：性能历史图布局重新设计

**位置:** `src/iterative_optimization.py` — `plot_performance_history()`

**修改前:** 原始单面板（左轴 MAE，右轴 R²），新双面板混合了 MAE 和 R² 在同一面板上。

**修改后:** 按指标**类型**分组的双面板布局：

```
┌─────────────────────────────────────────┐
│  Top panel: ALL MAE metrics together     │
│  ├─ 5×5 RKfold MAE ± std (蓝, PRIMARY)  │
│  ├─ LOOCV MAE           (绿, auxiliary)  │
│  └─ 100-split avg MAE   (灰虚线, legacy) │
├─────────────────────────────────────────┤
│  Bottom panel: ALL R² metrics together   │
│  ├─ 5×5 RKfold R² ± std (红, PRIMARY)  │
│  ├─ LOOCV R²            (橙, auxiliary) │
│  └─ Test R² single split(灰虚线, legacy) │
└─────────────────────────────────────────┘
```

**设计原则:** 同类型指标（MAE vs MAE、R² vs R²）放在同一面板，直观对比 RepeatedKFold / LOOCV / legacy 之间的差异。每个特征移除点标注在被移除特征名和剩余特征数。

**向后兼容:** 如果 performance_history 中没有新列（旧格式 CSV），自动回退到 legacy 单面板布局。

---

### 修改 10：main.py 的 custom_min_features 示例更新

**位置:** `main.py:89-116`

**修改内容:** 更新了 `custom_min_features` 的注释文档和示例值。

**关键变更:**
- 添加了 SHAP-RFECV 切换阈值（特征 < 10）的说明
- 示例值从 3-6 调整为 **7-8**，因为之前训练中所有模型降至 3 个特征时 MAE 系统性退化
- 添加了 Ridge（RidgeCV，跳过 Optuna）和 Lasso（LassoCV）的简要说明
- 添加了 SVR 参数搜索范围的说明

**使用方法:** 取消注释 `custom_model_min_features` 字典并根据需要调整每个模型的最小特征数。

---

## 输出结果解读

### 如何判断模型优劣

1. **首选指标**: 5×5 RepeatedKFold MAE 均值（越低越好）+ 标准差（越小越稳定）
2. **次选指标**: 5×5 RepeatedKFold R² 均值（越高越好）
3. **参考指标**: LOOCV R²（与传统 QSAR 文献对比）
4. **警告信号**: RepeatedKFold 与 LOOCV 差异过大 → 模型不稳定

### 如何阅读 SHAP-RFECV 输出

当特征数降至 < 10 时，日志会显示 `=== SHAP-RFECV Mode ===`：
```
SHAP Importance Ranking:
  [1] VBur_C: 35.2%
  [2] lumo_energy: 28.1%
  ...
  → SHAP-RFECV selected for removal: 'homo_energy' (importance: 0.06%)
```
重要性最小的特征被移除。注意：尽管某个特征重要性看似很低（如 0.06%），在特征数少时仍需谨慎移除。

### 如何阅读 y-Randomization 结果

```
p-value (MAE): 0.0100
✓ PASSED — model learned genuine SAR (p=0.0100 < 0.05)
```
- p < 0.05: 模型性能显著优于随机基线 → 可信
- p 0.05-0.10: 边缘状态 → 需更多数据验证
- p ≥ 0.10: 模型可能学到的是偶然模式 → 不应使用

---

## 注意事项

### 运行时间影响

| 修改项 | 对运行时间的影响 |
|---|---|
| SVR epsilon 缩小 | 无影响（仅改变搜索范围） |
| RidgeCV/LassoCV | Ridge 大幅加速（跳过 Optuna）；Lasso 轻微加速 |
| 5×5 RepeatedKFold | 每轮增加约 25 次模型拟合（比原来仅 100-split 多出 25 次） |
| SHAP-RFECV | 特征 < 10 时每轮增加一次完整 SHAP 计算（KernelExplainer 较慢） |
| y-Randomization | 最后增加 100 × 25 = 2500 次模型拟合（每个模型） |

**总体预估**: 单模型完整运行时间可能增加 30-50%。

### 依赖包

所有修改使用标准依赖（scikit-learn, numpy, matplotlib, shap, optuna），无需安装新包。

### 向后兼容性

- `train_and_evaluate()` 返回值从 3 个变为 4 个（新增 `rkf_results`）
- `hyperparameter_optimization_and_training()` 返回值从 8 个变为 9 个
- `feature_selection()` 函数签名未变，新增了独立的 `shap_rfecv_select_worst_feature()` 函数
- 旧 `.joblib` 文件仍然可加载（model_info 字典结构未变）

---

## 未解决的问题 / 待审核项

1. **Lasso tol 优化策略**: 经分析（见 `methods_explanation.md` 问题五），tol 的边际收益极低。**用户已决定保留 Optuna 优化 tol**（当前做法不变）。

2. **~~SVR gamma 搜索范围~~**: 已修改为 `0.1 ~ 10`。

3. **~~LOOCV MAE 缺失~~**: 已修改 `leave_one_out_validation()` 同时返回 R² 和 MAE。

4. **~~性能历史图 MAE/R² 混排~~**: 已重设计为按指标类型分组（MAE 面板 + R² 面板）。

5. **~~main.py custom_min_features~~**: 已更新示例值和注释文档。

6. **y-Randomization 的 model_class 参数**: 在 `iterative_optimization.py` 中，`model_class` 来自 `models` 字典的 values（即类引用），传递到 `y_randomization_test()` 是安全的。但如果用户自定义了 `models` 字典，需要确保其中是类而非实例。

7. **100-split MAE 与单次划分 R² 不一致**: legacy 指标来自不同计算路径（MAE 是 100-split 平均，R² 是单次划分）。不影响决策（现在主要依赖 RepeatedKFold），但可选修复：在 `train_and_evaluate()` 的 100-split 循环中同时记录 R²。

8. **SHAP-RFECV 路径仅从 <10 特征开始记录**: 如果最优特征数恰好 ≥10（虽然不太可能），自动选择将无法发现。如果经常出现这种情况，可以将记录阈值上调到更大的值。当前设计基于分析报告结论：最优特征数通常出现在 5-9 之间。

---

## Checkpoint 使用指南

训练完成后，`models/<ModelName>/` 目录下保存了两种 `.joblib` 文件。以下是如何在**不修改源代码**的情况下，手动检查、对比和使用任意 checkpoint。

详细代码见 `example/load_checkpoint_guide.py`。

### 文件类型

| 文件模式 | 内容 | 用途 |
|---|---|---|
| `*_iteration_N_<ts>.joblib` | 第 N 轮迭代的模型快照（含特征集、全部指标） | 手动回溯到任意特征数 |
| `*_final_<ts>.joblib` | 自动选择 / `--force_n_features` 指定的最终模型 | 正式部署预测 |

### 常见操作

#### 列出所有 checkpoint
```python
import joblib, glob
for f in sorted(glob.glob('models/SVR/*_iteration_*.joblib')):
    info = joblib.load(f)
    print(f"{f[-40:]:<45s} features={len(info['features']):2d}")
```

#### 加载自动选的最终模型
```python
info = joblib.load('models/SVR/SVR_final_20260510_120000.joblib')
model, features = info['model'], info['features']
# 预测: model.predict(scaler_X.transform(X_new[features]))
```

#### 手工选择特定特征数的 checkpoint
```python
def load_by_feature_count(model_dir, target_n):
    """从所有迭代 checkpoint 中找到恰好 target_n 个特征且 MAE 最低的那个"""
    import glob, joblib
    candidates = []
    for f in glob.glob(f'{model_dir}/*_iteration_*.joblib'):
        info = joblib.load(f)
        if len(info['features']) == target_n:
            candidates.append((info['metrics']['mae_mean'], f, info))
    if not candidates:
        raise ValueError(f"No checkpoint with {target_n} features")
    candidates.sort()
    return candidates[0][2]  # 返回 MAE 最低的那个
```

#### 并排对比两个特征数的指标
```python
# 对比 7 特征 vs 5 特征：哪个综合更好？
compare_checkpoints('models/SVR', 7, 5)
# 输出：每个指标的并排比较 + 差异特征
```

### 典型决策流程

```
SHAP-RFECV Path Summary (控制台输出)
         │
         ▼
  ┌─ 同意自动选择？ → 直接用 *_final_*.joblib
  │
  └─ 不同意 → 1. 看 Path Summary 里哪个特征数的综合表现更好
             2. compare_checkpoints('models/SVR', auto_n, your_n)
             3. 确认后用 load_by_feature_count('models/SVR', your_n)
             4. 如需重跑整个流程: --force_n_features your_n
```

---

## 变更文件清单

```
修改的文件:
  src/train_and_evaluate.py              (SVR ε + SVR gamma + RidgeCV/LassoCV + RepeatedKFold)
  src/hyperparameter_optimization_and_training.py  (返回值扩展: 8→9)
  src/iterative_optimization.py          (SHAP-RFECV + y-randomization + 双重CV输出表 + 性能历史图重设计)
  src/feature_selection.py               (SHAP-RFECV 函数: shap_rfecv_select_worst_feature)
  src/feature_filter.py                  (返回值解包更新)
  src/leave_one_out_validation.py        (返回值扩展: R²→(R², MAE), 移除无意义的 np.clip)
  src/visualization.py                   (返回值解包更新)
  main.py                                (custom_min_features 示例值和注释更新)

新增的文件:
  src/y_randomization.py                 (y-randomization 检验模块)

文档更新:
  README.md                              (双重CV体系 + 混合特征筛选 + y-randomization)
  README_CN.md                           (同上，中文版)
  pipeline.md                            (Stage 2-5 更新: SVR/Ridge/Lasso/双CV/RFECV/y-randomization)
  modifications_explained.md             (本文件)
  methods_explanation.md                 (问题四: k-fold选择, 问题五: 线性模型参数详解)
```
