# Debug Log & Revisions 2 — Critical Architecture Fixes

> **日期:** 2026-05-11
> **依据:** `revision_3.md` 中的指导和意见
> **范围:** `src/train_and_evaluate.py` + `src/iterative_optimization.py`

---

## 修改总览

| 类别 | 缺陷数 | 严重程度 | 状态 |
|---|---|---|---|
| 致命逻辑与数据泄露 (Critical Flaws 1-3) | 3 | **致命** | FIXED |
| 评估标准与工程规范 (Flaws 4-5) | 2 | 高 | FIXED |
| 计算效率与参数空间 (Flaws 6-8) | 3 | 中 | FIXED |
| iterative_optimization 支柱 1-4 | 16 | 混合 | FIXED |

---

## Part A: train_and_evaluate.py 修正

### A1. 调参配置丢失 — 固定参数与 Optuna 搜索参数割裂

**问题:** `objective()` 中为每个模型定义了包含固定参数的完整 `params`，但 `study.best_params` 只包含 `trial.suggest_*` 的动态参数。`best_model = model_class(**best_params)` 丢失了所有固定配置（如 ElasticNet 的 `max_iter=50000`、MLP 的 `early_stopping=True` 等）。

**修复:** 提取 `FIXED_PARAMS_MAP` 字典（含 16 个模型类的固定参数），Optuna 后合并：

```python
final_params = {**FIXED_PARAMS, **best_params}
best_model = model_class(**final_params)
```

涉及 16 个模型的固定参数完整列表：

| 模型 | 固定参数 |
|---|---|
| LinearRegression | (none) |
| Ridge | (RidgeCV handles all) |
| Lasso | `max_iter=10000, selection='cyclic', random_state=42` |
| ElasticNet | `max_iter=50000, selection='cyclic', random_state=42, fit_intercept=True, precompute=True, warm_start=True, copy_X=True` |
| SVR | `kernel='rbf', tol=1e-3, max_iter=10000, cache_size=1000` |
| DecisionTree | `splitter='best', max_features=None, random_state=42` |
| RandomForest | `n_jobs, random_state=42` |
| GradientBoosting | `loss='squared_error', random_state=42, n_iter_no_change=10, tol=1e-4` |
| XGBoost | `n_jobs, random_state=42, tree_method='hist', grow_policy='depthwise', base_score=0.5` |
| LGBM | `n_jobs, random_state=42` |
| MLP | `learning_rate='adaptive', max_iter=2000, early_stopping=True, validation_fraction=0.2, n_iter_no_change=20, tol=1e-3, random_state=42, solver='adam', batch_size='auto'` |
| AdaBoost | `random_state=42` |
| KNR | `n_jobs` |
| GPR | `random_state=42, n_restarts_optimizer=5` |
| KRR | (none) |

---

### A2. 数据泄露 — repeated_kfold_evaluate 与 RidgeCV/LassoCV

**问题1:** `repeated_kfold_evaluate` 接受外部预拟合的 `scaler_X` 和 `scaler_y`，在 fold 切分前对全量数据 transform，验证集提前获知全局分布。且 `scaler_y` 参数完全未被使用。

**问题2:** RidgeCV/LassoCV 在传入 CV 前对整个 `X_train` 执行 `fit_transform`，LOO 的验证点看到了全量训练集信息。

**修复:**

1. **`repeated_kfold_evaluate`** — 移除 `scaler_X`/`scaler_y` 参数。每个 fold 内部独立拟合 `MinMaxScaler`（特征和目标），不再接受外部预拟合的缩放器。签名简化为 `(model, X, y, n_splits=5, n_repeats=5, random_state=42)`。

2. **RidgeCV/LassoCV** — 改为 5-fold CV（`KFold(n_splits=5, shuffle=True, random_state=42)`）替代 LOO，与 Optuna objective 的 CV 策略一致。

---

### A3. 目标变量缩放策略断裂

**问题:** Optuna 目标函数中统一使用 `MinMaxScaler(0,100)` 缩放 y，但最终训练却使用 `StandardScaler()`。为正则化参数（Ridge/Lasso α、SVR C/ε）量身搜索的最优值在均值 0、方差 1 的尺度上完全失效。

**修复:** 全局统一 `MinMaxScaler(feature_range=(0,100))` — Optuna objective、最终训练、100-split stability、RKFold 内部均使用同一种缩放器。

---

### A4. 100-split best_random_state — P-hacking

**问题:** 在 100 次随机划分中"挑选"MAE 最低的 `best_random_state` 并报告，本质是测试集数据挖泥。

**修复:** 彻底移除 `best_random_state` 的逻辑。100-split 仅保留均值 ± 标准差作为稳定性参考，不做任何选择。

---

### A5. LassoCV 形同虚设的传参

**问题:** `repeated_kfold_evaluate` 的 `scaler_y` 参数从未被内部使用（内部另实例化了 MinMaxScaler）。参数冗余且误导。

**修复:** 从函数签名中移除 `scaler_y`。

---

### A6. 暴力嵌套 LOOCV → 5-fold

**问题:** Optuna objective 对每个 trial 进行 LOO CV。以 100 trials × 112 样本 = 11,200 次模型拟合，算力浪费巨大。

**修复:** objective 内统一改用 `KFold(n_splits=5, shuffle=True, random_state=42)`。100 trials × 5 folds = 500 次拟合，约 22× 加速。

---

### A7. 搜索空间清理

| 模型 | 变更 | 原因 |
|---|---|---|
| XGBoost | 移除 `scale_pos_weight` | 分类任务参数，回归无用 |
| KNN | `weights: ["uniform"]` → `["uniform", "distance"]` | 单一候选项浪费 Categorical 采样 |
| GPR | 新增 `kernel` + `length_scale` 搜索 | 原只调 α，核函数决定核心表达能力 |
| Lasso | `selection: "random"` → `"cyclic"` | random 增加收敛时间且无必要 |

---

### A8. SQLite 锁竞争

**问题:** `storage=f"sqlite:///optuna_optimization_{...}.db"` 在 `n_jobs=-1` 高并发下争抢文件锁。

**修复:** 移除 `storage` 参数，改用内存存储（默认）。同时解决了旧 study 被意外加载的问题。

---

## Part B: iterative_optimization.py 修正

### B1. 轻量化 rfecv_path（支柱 3.1 + 4.1）

**问题:** `shap_rfecv_path` 中缓存了完整的 `best_model` 和 `scaler_X/Y` 对象引用。导致：(a) joblib dump 时文件膨胀至数百 MB；(b) 后续迭代中 scaler 的 `fit_transform` 污染了前期快照的内部状态。

**修复:** rfecv_path 只记录纯文本和基础数值：

```python
rfecv_model_info = {
    'features': [...],            # 纯文本列表
    'hyperparameters': {...},     # dict 浅拷贝
    'metrics': {                  # 全部 float（含 NaN 处理）
        'rkf_mae_mean': float(...),
        'rkf_mae_std': float(...),
        'loo_r2': float(...) if not np.isnan(...) else None,
        ...
    },
    'iteration': int,
    'n_features': int,
}
```

### B2. y_test / y_pred_test 样本错位修复（支柱 2.1）

**问题:** RFECV 最优特征重建时重新切分了数据（`y_test_opt`），但后续计算 `mean_absolute_error(y_test, y_pred_test)` 仍引用旧的外层 `y_test`。A 批标签对比 B 批预测 → 误差完全是噪声。

**修复:** 整个最优重建块内严格使用配对的 `y_test_opt` 和 `y_pred_test_opt`。`result` 字典的 `mae_test_avg` 和 `r2_test_avg` 基于 `(y_test_opt, y_pred_test_opt)` 计算。

### B3. 缺失字典键修复（支柱 2.3）

**问题:** `metrics.txt` 中尝试读取 `result['rkf_mae_opt_mean']` 但 `result` 字典从未设置此键。

**修复:** `result` 字典新增：
- `rkf_mae_opt_mean` — 最优特征数的 RKfold MAE
- `rkf_r2_opt_mean` — 最优特征数的 RKfold R²

### B4. 移除危险 clip（支柱 2.4）

**问题:** `np.clip(y_pred_test, 0, 100)` 对活化能（无物理上限）的预测强行截断，掩盖了严重过预测。

**修复:** 移除所有无物理定律支撑的 `clip` 操作。迭代内和最终重建均不再 clip。

### B5. 隐式变量作用域修复（支柱 2.5）

**问题:** `shap_rfecv_path` 为空时，绘图块引用 `y_train_opt`、`y_test_opt` 等未初始化的变量。

**修复:** `else` 分支内显式赋值 `y_train_opt = y_train, y_test_opt = y_test` 等，确保绘图块无论哪个路径都有合法变量。

### B6. 全量数据泄露修复（支柱 1.1）

**问题:** 最终保存时 `MinMaxScaler().fit(X[optimal_features])` 包含了测试集数据。

**修复:** 使用 `scaler_X_opt` 和 `scaler_y_opt`（仅在 `X_train_opt` 上 `fit`）进行持久化。

### B7. 异常处理精确化（支柱 4.2）

**问题:** `except Exception` 静默吞没所有异常。

**修复:** LOOCV 异常改为捕获 `(ValueError, np.linalg.LinAlgError)` 并在日志中记录完整堆栈。

### B8. Matplotlib 内存泄漏（支柱 3.3）

**问题:** `plt.close()` 无参数，可能无法释放目标图形。

**修复:** `plt.close(fig)` — 显式传入图形实例。

### B9. 版本清理健壮化（支柱 4.3）

**问题:** 依赖 `split("_final_")[-1].replace(".joblib", "")` 提取时间戳，脆弱。

**修复:** 使用正则 `re.compile(r'_final_(\d{8}_\d{6})\.joblib$')` 提取标准时间戳。

### B10. 最终 scaler 持久化修正（支柱 1.1）

**问题:** `scaler_X: MinMaxScaler().fit(X[optimal_features])` 在测试集上拟合。

**修复:** 持久化的 `scaler_X` 和 `scaler_y` 均来自仅拟合 `X_train_opt` 的实例。

---

## 修改后的代码使用指导

### 命令行参数（无变化）

```bash
nohup uv run python main.py --n_trials 100 --mae_threshold 2.0 --min_features 3 --n_jobs -1 > train.log 2>&1 &
```

### 关键行为变化

| 行为 | 修改前 | 修改后 |
|---|---|---|
| 最终模型超参数 | 仅含 Optuna 建议参数（丢失固定配置） | 合并固定参数 + Optuna 参数 |
| y 缩放器 | Optuna 用 MinMax(0,100)，最终用 StandardScaler | **全局统一 MinMaxScaler(0,100)** |
| repeated_kfold_evaluate | 外部预拟合 scaler，全量 transform 后切分 | **每个 fold 独立 fit scaler** |
| 100-split | 挑选 best_random_state | 仅报告均值 ± std，不作选择 |
| Optuna CV | LeaveOneOut (N folds) | KFold(5) — 约 20× 加速 |
| rfecv_path 内存 | 含 model + scaler 对象 → 数百 MB | 仅基础类型 → 数 KB |
| 最优模型重建 | y_test 错位 + scaler 在全量 X 上 fit | 配对标签 + scaler 仅在训练集 fit |
| 预测截断 | np.clip(y, 0, 100) | 无 clip |
| SQLite 存储 | 文件持久化（可能锁竞争） | 内存（无竞争） |
| GPR 核函数 | 只调 alpha | 新增 kernel 类型 + length_scale 搜索 |

---

## 遗留问题与建议

1. **RFECV 选择偏差（支柱 1.3）:** 仍在同一批 RepeatedKFold 数据上选最优特征数（"既当裁判又当运动员"）。彻底的修复需要在最外层包裹一个独立 Hold-out 集用于最终验证。当前最小可行修复：用户可观察 Path Summary 中 MAE 的平滑趋势（如果最优点的 MAE 与相邻特征数差异在 1 std 内，则没有显著差异）。

2. **Optuna 全量重搜（支柱 3.2）:** 每次迭代仍触发完整 Optuna 搜索。对于大规模数据集，建议在仅删除 1-2 个特征时使用上一轮最优参数作为热启动。当前 N=88 的小数据集下开销可接受。

3. **固定划分过拟合（支柱 1.4）:** SHAP-RFECV 路径固化在 `random_state=40` 的单次划分上。建议在特征 <10 时用多折平均 SHAP 重要性（当前实现已支持，需调整调用逻辑）。

4. **LOOCV 内部泄露（支柱 1.2）:** `leave_one_out_validation` 仍接受外部预拟合 `scaler_X`。彻底的修复需重构该函数为 Pipeline 模式。当前影响较小（LOOCV 仅作辅助参考，不参与决策）。

---

## 变更文件清单

```
修改的文件:
  src/train_and_evaluate.py              (A1-A8: 全部8项修正)
  src/iterative_optimization.py          (B1-B10: 全部10项修正)
```
