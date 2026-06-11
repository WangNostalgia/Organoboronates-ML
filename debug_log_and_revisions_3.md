# Debug Log & Revisions 3 — Final Polishing

> **日期:** 2026-05-11
> **依据:** `revision_4.md` 终极修改建议合集
> **范围:** `src/train_and_evaluate.py` + `src/iterative_optimization.py` + `src/feature_selection.py`

---

## 修改总览

| 任务 | 描述 | 优先级 | 状态 |
|---|---|---|---|
| 任务一 | Ridge MAE 量纲错位 Bug | **极高** | FIXED |
| 任务二 | RFECV 路径边界记录盲区 | 高 | FIXED |
| 任务三 | 多折 CV 共识 SHAP | 架构进阶 | FIXED |
| 任务四 | 1-SE 法则抵抗选择偏差 | 发文必备 | FIXED |
| 任务五 | Lasso 内部评估泄漏 + Ridge 目标函数对称修复 | 高 | FIXED |

---

## 逐项修正详情

### 任务一：Ridge MAE 量纲错位 Bug（极高优先级）

**缺陷本质:**

`train_and_evaluate.py` 的 Ridge 外层代码直接返回 `-ridge_cv.best_score_` 作为 `best_value`。这个值是在 `MinMaxScaler(0,100)` 缩放后的 y 空间内计算的（数值通常在 0.5-5 之间）。而其他所有模型（Lasso、SVR、RandomForest 等）返回的都是反变换回真实物理空间（kcal/mol）的 MAE（数值通常在 2-20）。

**严重后果:** Ridge 的 MAE 曲线在日志和性能历史图中异常低（~0.5 vs 其他模型的 ~2.5），产生"Ridge 碾压所有模型"的虚假视觉错觉，干扰 SHAP-RFECV 自动选择。

**修复方案:**

废弃直接读取 `best_score_`，为 Ridge 补充与 Lasso 完全对称的 **5 折手动还原评估循环**：

1. RidgeCV 内置 CV **仅负责找出数学上最优的 alpha**
2. 然后手动跑 5 折 CV，**每折逆变换回 kcal/mol 再算 MAE**
3. 返回的 `best_value` 是真实物理量纲的 MAE

**修复位置:**
- `src/train_and_evaluate.py` — Ridge 外层块（~line 415）+ Ridge 目标函数内块（~line 310）

**修改前后对比:**

```python
# 修改前（错误）:
ridge_cv.fit(X_tr_s, y_tr_s)
best_value = -ridge_cv.best_score_  # MinMaxScaler 空间！(~0.5)

# 修改后（正确）:
ridge_cv.fit(X_tr_s, y_tr_s)
best_alpha = ridge_cv.alpha_
# Manual 5-fold with inverse-transform
mae_list = []
for train_idx, test_idx in kf.split(X_train):
    # ... per-fold fit + inverse_transform ...
    mae_list.append(mean_absolute_error(y_te, y_pred_orig))
best_value = float(np.mean(mae_list))  # Real kcal/mol！(~2.5)
```

---

### 任务二：RFECV 路径边界记录盲区（高优先级）

**缺陷本质:**

原代码 `RECORD_RFECV_PATH = (current_n_features < 10)`。如果用户设定 `min_features=10`，当特征删到 10 个时直接 `break`，**10 特征的快照根本没有被记录**。自动最优特征数挑选逻辑对 `min_features ≥ 10` 完全失效。

**修复方案:**

```python
RECORD_RFECV_PATH = (current_n_features <= max(12, eff_min_features + 5))
```

确保无论用户设定的下限多高，都有至少 5 个特征数的缓冲记录窗口。`max(12, eff_min_features + 5)` 保证了：
- 当 `min_features=3` 时，记录 ≤12 特征 → 覆盖 3-12（与原来 <10 效果类似但更宽）
- 当 `min_features=10` 时，记录 ≤15 特征 → 覆盖 10-15（原来完全漏掉）

---

### 任务三：两级 SHAP 策略 — 粗筛用单次，精选用多折共识（架构进阶）

**缺陷本质:**

单次 `random_state=40` 的 train/test 切分在小样本（N=88）下产生的 SHAP 排序存在极大的偶然性。但如果在所有迭代中都运行 5 折 CV 共识 SHAP，KernelExplainer 的计算开销在特征数多时过大。

**修复方案:**

**两级策略** — 按特征数量自动切换：

```
特征数 > max(10, min_features+3):
  → cv_folds=0 → Single-fit SHAP
  → fit once on all X_train, explain once
  → 快速（KernelExplainer ~0.5-2 分钟/轮）
  → 目的：快速剔除明显不重要/共线的特征

特征数 ≤ max(10, min_features+3):
  → cv_folds=5 → Multi-fold CV consensus SHAP
  → 5 折独立训练 + SHAP on 验证折 + 平均
  → 稳健（KernelExplainer ~3-8 分钟/轮）
  → 目的：精选阶段，每步决策需要统计稳健性
```

**多折共识 SHAP 核心逻辑：**

```
for each of 5 folds:
    1. 训练模型 on 训练折
    2. 计算 SHAP on 验证折（仅对未见过的样本）
    3. 累积 |SHAP| 绝对值
对 5 折的累积重要性取平均 → 共识重要性排名
```

**修复位置:** `src/feature_selection.py` — `shap_rfecv_select_worst_feature()` 新增 `cv_folds` 参数；`src/iterative_optimization.py` — 根据 `RECORD_RFECV_PATH` 决定传 `cv_folds=0` 还是 `cv_folds=5`

**函数签名:**

```python
def shap_rfecv_select_worst_feature(model, X, y, scaler_X, model_name, corr_threshold=0.8, cv_folds=5):
    # cv_folds=0 → simple single-fit (coarse filtering)
    # cv_folds=5 → multi-fold consensus (fine-grained selection)
```

---

### 追记⑥：RandomState 替代全局 np.random.seed（代码规范修正）

**缺陷:** 多折共识 SHAP 循环内部的 KernelExplainer 路径中，每折都调用 `np.random.seed(42)` 重置全局种子。若各折验证集大小相同，`np.random.choice` 产出的伪随机索引序列完全一致。

**修复:** 在循环外初始化局部 `rng = np.random.RandomState(42)`，循环内使用 `rng.choice()` 和 `rng.randint()` 替代 `np.random.choice()` 和 `np.random.seed()`。同时 `shap.kmeans()` 的 `random_state` 参数由 `rng.randint()` 动态赋值，确保每折使用不同的随机种子。

**修复位置:** `src/feature_selection.py` — 多折共识路径内部

---

### 追记⑦：RFECV 记录窗口阈值微调

**修改:** `max(12, eff_min_features + 5)` → `max(10, eff_min_features + 3)`

**效果:** 记录窗口更紧凑。`min_features=3` 时记录 ≤10 特征（原 ≤12），减少不必要的高特征数路径记录。

---

### 任务四：1-SE 法则（One Standard Error Rule）— 发文必备

**缺陷本质:**

在同一批 RepeatedKFold 结果上挑选绝对 MAE 最低的特征组合，存在"验证集挖泥"倾向。假设 10 特征的 MAE=2.15 而 7 特征的 MAE=2.18，绝对最小值法会选 10 特征，但这两个 MAE 在统计上没有显著差异。

**修复方案:**

采用统计学的 **"一倍标准差法则 (1-SE Rule)"**：

```
1. 找到绝对 MAE 最小的记录 → min_mae, min_std
2. 计算容忍阈值: threshold = min_mae + min_std
3. 筛选所有 MAE ≤ threshold 的候选
4. 从中选择特征数最少的（奥卡姆剃刀）
```

**效果:**
- 如果 10 特征 MAE=2.15±0.20，7 特征 MAE=2.18±0.18
- threshold = 2.15 + 0.20 = 2.35
- 7 特征的 2.18 ≤ 2.35 → 在容忍范围内 → 选 7 特征（更简约）
- 日志输出：`★ 1-SE Rule triggered: absolute minimum at 10 features, selecting simpler model with 7 features`

**修复位置:** `src/iterative_optimization.py` — SHAP-RFECV 自动选择逻辑

---

### 任务五：Lasso 内部评估泄漏 + Ridge 目标函数对称修复

**Lasso 泄漏:**

LassoCV 的目标函数中，先对全局 `X_train` 执行 `fit_transform` 后再送入 LassoCV。LassoCV 内部做 5 折 CV 时，其验证折提前看到了全量训练集的归一化极值。

**Ridge 目标函数同源 Bug:**

Ridge 的目标函数直接返回 `-ridge_cv.best_score_`（MinMaxScaler 空间），与外层 Ridge 的 Bug 完全同源。

**修复方案:**

Ridge 和 Lasso 的目标函数均改为对称结构：
1. 内置 CV 获取最优 alpha
2. 手动 5 折循环，每折独立 fit scaler + inverse-transform 后计算 MAE
3. 返回真实物理量纲的 MAE

**修复位置:** `src/train_and_evaluate.py` — objective() 中 Ridge/Lasso 分支完全重写

---

## 修改后的代码使用指导

### 命令行参数（无变化）

```bash
nohup uv run python main.py --n_trials 100 --mae_threshold 2.0 --min_features 3 --n_jobs -1 > train.log 2>&1 &
```

### 关键行为变化

| 行为 | 修改前 | 修改后 |
|---|---|---|
| Ridge MAE | MinMaxScaler 空间 (~0.5) | 真实 kcal/mol (~2.5) |
| Lasso objective MAE | 5-fold 手动反变换（已正确） | 保持正确，结构更清晰 |
| RFECV 记录窗口 | `n_features < 10` | `n_features <= max(12, min_features + 5)` |
| SHAP 重要性来源 | 单次切分 (random_state=40) | **5 折 CV 共识**（平均 5 个独立折） |
| 最优特征数选择 | 绝对最小 MAE | **1-SE 法则**（最小 MAE ± 1 std，选最少特征） |

### 新增功能

- **1-SE Rule 输出**: 当触发时日志明确显示 "1-SE Rule triggered: absolute minimum at X features, selecting simpler model with Y features"
- **多折 SHAP 状态**: 日志显示 "Multi-fold (n=5) consensus SHAP ranking"
- **Ridge MAE 标签**: 日志显示 "5-Fold MAE (real kcal/mol)"

---

## 遗留问题与建议

1. **Optuna 热启动 (Warm Start):** revision_4.md 提到利用 `enqueue_trial` API 在下一轮迭代时注入上一轮的最优参数。当前 N=88 时开销可接受，暂未实现。如果未来数据量增大，建议实现：在 while 循环中保存上一轮 best_params，用 `study.enqueue_trial(best_params)` 注入。

2. **多折 SHAP 计算开销:** 5 折 × KernelExplainer 的计算量约为原来的 5 倍。对于树模型（TreeExplainer 极快），开销几乎可忽略；对 SVR/GPR/KRR（KernelExplainer），特征 <10 时每轮约 3-8 分钟，总体可接受。

3. **Pipeline 彻底化:** LassoCV/RidgeCV 的内置 CV 仍接受 pre-scaled 数据（因为需要通过缩放器拟合）。完全的 Pipeline 方案需要将 CV 包裹在 `cross_val_score` 中，这在当前架构下会导致 alpha 选择逻辑变化。当前方案（CV 仅用于选 alpha，手动 5 折用于评估）是合理的折中。

---

## 变更文件清单

```
修改的文件:
  src/train_and_evaluate.py              (任务一: Ridge MAE 修复 + 任务五: Ridge/Lasso objective 修复)
  src/iterative_optimization.py          (任务二: RFECV 边界修复 + 任务四: 1-SE Rule + 两级 SHAP 调度)
  src/feature_selection.py               (任务三: 两级 SHAP 策略 + cv_folds=0/5 + 追记⑥: RandomState)
```
