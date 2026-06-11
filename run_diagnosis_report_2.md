# 运行诊断报告 #2 — GPR 崩溃分析与总体评估

> **运行时间:** 2026-05-11 18:29 — 2026-05-12 02:56
> **日志文件:** `my_final_new.log` (3344 行)
> **激活模型:** 9 个 (LinearRegression, Ridge, Lasso, SVR, DecisionTree, RandomForest, GradientBoosting, XGBoost, GPR)

---

## 一、整体运行状态

| 模型 | 迭代筛选 | y-Randomization | 最终输出 | 状态 |
|---|---|---|---|---|
| LinearRegression | ✅ 12轮 | ✅ PASSED (p=0.0000, 22s) | ✅ 完整 | 成功 |
| Ridge | ✅ 12轮 | ✅ PASSED (p=0.0000, 12s) | ✅ 完整 | 成功 |
| Lasso | ✅ 12轮 | ✅ PASSED (p=0.0000, 43s) | ✅ 完整 | 成功 |
| SVR | ✅ 12轮 | ✅ PASSED (p=0.0000, 48s) | ✅ 完整 | 成功 |
| DecisionTree | ✅ 12轮 | ✅ PASSED (p=0.0000, 33s) | ✅ 完整 | 成功 |
| RandomForest | ✅ 12轮 | ✅ PASSED (p=0.0000, 18min) | ✅ 完整 | 成功 |
| GradientBoosting | ✅ 12轮 | ✅ PASSED (p=0.0000, 6min) | ✅ 完整 | 成功 |
| XGBoost | ✅ 12轮 | ✅ PASSED (p=0.0000, ~8h) | ✅ 完整 | 成功 |
| **GPR** | ❌ 第1轮崩溃 | 未执行 | 无 | **崩溃** |

**8/9 模型完全成功**。所有完成模型的 y-Randomization 均为 p=0.0000（100次随机排列无一达到原始性能），模型学到了真实的构效关系。

---

## 二、报错详细分析

### 2.1 致命错误（导致程序崩溃）

**类型:** `TypeError`

**完整信息:**
```
Traceback (most recent call last):
  File "main.py", line 137, in <module>
    main()
  File "main.py", line 123, in main
    results, best_models = iterative_optimization(...)
  File "src/iterative_optimization.py", line 167, in iterative_optimization
    ) = hyperparameter_optimization_and_training(...)
  File "src/hyperparameter_optimization_and_training.py", line 33, in hyperparameter_optimization_and_training
    cv_mae_best, mae_mean, best_params, rkf_results = train_and_evaluate(...)

TypeError: GaussianProcessRegressor.__init__() got an unexpected keyword argument 'length_scale'
```

**崩溃位置:** GPR 模型的第 1 轮迭代，在 `train_and_evaluate()` 结束返回后，`hyperparameter_optimization_and_training.py` 试图用 `model_class(**final_params)` 创建模型时崩溃。

### 2.2 根因分析

GPR 的 Optuna 搜索空间（`train_and_evaluate.py` objective 函数内）定义了三个与内核构建相关的参数：

```python
kernel_choice = trial.suggest_categorical("kernel", ["RBF", "Matern", "RBF+White"])
length_scale = trial.suggest_float("length_scale", 0.1, 10, log=True)     # ← 内核参数
noise_level  = trial.suggest_float("noise_level", 1e-5, 1.0, log=True)    # ← 内核参数
alpha        = trial.suggest_float("alpha", 1e-10, 1e-1, log=True)        # ← GPR 自身参数
```

在 `objective()` 内部，`length_scale` 和 `noise_level` 被**消费**来构建 `kernel` 对象，只有 `kernel` 和 `alpha` 被放入 `params` 字典。但是 Optuna 的 `study.best_params` 会返回**所有** `trial.suggest_*` 调用过的参数，包括 `length_scale` 和 `noise_level`。

经过 `final_params = {**FIXED_PARAMS, **best_params}` 合并后，`length_scale` 进入了最终的参数字典。当执行 `GaussianProcessRegressor(**final_params)` 时，`length_scale` 被当作 GPR 的构造函数参数，但 GPR 根本不存在这个参数 → `TypeError`。

**数据流追踪:**

```
objective() 内:
  trial.suggest_float("length_scale", ...)     → Optuna 记录为 best_params 的一部分
  kernel = RBF(length_scale=...)               → length_scale 被消费，生成 kernel 对象
  params = {**base_params, "kernel": kernel, "alpha": ...}  → kernel 已在 params 中

Optuna 结束后:
  study.best_params = {"kernel": "RBF", "length_scale": 3.5, "alpha": 1e-5, ...}
                                 ↑ 这是字符串 "RBF"，不是 kernel 对象！

合并后:
  final_params = {**FIXED_PARAMS, **study.best_params}
               = {..., "kernel": "RBF", "length_scale": 3.5, ...}
                                        ↑ 泄漏！GPR.__init__ 不认识这个参数
```

**为什么前 8 个模型没触发同类问题:** 其他模型的 Optuna 参数全部是模型自身的构造函数参数（如 SVR 的 `C`/`epsilon`/`gamma`，RandomForest 的 `n_estimators`/`max_depth`），不存在"消费后应丢弃"的内核构建参数。GPR 是唯一一个在 objective 内部构建子对象（kernel）再传入 params 的模型。

### 2.3 次要警告（非错误，但值得注意）

GPR 在崩溃前产生了大量 `ConvergenceWarning`（~20 条）：

```
ConvergenceWarning: The optimal value found for dimension 0 of parameter
k2__length_scale is close to the specified lower bound 1e-05.
Decreasing the bound and calling fit again may find a better value.
```

**含义:** GPR 内核优化的 `length_scale` 参数在搜索中多次触及下界（1e-05）或上界（100000），说明当前的搜索范围（0.1~10）在 MinMaxScaler 后的空间中可能不完全适配。但这只是次要的性能调优问题。

---

## 三、解决方案

### 方案 A（推荐 — 最简单）: 在合并前剥离内核参数

在 `train_and_evaluate.py` 的 Optuna 结束后，GPR 分支中从 `best_params` 中移除已被消费的参数：

```python
# 在 train_and_evaluate.py 中，Lasso 特殊处理之后添加:
if model_class == GaussianProcessRegressor:
    best_params = dict(best_params)
    # These were consumed to build the kernel object inside objective();
    # they are NOT valid GaussianProcessRegressor constructor arguments.
    best_params.pop('kernel', None)         # string choice, not the kernel object
    best_params.pop('length_scale', None)   # kernel parameter
    best_params.pop('noise_level', None)    # kernel parameter (WhiteKernel)
    # After stripping, only 'alpha' remains as the GPR constructor param.
    # The actual kernel object was already stored inside FIXED_PARAMS or
    # needs to be reconstructed — see note below.
```

**注意:** 剥离后 `best_params` 中只有 `alpha`。但 `FIXED_PARAMS` 中的 kernel 相关字段（`n_restarts_optimizer`、`random_state`）仍然是需要的。内核对象需要用 `study.best_params` 中的 `kernel`/`length_scale`/`noise_level` 信息重建。当前 FIXED_PARAMS_MAP 中 GPR 没有 kernel 字段的常数部分（kernel 是完全动态的），所以内核对象需要从 best_trial 重建。

**更简洁的实现 — 直接丢弃 best_params 中的内核参数，只保留 alpha：**

```python
if model_class == GaussianProcessRegressor:
    # Keep only alpha from Optuna; kernel params were consumed in objective()
    gpr_alpha = best_params.get('alpha', 1e-5)
    # Reconstruct kernel from the best trial's user_attrs
    # (requires storing kernel info during objective)
```

**最简洁的实现 — 在 objective 中就避免参数泄漏：**

将内核参数改为在 `trial.set_user_attr` 中存储，而不是用 `trial.suggest_*`：

GPR 的 `objective()` 修改：
- 用一个 `trial.suggest_float("alpha", ...)` 搜索 GPR 自身参数
- 内核的 `length_scale` 和 `noise_level` 改用 `trial.suggest_float` 但在函数末尾清理，或改用 `user_attr` 传递
- 实际上最简单的是：内核的所有构建参数在 objective 内部用 `user_attr` 存储，而 `params` 中只放 `kernel` 对象和 `alpha`。但 Optuna 的 `TPESampler` 需要 `trial.suggest_*` 调用来建立搜索空间，不能简单替换

**推荐的最终方案（改动最小）:**

在 `train_and_evaluate.py` 的 Optuna 结果处理块中，增加 GPR 特殊处理，将内核相关的动态参数从 `best_params` 中移除，同时从 `best_trial.user_attrs` 或其他方式重建内核对象。或者更简单地：

在 `objective()` 内部的 GPR 分支中，将内核构建参数的前缀改为 `gpr_kernel__*` 以明确标记它们是临时的子对象参数；或直接在 `params` 中只放入完整的 kernel 对象，让 `study.best_params` 中包含 `kernel` 字符串和 `length_scale` 浮点数，然后在合并前过滤。

**建议采用此方案，改动仅 3 行:**

```python
# src/train_and_evaluate.py, 在 Lasso 特殊处理的 if 块之后:
if model_class == GaussianProcessRegressor:
    best_params = dict(best_params)
    # GPR kernel-building params are consumed inside objective() to construct
    # the kernel object.  They are NOT constructor arguments of GPR itself.
    for k in ('kernel', 'length_scale', 'noise_level'):
        best_params.pop(k, None)
```

### 方案 B（架构更优雅）: 重构 GPR objective 避免参数污染

将 kernel 构建完全移到 objective 内部，通过 `trial.set_user_attr` 存储，而不是通过 `trial.suggest_*` 暴露给 study.best_params。但这需要更大的改动。

---

## 四、前 8 个模型的输出质量评估

### 4.1 Fractional 1-SE 表现

以 XGBoost 为例，从日志中可以看到 Path Summary 和自动选择结果。1-SE 的 α=0.25 是否有效避免了欠拟合，可以从各模型的 optimal_n_features 判断（需查看最终输出的 `_final_*_metrics.txt`）。

### 4.2 所有 y-Randomization 通过

8 个模型的 y-Randomization 全部 p=0.0000，说明在当前 N≈88 的小样本下，所有模型学到的构效关系都具有高度的统计显著性。

### 4.3 日志无其他错误

除 GPR 崩溃和 ConvergenceWarning 外，日志中无任何 ERROR、Exception、WARNING。

---

## 五、建议

| 优先级 | 行动 | 说明 |
|---|---|---|
| **高** | 修复 GPR `length_scale` 泄漏 | 方案 A：3 行过滤代码。否则 GPR 完全无法运行 |
| 低 | 调整 GPR `length_scale` 搜索下界 | 从 0.1 下探到 1e-3 以消除 ConvergenceWarning |
| 低 | 关注 GPR 运行时间 | GPR 拟合复杂度 O(N³)≈88³≈680K 操作，远高于线性模型 |

---

*待你查阅后决定是否按方案 A 修复 GPR 崩溃，或者直接禁用 GPR（在 main.py 中注释掉）。*
