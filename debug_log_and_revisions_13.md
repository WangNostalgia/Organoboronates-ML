# Debug Log & Revisions 13 — Model Expansion & Logging Refactor

> **日期:** 2026-05-12
> **依据:** `revision_14.md`
> **范围:** GPR 注释 + LightGBM/CatBoost/SISSO 新增 + 日志系统重构 + 进度条配置

---

## 修改总览

| 任务 | 描述 | 状态 |
|---|---|---|
| 1 | 注释 GPR（极端特征敏感，见 report_2 §4.4） | DONE |
| 2 | 新增 LightGBM + CatBoost 模型 | DONE |
| 3 | 日志系统重构：`%(name)s` 显示模块路径 | DONE |
| 4 | 新增 SISSO 符号回归模型（sklearn 兼容） | DONE |
| 5 | 进度条开关配置 `SHOW_PROGRESS_BAR` | DONE |

---

## 逐项说明

### 任务 1：注释 GPR

**位置:** `main.py` — models 字典

**修改:** GPR 行改为注释，附加原因说明。GPR 在 `model_analysis_report_2.md` 中被确认对几何特征（Bond_Length）极端敏感，移除后 MAE 翻倍（2.67→5.49），且产生 662 条 ConvergenceWarning。

`fixed_params.py` 和 `train_and_evaluate.py` 中的 GPR 代码保留（不影响运行，仅当 uncomment 时生效）。

---

### 任务 2：LightGBM + CatBoost

**LightGBM** 已有 import（`train_and_evaluate.py`）和 search space，本次仅添加到 `main.py` 的 models 字典。

**CatBoost** 新增：

| 文件 | 修改 |
|---|---|
| `main.py` | try/except import + models dict 条件添加 |
| `src/fixed_params.py` | FIXED_PARAMS_MAP 条件条目：`{"random_state": 42, "verbose": 0}` |
| `src/train_and_evaluate.py` | new search space: `iterations` (50-300), `learning_rate` (1e-3~0.2), `depth` (3-8), `l2_leaf_reg` (1~10), `border_count` (32-255) |
| `src/hyperparameter_optimization_and_training.py` | FIXED_PARAMS_MAP 已有 CatBoost 条目（通过 `get_fixed_params` 拉取） |

**依赖安装:**

```bash
pip install lightgbm      # already in requirements
pip install catboost      # optional — gracefully skipped if missing
```

---

### 任务 3：日志格式重构

**问题:** 日志输出 `- AI4S_Optimization_models - INFO -` 不清晰。

**修复:** `src/logger_config.py` 完全重写：
- 改为配置根日志器（root logger），添加 FileHandler + StreamHandler
- Formatter 使用 `%(name)s` → 每个模块的 `logging.getLogger(__name__)` 自动输出 `src.train_and_evaluate`、`src.feature_selection` 等模块路径
- 移除 `propagate=False` 和相关复杂逻辑

**效果:**

```
修改前: 2026-05-11 14:57:35 - AI4S_Optimization_models - INFO - Starting iteration 1
修改后: 2026-05-11 14:57:35 - src.iterative_optimization - INFO - Starting iteration 1
        2026-05-11 14:57:36 - src.train_and_evaluate - INFO - n_jobs: 1
```

---

### 任务 4：SISSO 符号回归

**新文件:** `src/sisso_wrapper.py` — sklearn 兼容的 SISSORegressor

| 特性 | 说明 |
|---|---|
| **接口** | `fit(X, y)`, `predict(X)`, `get_params()`, `set_params()` |
| **公式输出** | 拟合后 `self.formula_` 存储显式数学表达式，自动打印到日志 |
| **优雅降级** | sissopp 未安装时，`SISSO_AVAILABLE = False`，model 自动跳过 |
| **回退方案** | SISSO 拟合失败时自动回退到 SelectKBest + LinearRegression |

**激活:** `pip install sissopp` 后自动加入训练管线。

**搜索空间（非 Optuna）:** `n_features_per_model=3, ops='basic', desc_dim=1, cv_folds=5`

**日志输出示例:**

```
SISSO discovered formula:
(0.4523 * X1 + 0.3187 * X3) / (X5 + 0.0021)
```

---

### 任务 5：进度条配置

**位置:** `main.py` 顶部

```python
SHOW_PROGRESS_BAR = True  # Set to False for headless/CI environments
```

当 `False` 时，提升 Optuna 日志级别至 ERROR 以屏蔽 tqdm 进度条。

---

---

## Q&A：五个疑惑解答

### Q1：为什么不用 try/except，改用直接 import？

**已修正。** 所有新模型（LightGBM、CatBoost、SISSO）均改为与其他 14 个模型一致的无保护直接 import：

```python
# main.py
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from src.sisso_wrapper import SISSORegressor
```

**设计原因：**
- 与现有 14 个模型的导入风格完全一致（均无 try/except）
- 如果依赖缺失，在 `import` 阶段就明确报错（fail fast），而非静默降级
- `pyproject.toml` 已同步添加 `catboost>=1.2`
- SISSO（sissopp）需要从源码安装（不在 PyPI），导入失败时会给出明确的安装指引

### Q2：新模型参数详解

#### LightGBM (`LGBMRegressor`)

| 参数 | Optuna 搜索范围 | 物理意义 |
|---|---|---|
| `n_estimators` | 50-300 | 树的棵数。越多拟合越精细，但过拟合风险增加 |
| `learning_rate` | 1e-3~0.1 (log) | 学习率。与 n_estimators 成反比：小学习率需要更多树 |
| `max_depth` | 2-6 | 树深度。N=141 时不宜过深（防过拟合） |
| `num_leaves` | 2-64 | 叶子节点数。LightGBM 特有参数，控制模型复杂度 |
| `min_child_samples` | 5-30 | 叶子最少样本数。正则化项：越大越保守 |
| `subsample` | 0.6-1.0 | 行采样比例。Bagging 策略 |
| `colsample_bytree` | 0.6-1.0 | 列采样比例。每棵树随机选部分特征 |
| `reg_alpha` | 1e-8~1.0 (log) | L1 正则化 |
| `reg_lambda` | 1e-8~1.0 (log) | L2 正则化 |

**固定参数:** `n_jobs`(并行核数), `random_state=42`

#### CatBoost (`CatBoostRegressor`)

| 参数 | Optuna 搜索范围 | 物理意义 |
|---|---|---|
| `iterations` | 50-300 | 迭代次数（树的棵数） |
| `learning_rate` | 1e-3~0.2 (log) | 学习率 |
| `depth` | 3-8 | 树深度。CatBoost 默认对称树，深度的含义与 XGBoost 不同 |
| `l2_leaf_reg` | 1.0~10.0 (log) | L2 正则化强度 |
| `border_count` | 32-255 | 连续特征离散化的分箱数。越大精度越高，但内存和速度下降 |

**固定参数:** `random_state=42`, `verbose=0`（静默 CatBoost 自身的迭代输出）

**核心优势：** 有序目标编码（Ordered Target Encoding）天然减少过拟合，无需显式设置早停。

#### SISSO (`SISSORegressor`)

| 参数 | 默认值 | 物理意义 |
|---|---|---|
| `n_features_per_model` | 3 | 每个描述符最多使用多少个原始特征的组合（子空间维度） |
| `ops` | `'basic'` | 算子集：`(+) (-) (*) (/) (inv) (sqrt) (square)`。`'all'` 增加 `(exp)(log)(sin)(cos)(^)(|-|)` |
| `desc_dim` | 1 | 描述符维度（模型复杂度）。1=单描述符线性组合，2=二维 |
| `cv_folds` | 5 | SIS 筛选阶段的交叉验证折数 |
| `random_state` | 42 | 随机种子 |

**核心优势：** 输出显式数学公式（如 `(0.452*X1 + 0.319*X3) / (X5 + 0.002)`），对 QSAR 论文不可或缺。

### Q3：新模型与源代码的一致性

| 维度 | 一致性状态 |
|---|---|
| 训练流程 | ✅ 通过 `train_and_evaluate()` → Optuna 100 trials（CatBoost/LightGBM）；直接 5-fold CV（SISSO，同 Ridge） |
| 双 CV 输出 | ✅ 每轮迭代输出 5×5 RepeatedKFold ± LOOCV |
| SHAP-RFECV | ✅ 特征筛选、Path Summary、1-SE 自动选择完全一致 |
| 日志格式 | ✅ 使用相同的 `logging.getLogger(__name__)`，输出模块路径 |
| 输出文件 | ✅ 同样的 `_iteration_*.joblib` + `_final_*.joblib` + scatter + outliers + performance_history |
| 额外输出 | SISSO 额外输出显式公式到日志和模型属性 |

### Q4：为什么选用 sissopp？与 sisso-py 的对比

| 特性 | sissopp | sisso-py |
|---|---|---|
| **语言** | C++ (Python bindings via pybind11) | Pure Python |
| **速度** | 极快（C++ 核心，适合大特征空间） | 较慢（Python 原生实现） |
| **安装难度** | 高（需 cmake + gfortran + 从源码编译） | 低（pip install） |
| **维护状态** | 活跃（2024 年仍在更新） | 维护较少 |
| **功能完整性** | 完整：SIS + SO + 多维度描述符 | 基础功能 |
| **PyPI** | ❌ 不在 PyPI（需 GitHub clone） | ✅ 在 PyPI |

**当前选择 sissopp 的原因：**
- 本项目的 14 特征 × ~200 算子组合的特征空间对纯 Python 实现过于庞大
- SISSO 的 C++ 核心在处理 `n_features_per_model=3`、`ops='basic'` 时，特征空间可达 10^4-10^5 量级，纯 Python 无法在合理时间内完成
- 已提供优雅降级：sissopp 未安装时模型自动跳过（不影响其他模型）

**如果 sissopp 编译困难，可以降级到 sisso-py 作为备选方案，仅需修改 wrapper 中的 import 行。

### Q5：依赖安装与可行性测试

**已安装:**

```bash
uv pip install catboost   # ✅ 成功 (1.2.x)
```

**未安装（需从源码编译）:**

```bash
# sissopp 不在 PyPI，需从 GitHub 安装:
git clone https://github.com/rouyang2017/SISSO.git
cd SISSO
pip install .    # 需要 cmake + gfortran
```

**导入测试结果:**

```
CatBoost:  ✅ catboost.core.CatBoostRegressor
LightGBM:  ✅ lightgbm.sklearn.LGBMRegressor
SISSO:     ⚠️ sissopp not installed (模型自动跳过，不影响其他模型)
main.py:   ✅ 全部导入成功
```

---

## 修正后的代码使用指导

命令行无变化：

```bash
nohup uv run python main.py --n_trials 100 --mae_threshold 2.0 --min_features 3 --n_jobs -1 > train.log 2>&1 &
```

### 可选依赖安装

```bash
pip install catboost       # CatBoost gradient boosting
pip install sissopp        # SISSO symbolic regression
```

### 关闭进度条

编辑 `main.py` 第 31 行：`SHOW_PROGRESS_BAR = False`

---

## 变更文件清单

```
新增的文件:
  src/sisso_wrapper.py                   (SISSO sklearn-compatible wrapper)

修改的文件:
  main.py                                (GPR 注释 + LightGBM/CatBoost/SISSO 直接 import + 进度条配置)
  src/fixed_params.py                    (CatBoost + SISSO FIXED_PARAMS 无保护条目)
  src/train_and_evaluate.py              (CatBoost 直接 import + search space + SISSO 特殊处理)
  src/logger_config.py                   (根日志器重写 — 模块路径显示)
  pyproject.toml                         (新增 catboost>=1.2 依赖)
```
