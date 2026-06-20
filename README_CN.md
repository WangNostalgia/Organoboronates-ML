# 稳定有机硼酸酯活化能预测机器学习工具

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)]() [![License](https://img.shields.io/badge/License-MIT-green)]()

[English](README.md) | [中文](README_CN.md)

本仓库提供一个用于 organoboronates 活化能预测的机器学习工作流。当前任务格式是单样本回归：一个化合物 / 构象 / 记录对应一个 `activation_energy`。

当前有效文档：

- 概念流程：[pipeline.md](pipeline.md)
- 操作手册：[user_manual.md](user_manual.md)

## 环境要求

- Python >= 3.12
- 推荐且可复现的安装方式：`uv sync --locked`
- `requirements.txt` 只是从 lockfile 导出的 pip 兼容快照，不应手写维护，也不应作为第二套依赖真相源。

默认完整环境除 scikit-learn 系列外，还需要 XGBoost、LightGBM、CatBoost，以及 `gplearn==0.4.2`。

## 输入数据格式

训练数据推荐格式：

```text
ID, SMILES, filename, activation_energy, descriptor_1, descriptor_2, ...
```

说明：

- `activation_energy` 是训练和带标签评估所需 target。
- `ID`、`SMILES`、`filename` 是推荐 metadata，不作为模型特征。
- 模型特征来自数值型 descriptor columns。
- 外部预测 CSV 可以没有 `activation_energy`；这种情况下进入 prediction-only mode。

## 快速开始

```bash
uv sync --locked
python main.py --n_trials 100 --min_features 5
python main.py --n_trials 20 --min_features 5
python main.py --help
python example/standalone_y_randomization.py
```

训练后常用命令：

```bash
python example/manual_selection_and_plot.py
python src/external_validation.py --list-models
python src/external_validation.py --model SVR --n_features 5 --data external.csv
python src/applicability_domain.py --model SVR --n_features 5 --training example/B_dataset.csv --external external.csv
```

## 当前默认模型注册表

`main.py` 默认启用：

- LinearRegression
- Ridge
- Lasso
- SVR
- DecisionTree
- RandomForest
- GradientBoosting
- XGBoost
- KRR
- MLP
- AdaBoost
- ElasticNet
- KNR
- LightGBM
- CatBoost
- GPlearn

`GaussianProcessRegressor` 仍保留在源码中，但默认注释禁用。

## 当前训练与评估协议

### 1. 固定的 development / final test 边界

`iterative_optimization()` 在入口处先做一次 80/20 划分，`random_state=40`，所有默认模型共享同一组样本边界。

- Development 集：调参、SHAP 特征消除、特征数路径评估、内部 5×5 RepeatedKFold、LOOCV、100-split stability
- Final test 集：在特征数和超参数锁定后只评估一次

测试集不参与模型选择。Split indices 会保存到 checkpoint 的 `evaluation_protocol` 中，并被后处理脚本复用。

### 2. 只在 development 上调参

`src/train_and_evaluate.py` 只用 development 数据调参。

- Optuna 目标函数是内部 5-fold MAE
- Optuna trial 并行由 `--optuna_jobs` 控制，默认 `1`
- 模型内部并行由 `--model_jobs` 控制，默认 `-1`
- Ridge 和 Lasso 使用显式的 fold-local alpha 循环
- 每折只在训练部分拟合 scaler，再把预测值逆变换回 kcal/mol 后计算 MAE / R²

### 3. SHAP 驱动的特征消除

对非 GPlearn 模型，每一轮都会删掉且只删掉一个特征：

- 若存在高相关特征对，先删 SHAP 更弱的那个
- 否则删全局最不重要的特征

多折 SHAP consensus 中，非树模型的 background / masker 来自 training fold，而不是 held-out fold。路径会持续到配置的特征数下限。`--min_features` 默认是 `5`。

`--force_n_features` 表示“先评估完整路径，再强制选择某个已评估的特征数”。但默认注册表仍包含 GPlearn，因此 `main.py` 会立即拒绝这个选项；精确特征数强制选择只适用于排除 GPlearn 的自定义注册表。

### 4. 自动 final 与 manual-final

自动 final checkpoint 会在 development-only 证据确定特征数和超参数后生成，然后 final test 只评估一次。

Manual-final checkpoint 由 `example/manual_selection_and_plot.py` 根据 `example/manual_feature_selection.csv` 生成。该脚本会：

- 精确加载指定特征数的 iteration checkpoint
- 不允许 closest-feature fallback
- 复用 checkpoint 中保存的 development/final-test indices
- 用 selected features 和完整参数在 development rows 上重新拟合
- final test 只评估一次
- 保存 manual-final checkpoint、metrics txt 和带有 manual feature-count selection 标记的 scatter plot

Manual 特征数必须在查看 final-test 图之前，根据 development-only 证据和化学可解释性决定。

### 5. 指标角色与 checkpoint schema

Final checkpoints 使用嵌套指标：

- `metrics.primary.final_test.test_mae`
- `metrics.primary.final_test.test_r2`
- `metrics.secondary.internal_cv.*`
- `metrics.secondary.stability.*`
- `metrics.secondary.loo.*`

Iteration checkpoints 保存 development 路径指标：

- `metrics.internal_cv.*`
- `metrics.stability.*`
- `metrics.loo.*`

兼容旧字段可能仍会出现，但当前读取逻辑应优先使用新字段，只在必要时 fallback。

### 6. Standalone y-randomization

主流程中仍不自动运行 full-pipeline y-randomization。支持的路径是 `example/standalone_y_randomization.py`，该脚本：

- 加载已选 checkpoint
- 默认只使用 checkpoint 中保存的 development indices
- 对真实标签和随机标签复用同一组预计算的 5×5 RepeatedKFold split
- 报告有限置换修正 p-value `(b + 1) / (m + 1)`
- 将直方图写入 `models/y_randomization_<ModelName>.png`

### 7. Checkpoint 加载

`src.external_validation.load_model()`：

- 同时搜索 final 和 iteration checkpoints
- 默认要求特征数精确匹配
- 精确匹配时优先 final checkpoint，而不是 iteration checkpoint
- 用文件名 timestamp 打破平局
- 只有在 `allow_closest=True` / `--allow-closest` 时才允许最近特征数 fallback

`example/load_checkpoint_guide.py` 使用同样的选择逻辑，并能安全格式化当前和旧版指标布局。

### 8. 外部验证与 metadata 保留

`src/external_validation.py` 支持单模型和 ensemble 外部验证。

预测输出默认保留：

- `ID`、`SMILES`、`filename` 等优先 metadata
- 所有其他非 feature、非 target 的 metadata columns
- 模型实际使用的 feature columns
- `predicted_activation_energy`
- 如果存在真实标签，则额外加入 `activation_energy` 和误差列

可以用 `--id-cols` 指定 metadata 优先顺序；如果不想保留所有 metadata，可以加 `--only-id-cols`。

### 9. Applicability domain

Applicability-domain 分析默认使用 checkpoint development indices 进行校准，并使用：

- training/development rows 的 5-fold OOF residuals
- MAD-based residual scale，并带有限值 fallback
- descriptor space leverage

只有在明确接受 full-CSV 校准时才使用 `--allow-full-training-csv-for-ad`。当外部数据没有标签时，prediction-only mode 只使用 leverage；没有外部 `sqrt(1-h)` 修正项。

## `main.py` CLI 参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--n_trials` | `100` | 每个模型在 development 集上的 Optuna trial 数 |
| `--model_jobs` | `-1` | 模型内部并行 CPU 数；`-1` 表示支持时使用全部可用核心 |
| `--optuna_jobs` | `1` | Optuna trial 并行数；默认串行以提高可复现性并避免资源爆炸 |
| `--n_jobs` | `None` | 已废弃的 `--model_jobs` 别名 |
| `--keep_versions` | `2` | 每个模型保留的最近 checkpoint family 数量 |
| `--min_features` | `5` | SHAP-RFECV 路径评估的特征数下限 |
| `--force_n_features` | `None` | 精确选择已评估特征数；默认 GPlearn 注册表下会被拒绝 |

## 输出结构

```text
models/
├── <ModelName>/
│   ├── <ModelName>_iteration_<N>_<timestamp>.joblib
│   ├── <ModelName>_iteration_<N>_<timestamp>_metrics.txt
│   ├── <ModelName>_final_<timestamp>.joblib
│   ├── <ModelName>_final_<timestamp>_metrics.txt
│   ├── <ModelName>_manual_final_<N>feat_<timestamp>.joblib
│   ├── <ModelName>_manual_final_<N>feat_<timestamp>_metrics.txt
│   ├── final_scatter_<timestamp>.png
│   ├── final_scatter_<timestamp>_outliers.csv
│   ├── performance_history_<timestamp>.csv
│   └── performance_history_<timestamp>.png
└── optimization_<timestamp>.log
```

Iteration checkpoints 保存 development 路径结果。Final checkpoints 额外保存一次性 final-test 结果、split metadata、已拟合 scaler、完整超参数以及最终特征顺序。Manual-final checkpoints 使用相同 final-test 协议，但特征数来自 `example/manual_feature_selection.csv`。

## 可复现性说明

- 固定 development/final split seed：`40`
- 固定 development 侧评估 seed：`42`
- 不裁剪 target values
- 优化流程中 Matplotlib 使用 `Agg` backend
- `src/fixed_params.py` 是固定模型参数的单一来源
- `src/evaluation.py` 是 5×5 RepeatedKFold 评估的单一来源
- `uv.lock` 是精确依赖解析的单一来源；需要 pip 兼容时再从 lockfile 导出 `requirements.txt`
