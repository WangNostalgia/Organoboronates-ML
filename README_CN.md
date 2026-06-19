# 稳定有机硼酸酯活化能预测机器学习工具

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)]() [![License](https://img.shields.io/badge/License-MIT-green)]() [![DOI](https://img.shields.io/badge/DOI-10.1038%2Fs41467--025--60674--9-blue)](https://doi.org/10.1038/s41467-025-60674-9)

[English](README.md) | [中文](README_CN.md)

本仓库是 Nature Communications 论文《Organometallic-type reactivity of stable organoboronates for selective (hetero)arene C−H/C-halogen borylation and beyond》的配套机器学习工作流代码。

当前有效文档：

- 概念流程：[pipeline.md](pipeline.md)
- 操作手册：[user_manual.md](user_manual.md)
- 代理说明：[AGENTS.md](AGENTS.md)

## 环境要求

- Python >= 3.12
- 推荐安装：`uv sync`
- 备用安装：`pip install -r requirements.txt`

默认完整环境除 scikit-learn 系列外，还需要 XGBoost、LightGBM、CatBoost，以及 `gplearn==0.4.2`。

## 快速开始

```bash
uv sync
python main.py --n_trials 100 --min_features 5
python main.py --n_trials 20 --min_features 5
python main.py --help
python example/standalone_y_randomization.py
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

测试集不参与模型选择。

### 2. 只在 development 上调参

`src/train_and_evaluate.py` 只用 development 数据调参。

- Optuna 目标函数是内部 5-fold MAE
- Optuna study 为内存对象
- Ridge 和 Lasso 使用显式的 fold-local alpha 循环
- 每折只在训练部分拟合 scaler，再把预测值逆变换回 kcal/mol 后计算 MAE / R²

### 3. SHAP 驱动的特征消除

对非 GPlearn 模型，每一轮都会删掉且只删掉一个特征：

- 若存在高相关特征对，先删 SHAP 更弱的那个
- 否则删全局最不重要的特征

因此路径会一直走到配置下限。`--min_features` 默认值是 `5`。

`--force_n_features` 的语义仍然是“先评估路径，再选精确特征数”，但由于默认 registry 含 GPlearn，`main.py` 会在最开始直接拒绝该选项。只有自定义且不含 GPlearn 的 registry 才适合精确特征数强制选择。

### 4. 指标角色与 checkpoint schema

Final checkpoint 使用嵌套指标结构：

- `metrics.primary.final_test.test_mae`
- `metrics.primary.final_test.test_r2`
- `metrics.secondary.internal_cv.*`
- `metrics.secondary.stability.*`
- `metrics.secondary.loo.*`

Iteration checkpoint 保留扁平 development-path 指标：

- `metrics.internal_cv.*`
- `metrics.stability.*`
- `metrics.loo.*`

旧别名仍可能为了兼容而保留，但读取时应优先使用当前键，再回退到旧别名。

### 5. 独立 y-randomization

自动主流程 y-randomization 仍然关闭。当前支持方式是 `example/standalone_y_randomization.py`，它会：

- 读取已锁定 checkpoint
- 让原始标签与置乱标签复用同一组预计算 5×5 RepeatedKFold splits
- 输出修正后的有限置换 p 值 `(b + 1) / (m + 1)`

### 6. Checkpoint 加载

`src.external_validation.load_model()` 的规则是：

- 同时搜索 final 与 iteration checkpoint
- 默认要求精确特征数
- 精确匹配时 final 优先于 iteration
- 同类候选按文件名时间戳确定性排序
- 只有显式使用 `allow_closest=True` / `--allow-closest` 才允许最近邻回退

`example/load_checkpoint_guide.py` 与这套规则保持一致，并能安全格式化当前键和旧别名。

### 7. Applicability Domain

AD 当前使用：

- 训练集 5-fold OOF residuals
- 基于 MAD 的残差尺度与有限回退
- 描述符空间 leverage

如果外部数据没有真实标签，预测模式会退化为 leverage-only；不会使用外部 `sqrt(1-h)` 修正项。

## `main.py` CLI 参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--n_trials` | `100` | development 集上的 Optuna trial 数 |
| `--n_jobs` | `-1` | CPU 核数（`-1` 表示全部可用） |
| `--keep_versions` | `2` | 每个模型保留的最近 checkpoint 家族数 |
| `--min_features` | `5` | SHAP-RFECV 路径评估下限 |
| `--force_n_features` | `None` | 精确特征数选择；默认 GPlearn registry 下会被 `main.py` 直接拒绝 |

## 输出结构

```text
models/
├── <ModelName>/
│   ├── <ModelName>_iteration_<N>_<timestamp>.joblib
│   ├── <ModelName>_iteration_<N>_<timestamp>_metrics.txt
│   ├── <ModelName>_final_<timestamp>.joblib
│   ├── <ModelName>_final_<timestamp>_metrics.txt
│   ├── final_scatter_<timestamp>.png
│   ├── final_scatter_<timestamp>_outliers.csv
│   ├── performance_history_<timestamp>.csv
│   └── performance_history_<timestamp>.png
└── optimization_<timestamp>.log
```

Iteration checkpoint 保存 development-path 结果；final checkpoint 额外保存一次性的 final-test 结果、split 元数据、拟合后的 scalers、合并后的超参数以及最终特征顺序。

## 复现说明

- development/final split 固定种子：`40`
- development 内部评估固定种子：`42`
- 目标值不裁剪
- 优化流程使用 `Agg` backend
- `src/fixed_params.py` 是固定参数单一真源
- `src/evaluation.py` 是 5×5 RepeatedKFold 评估单一真源
