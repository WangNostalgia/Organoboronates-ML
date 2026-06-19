# 稳定有机硼酸酯反应活化能预测机器学习工具

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)]() [![License](https://img.shields.io/badge/License-MIT-green)]() [![DOI](https://img.shields.io/badge/DOI-10.1038%2Fs41467--025--60674--9-blue)](https://doi.org/10.1038/s41467-025-60674-9)

[English](README.md) | [中文](README_CN.md)

本仓库是 Nature Communications 论文 [“Organometallic-type reactivity of stable organoboronates for selective (hetero)arene C−H/C-halogen borylation and beyond”](https://doi.org/10.1038/s41467-025-60674-9) 的机器学习补充代码。项目用于预测稳定有机硼酸酯反应的活化能（kcal/mol），并提供训练、检查点管理、外部验证、适用域分析和独立 y-randomization 工具。

进一步说明见：

- 概念流程： [Pipeline.md](Pipeline.md)
- 执行手册： [user_manual.md](user_manual.md)
- 仓库代理说明： [AGENTS.md](AGENTS.md)

## 环境要求

- Python >= 3.12
- 推荐安装方式：`uv sync`
- 备选安装方式：`pip install -r requirements.txt`

当前支持的完整默认安装包含：

- scikit-learn
- optuna
- shap
- xgboost
- lightgbm
- catboost
- gplearn
- matplotlib / seaborn / pandas / numpy / joblib

## 快速开始

```bash
uv sync
python main.py --n_trials 100 --min_features 5
```

常用变体：

- 快速测试：`python main.py --n_trials 20 --min_features 5`
- 查看 CLI 帮助：`python main.py --help`
- 评估笔记本：`jupyter notebook example/example_pic.ipynb`
- 预测笔记本：`jupyter notebook example/prediction_round2.ipynb`
- 独立 y-randomization：`python example/standalone_y_randomization.py`

## 当前默认模型注册表

`main.py` 当前默认启用的模型如下：

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

`GaussianProcessRegressor` 仍保留在源码中，但目前是注释禁用状态，不属于默认运行集合。

## 当前训练与评估协议

### 1. 固定的 development / final test 边界

`iterative_optimization()` 在入口处先做一次 80/20 划分，`random_state=40`，并且所有模型共享这一组样本边界。

- Development 集（80%）：超参数优化、SHAP 特征消除、特征数路径评估、内部 5×5 RepeatedKFold、LOOCV、100 次稳定性分析
- Final test 集（20%）：在特征数与超参数锁定后只评估一次

测试集不参与模型选择。

### 2. 超参数优化

`src/train_and_evaluate.py` 只在 development 数据上调参。

- Optuna 目标函数使用 development 内部 5-fold MAE
- Study 按每次运行在内存中完成
- Ridge 和 Lasso 采用显式的折内 alpha 循环，并且每折单独缩放
- 每一折都只在训练部分拟合 `X`/`y` 的 scaler，再把预测值逆变换回 kcal/mol 后计算 MAE

### 3. 特征消除与特征数选择

除 GPlearn 外，其余模型都走 SHAP 驱动的特征消除路径：

- 特征很多时，用单次拟合 SHAP 做粗筛
- 特征减少到阈值附近时，切换到 5-fold 共识 SHAP-RFECV
- 优先去掉高相关特征对中 SHAP 更弱的一方；若没有高相关对，再去掉全局最弱特征

`--min_features` 是路径停止下限，默认值为 `5`。

`--force_n_features` 不会跳过路径评估。它会先把 SHAP-RFECV 路径跑到 `--min_features`，再从已评估路径中选择指定的精确特征数。

### 4. 指标角色

主指标（最终泛化）：

- `test_mae`
- `test_r2`

次级 development-only 指标：

- `internal_cv.rkf_mae_mean/std`
- `internal_cv.rkf_r2_mean/std`
- `stability.mae_mean/std`
- `loo.mae`
- `loo.r2`

旧别名仍会为了兼容性保留，但已经不是主要的模型选择依据。

### 5. y-randomization

自动全流程 y-randomization 由于耗时过高仍然关闭。当前受支持的方式是独立脚本：

- `example/standalone_y_randomization.py`

该脚本会：

- 读取已经锁定的检查点
- 让原始标签和置乱标签复用同一组预计算 5×5 RepeatedKFold 切分
- 输出修正后的有限置换 p 值 `(b + 1) / (m + 1)`

### 6. 检查点加载与外部验证

检查点选择规则由 `src.external_validation.load_model()` 定义：

- 同时搜索 final 与 iteration 检查点
- 默认必须精确匹配特征数
- 精确匹配时 final 优先于 iteration
- 在同类候选中按文件名时间戳选最新
- 只有显式启用 `allow_closest=True` / `--allow-closest` 才允许最近邻回退

`example/load_checkpoint_guide.py` 给出了与该逻辑一致的示例。

### 7. 适用域（Applicability Domain）

适用域分析当前使用：

- 训练集 5-fold OOF 残差
- 基于 MAD 的残差尺度，并带有限值回退
- 描述符空间 leverage

如果外部数据没有真实标签，Williams 图会自动退化为仅 leverage 的预测视图，而不会从缺失标签中硬造残差阈值。

## `main.py` CLI 参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--n_trials` | `100` | 每个模型在 development 集上的 Optuna trial 数 |
| `--n_jobs` | `-1` | CPU 核数（`-1` 表示全部可用） |
| `--keep_versions` | `2` | 每个模型保留的近期 final/iteration 检查点族数量 |
| `--min_features` | `5` | SHAP-RFECV 路径停止下限 |
| `--force_n_features` | `None` | 不自动选最优点，改为从已评估路径中选指定特征数 |

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

Iteration 检查点保存的是 development-only 指标。Final 检查点在此基础上增加一次性的 final test 结果、split 元数据、scaler、完整超参数和最终特征顺序。

## 可复现性说明

- development/final split 固定种子：`40`
- development 内部评估默认种子：`42`
- 目标值不会被裁剪
- 优化流程中的 Matplotlib backend 固定为 `Agg`
- `src/fixed_params.py` 是固定参数单一真源
- `src/evaluation.py` 是 5×5 RepeatedKFold 的单一真源
