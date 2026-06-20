# 机器学习流程说明：稳定有机硼酸酯活化能预测

本文档描述当前代码实际执行的工作流。Windows 上 Git 只能稳定跟踪一个 canonical 文件名，因此这里以小写的 `pipeline.md` 作为唯一有效流程文档。

## 总览

主入口是 `main.py`。当前任务是单个分子 / 单个构象 / 单个片段样本到 `activation_energy` 的回归预测。输入 CSV 推荐格式为：

```text
ID, SMILES, filename, activation_energy, descriptor_1, descriptor_2, ...
```

其中：

- `activation_energy` 是训练和带标签评估所需 target。
- `ID`、`SMILES`、`filename` 是推荐 metadata，不作为模型特征。
- 模型特征来自数值型 descriptor columns。
- 外部预测 CSV 可以没有 `activation_energy`；此时进入 prediction-only mode。

当前默认模型注册表包含：

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

`GaussianProcessRegressor` 仍在源码中，但默认注释禁用。

整个流程围绕一个核心原则展开：

- 先固定一次 development / final test 的 80/20 划分。
- 所有默认模型共享同一组样本边界。
- 所有调参、特征删除、特征数选择和稳定性分析都发生在 development 集。
- Final test 只在特征数和超参数锁定后评估一次。
- 后处理脚本默认复用 checkpoint 中保存的 development/final-test indices，不重新创造样本边界。

## 阶段 1：数据准备与固定划分

`main.py` 会：

1. 读取 `example/B_dataset.csv`
2. 删除全空列
3. 使用 `activation_energy` 作为目标列
4. 显式排除 metadata columns：`ID`、`SMILES`、`filename`
5. 从其余数值列中构建 descriptor feature matrix

随后 `iterative_optimization()` 在入口处固定一次：

- `test_size=0.2`
- `random_state=40`

得到：

- Development 集：调参、SHAP 特征消除、特征数路径评估、内部 5×5 RepeatedKFold、LOOCV、100-split stability
- Final test 集：最终一次性评估

测试集不参与任何模型选择。切分结果会保存到 checkpoint 的 `evaluation_protocol["development_indices"]` 和 `evaluation_protocol["final_test_indices"]` 中，供 manual-final、AD、standalone y-randomization 等后处理脚本复用。

## 阶段 2：development-only 调参与并行控制

`src/train_and_evaluate.py` 只在 development 数据上选参数。

当前实现要点：

- Optuna 目标函数是内部 5-fold MAE。
- Optuna study 是内存对象。
- `--optuna_jobs` 控制 Optuna trial 并行数，默认 `1`，以减少 TPE 顺序不稳定和嵌套并行资源爆炸。
- `--model_jobs` 控制模型内部并行，默认 `-1`。
- `--n_jobs` 仍作为 `--model_jobs` 的废弃别名保留。
- Ridge 和 Lasso 使用显式的 fold-local alpha 循环。
- 每一折各自拟合 scaler。
- MAE / R² 都在逆变换回 kcal/mol 后计算。

## 阶段 3：development 指标记录

当前 checkpoint 里的指标有明确层级。

Final checkpoint:

- `metrics.primary.final_test.test_mae`
- `metrics.primary.final_test.test_r2`
- `metrics.secondary.internal_cv.*`
- `metrics.secondary.stability.*`
- `metrics.secondary.loo.*`

Iteration checkpoint:

- `metrics.internal_cv.*`
- `metrics.stability.*`
- `metrics.loo.*`

旧别名仍可能存在，但只是兼容层，不再是首选 schema。

## 阶段 4：SHAP 驱动的迭代特征消除

除 GPlearn 外，其余模型都走迭代删特征路径。

每一轮：

1. 在当前 development 特征子集上完成调参与训练。
2. 保存 development-path iteration checkpoint。
3. 记录当前特征数、特征列表、development 指标和已删除特征。
4. 用 SHAP-RFECV 逻辑删除一个特征。

删除优先级：

1. 如果存在高相关特征对，删除其中 SHAP 更弱的那个。
2. 否则删除全局最不重要特征。

多折 SHAP consensus 中，非树模型的 `LinearExplainer` / `KernelExplainer` background 或 masker 来自 training fold，而不是 held-out fold。Held-out fold 只作为待解释样本使用。这样验证 fold 的特征分布不会参与解释基线定义。

流程会持续到达到下限。`--min_features` 当前默认值是 `5`。

`--force_n_features` 的语义仍然是“先把路径评估完，再选精确特征数”，但 `main.py` 默认 registry 含 GPlearn，所以 CLI 会在运行最开始直接拒绝该参数。若需要精确特征数选择，应通过不含 GPlearn 的自定义 registry 调用 API，或使用 manual-final 流程基于已有 iteration checkpoint 做人工特征数锁定。

## 阶段 5：自动选择最终特征数与一次性 final test

路径评估结束后，自动选择逻辑只看 development 内部指标：

1. 找到 `internal_cv.rkf_mae_mean` 最低的路径点。
2. 计算阈值：`best_mae + 0.25 * best_std`。
3. 在阈值内选择特征数最少的路径点。
4. 锁定该特征数、特征列表和对应 estimator 配置。
5. 在全部 development 样本上重新拟合最终模型。
6. 在 untouched final test 上只评估一次。

最终主结果是：

- `metrics.primary.final_test.test_mae`
- `metrics.primary.final_test.test_r2`

Development 证据保留在：

- `metrics.secondary.internal_cv.*`
- `metrics.secondary.stability.*`
- `metrics.secondary.loo.*`

## 阶段 6：manual-final 人工特征数选择

`example/manual_selection_and_plot.py` 现在是严格的 manual-final artifact 生成脚本，而不只是画图工具。

输入文件：

```text
example/manual_feature_selection.csv
```

格式：

```csv
model_name,n_features
SVR,5
RandomForest,7
XGBoost,4
```

执行流程：

1. 读取 manual feature-count CSV。
2. 对每个模型精确加载指定 `n_features` 的 iteration checkpoint。
3. 如果没有精确匹配，直接报错，不自动使用 closest checkpoint。
4. 从 checkpoint 读取 `evaluation_protocol["development_indices"]` 和 `evaluation_protocol["final_test_indices"]`。
5. 用 selected features + complete params 在 development rows 上重新拟合。
6. Final test 只评估一次。
7. 保存 manual-final checkpoint、metrics txt 和 final scatter。
8. 在 checkpoint、metrics 和图标题中标记 `manual_feature_count_selection`。

Manual-final checkpoint 与自动 final checkpoint 可平行比较，因为二者使用同一组 locked split indices，且 final test 都只在锁定特征数后评估一次。需要注意：manual 特征数选择本身必须基于 development-only 证据和化学可解释性，不能在看过 final-test 图后反向调整。

## 阶段 7：保存、外部验证与后续分析

### Checkpoint 保存

流程会保存 iteration checkpoint、automatic final checkpoint 和 manual-final checkpoint。

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

### Checkpoint 加载规则

`src.external_validation.load_model()` 当前规则：

- 同时搜索 final 与 iteration checkpoint。
- 默认要求精确特征数。
- 精确匹配时 final 优先于 iteration。
- 同类候选按文件名时间戳确定性排序。
- 只有显式 `allow_closest=True` / `--allow-closest` 才允许最近邻回退。

`example/load_checkpoint_guide.py` 与这套规则保持一致，并优先读取当前 metric schema，再回退到旧别名。

### 外部验证与预测输出

`src/external_validation.py` 支持单模型和 ensemble 外部验证。

默认输出 CSV 会保留：

- 优先 metadata：`ID`、`SMILES`、`filename`
- 其他所有非 feature、非 target 的 metadata columns
- 模型实际使用的 feature columns
- `predicted_activation_energy`
- 如果有真实标签，则加入 `activation_energy` 和 error columns

这保证外部预测结果能追踪具体化合物、构象、来源文件或批次。若只想保留指定 ID 列，可使用 CLI 的 `--id-cols` 和 `--only-id-cols`。

### y-randomization

自动主流程 y-randomization 仍关闭，当前支持方式是 `example/standalone_y_randomization.py`。它会：

- 读取锁定后的 checkpoint。
- 默认只在 checkpoint 保存的 development indices 上运行。
- 若 checkpoint 没有 development indices，默认报错，而不是回退到全数据。
- 让原始标签与置乱标签复用同一组预计算 5×5 RepeatedKFold splits。
- 输出修正后的有限置换 p 值 `(b + 1) / (m + 1)`。
- 将图保存为根模型目录下的 `models/y_randomization_<ModelName>.png`。

### Applicability Domain

当前 AD CLI 默认从 checkpoint 读取 development indices，并从用户传入的原始 training CSV 中筛出 development rows 做校准。

当前 AD 逻辑使用：

- checkpoint development rows，而不是 full training CSV，作为 AD calibration reference。
- 训练集 5-fold OOF residuals。
- MAD 残差尺度与有限回退。
- 描述符空间 leverage。

如果用户确实想用完整 training CSV 做 AD 校准，必须显式传入 `--allow-full-training-csv-for-ad`。如果外部数据没有真实标签，则进入 leverage-only prediction 视图；不会使用外部 `sqrt(1-h)` 修正。

## 环境与依赖管理

主安装路径是：

```bash
uv sync --locked
```

依赖职责分工：

- `pyproject.toml`：声明依赖范围。
- `uv.lock`：精确、可复现的解析结果，是环境真相源。
- `requirements.txt`：从 lockfile 导出的 pip 兼容快照，不应手工维护。

如需 pip 路径，应先从 lockfile 导出并注明平台，避免不同机器安装出不同环境。
