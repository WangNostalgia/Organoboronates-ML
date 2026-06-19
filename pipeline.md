# 机器学习流程说明：稳定有机硼酸酯活化能预测

本文档描述当前代码实际执行的工作流。Windows 上 Git 只能稳定跟踪一个 canonical 文件名，因此这里以小写的 `pipeline.md` 作为唯一有效流程文档。

## 总览

主入口是 `main.py`。当前默认模型注册表包含：

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

- 先固定一次 development / final test 的 80/20 划分
- 所有默认模型共享同一组样本边界
- 所有选择行为都发生在 development 集
- final test 只在最后锁定后评估一次

## 阶段 1：数据准备与固定划分

`main.py` 会：

1. 读取 `example/B_dataset.csv`
2. 删除全空列
3. 选择数值列
4. 使用 `activation_energy` 作为目标列

随后 `iterative_optimization()` 在入口处固定一次：

- `test_size=0.2`
- `random_state=40`

得到：

- Development 集：调参、SHAP、特征数路径、内部 5×5 RepeatedKFold、LOOCV、100-split stability
- Final test 集：最终一次性评估

测试集不参与任何模型选择。

## 阶段 2：development-only 调参

`src/train_and_evaluate.py` 只在 development 数据上选参数。

当前实现要点：

- Optuna 目标函数是内部 5-fold MAE
- Optuna study 是内存对象
- Ridge 和 Lasso 使用显式的 fold-local alpha 循环
- 每一折各自拟合 scaler
- MAE / R² 都在逆变换回 kcal/mol 后计算

## 阶段 3：development 指标记录

当前 checkpoint 里的指标有明确层级：

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

1. 在当前 development 特征子集上完成调参与训练
2. 保存 development-path iteration checkpoint
3. 用 SHAP-RFECV 逻辑删除一个特征

删除优先级：

1. 如果存在高相关特征对，删除其中 SHAP 更弱的那个
2. 否则删除全局最不重要特征

因此流程不会因为“没有可删候选”而停止；它会持续到达到下限。`--min_features` 当前默认值是 `5`。

`--force_n_features` 的语义仍然是“先把路径评估完，再选精确特征数”，但 `main.py` 默认 registry 含 GPlearn，所以 CLI 会在运行最开始直接拒绝该参数。若需要精确特征数选择，应通过不含 GPlearn 的自定义 registry 调用 API。

## 阶段 5：最终特征数与一次性 final test

路径评估结束后：

1. 锁定特征数
2. 在全部 development 样本上重新拟合最终模型
3. 在 untouched final test 上只评估一次

最终主结果是：

- `metrics.primary.final_test.test_mae`
- `metrics.primary.final_test.test_r2`

development 证据保留在：

- `metrics.secondary.internal_cv.*`
- `metrics.secondary.stability.*`
- `metrics.secondary.loo.*`

## 阶段 6：保存、外部验证与后续分析

### Checkpoint 保存

流程会保存 iteration checkpoint 和 final checkpoint。

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

### Checkpoint 加载规则

`src.external_validation.load_model()` 当前规则：

- 同时搜索 final 与 iteration checkpoint
- 默认要求精确特征数
- 精确匹配时 final 优先于 iteration
- 同类候选按文件名时间戳确定性排序
- 只有显式 `allow_closest=True` / `--allow-closest` 才允许最近邻回退

`example/load_checkpoint_guide.py` 与这套规则保持一致，并优先读取当前 metric schema，再回退到旧别名。

### y-randomization

自动主流程 y-randomization 仍关闭，当前支持方式是 `example/standalone_y_randomization.py`。它会：

- 读取锁定后的 checkpoint
- 让原始标签与置乱标签复用同一组预计算 5×5 RepeatedKFold splits
- 输出修正后的有限置换 p 值 `(b + 1) / (m + 1)`
- 将图保存为根模型目录下的 `models/y_randomization_<ModelName>.png`

### Applicability Domain

当前 AD 逻辑使用：

- 训练集 5-fold OOF residuals
- MAD 残差尺度与有限回退
- 描述符空间 leverage

如果外部数据没有真实标签，则进入 leverage-only prediction 视图；不会使用外部 `sqrt(1-h)` 修正。
