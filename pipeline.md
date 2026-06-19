# 机器学习流程说明：稳定有机硼酸酯活化能预测

本文档按当前代码实现说明项目的六个核心阶段。它描述的是“现在仓库实际如何运行”，不是历史版本的旧流程。

## 总览

主入口是 `main.py`，当前默认启用的模型为：

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

整个训练流程围绕一个固定原则展开：

- 先在入口处做一次 development / final test 的固定 80/20 划分
- 所有模型共享同一组样本边界
- 所有选择行为都发生在 development 集上
- final test 只在最后锁定后评估一次

## 阶段 1：数据准备与固定划分

`main.py` 会：

1. 读取 `example/B_dataset.csv`
2. 删除全空列
3. 选择数值列
4. 用 `activation_energy` 作为目标列

随后 `iterative_optimization()` 在入口处做一次：

- `test_size=0.2`
- `random_state=40`

划分结果：

- Development 集（80%）：调参、SHAP、特征数路径、内部 5×5 RepeatedKFold、LOOCV、100-split stability
- Final test 集（20%）：最终一次性评估

这意味着测试集不会参与任何模型选择。

## 阶段 2：development-only 超参数优化

`src/train_and_evaluate.py` 的职责是：在 development 数据上挑选当前特征子集的最佳参数。

当前实现要点：

- Optuna 目标函数使用内部 5-fold MAE
- Study 按每次运行在内存中完成
- Ridge 和 Lasso 使用显式折内 alpha 循环
- 每一折都单独拟合 `MinMaxScaler`
- MAE 总是在逆变换回 kcal/mol 后计算

## 阶段 3：development 指标记录

每次训练会记录三类 development 指标：

| 指标层级 | 位置 | 含义 |
|---|---|---|
| Internal CV | `internal_cv` | 5×5 RepeatedKFold 结果，用于路径选择 |
| Stability | `stability` | development 集内部 100 次随机划分稳健性 |
| LOO | `loo` | development 集上的辅助敏感性/文献对照指标 |

主指标与次级指标的分工如下：

- 主指标：`test_mae`、`test_r2`
- 次级指标：`internal_cv.*`、`stability.*`、`loo.*`

兼容性别名仍然会保留在部分 checkpoint 中，但语义以上表为准。

## 阶段 4：SHAP 驱动的迭代特征消除

除 GPlearn 外，其余模型都走迭代删特征路径。

每轮会执行：

1. 用当前特征子集完成 development-only 调参
2. 保存 development-only iteration checkpoint
3. 根据 SHAP-RFECV 逻辑删除一个特征

删除优先级：

1. 如果存在高相关特征对，删除其中 SHAP 更弱的那个
2. 如果不存在高相关对，删除全局最不重要特征

模式切换：

- 特征较多：单次拟合 SHAP，偏速度
- 特征较少：5-fold 共识 SHAP-RFECV，偏稳健

停止条件：

- 到达 `min_features`
- 或者没有特征再适合删除

`--min_features` 当前默认值是 `5`。

## 阶段 5：特征数选择与最终测试

特征路径跑完之后：

### 自动选择模式

默认会从已评估路径中选择 development internal CV 表现最优的点，并配合 fractional 1-SE 风格阈值偏向更小特征数。

### 强制选择模式

如果传入 `--force_n_features`：

- 不会跳过路径评估
- 仍然先把路径完整评估到 `--min_features`
- 然后从已评估路径中选出指定的精确特征数

最终模型构建顺序：

1. 锁定特征数
2. 在全部 development 样本上重新拟合 scaler 与 estimator
3. 在 untouched final test 上只评估一次

最终主结果：

- `test_mae`
- `test_r2`

## 阶段 6：保存、外部验证与后续分析

### 检查点保存

每轮会保存 iteration checkpoint，结束时保存 final checkpoint。

结构示意：

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

final checkpoint 会保存：

- 拟合后的模型
- `scaler_X`
- `scaler_y`
- 最终特征顺序
- 完整超参数
- primary / secondary 指标
- split 元数据

### 检查点加载规则

`src.external_validation.load_model()` 当前规则：

- 同时搜索 final 与 iteration checkpoint
- 默认要求精确特征数匹配
- 精确匹配时 final 优先于 iteration
- 同类候选按文件名时间戳选最新
- 只有显式 `allow_closest=True` / `--allow-closest` 才允许最近邻回退

`example/load_checkpoint_guide.py` 已按这个规则提供示例。

### y-randomization

自动全流程 y-randomization 仍然关闭，原因是运行成本高。

当前支持方式：

- `example/standalone_y_randomization.py`

它会：

- 读取已锁定 checkpoint
- 让原始标签与置乱标签复用同一组预计算 5×5 RepeatedKFold splits
- 输出修正后的有限置换 p 值 `(b + 1) / (m + 1)`

### Applicability Domain

当前 AD 逻辑使用：

- 训练集 5-fold OOF 残差
- MAD 残差尺度及有限回退
- 描述符空间 leverage

如果外部数据没有真实标签，则进入 prediction-only 模式：

- 不计算残差阈值判断
- Williams 图退化为 leverage-only 视图

## 推荐命令

训练：

```bash
python main.py --n_trials 100 --min_features 5
```

快速测试：

```bash
python main.py --n_trials 20 --min_features 5
```

帮助：

```bash
python main.py --help
```

y-randomization：

```bash
python example/standalone_y_randomization.py
```

外部验证：

```bash
python src/external_validation.py --model SVR --data external.csv
python src/external_validation.py --model SVR --n_features 5 --data external.csv
```

## 关键结论

如果只记住三件事，请记住：

1. 所有模型选择只发生在 development 集
2. `test_mae` / `test_r2` 才是最终主指标
3. 旧的阈值式 CLI/工作流概念已经移除
