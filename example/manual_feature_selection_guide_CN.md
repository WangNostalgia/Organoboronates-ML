# 手动特征数选择与外部预测指南

> 配套代码: `example/manual_feature_selection.py`

---

## 背景

训练完成后，SHAP-RFECV 会根据 **RKfold MAE 最小**的原则自动选择一个最优特征数。但自动选择的唯一标准是"MAE 最低"，你在亲自检查 Path Summary 后可能会发现另一种特征数综合更优。

**典型场景：**

| 特征数 | RKfold MAE | RKfold R² | LOOCV R² | 保留的特征 | 你的判断 |
|---|---|---|---|---|---|
| 7 (自动选) | 2.15 | 0.81 | 0.85 | 7个特征 | MAE 最低但特征多，不够简约 |
| 5 (你想选) | 2.52 | 0.74 | 0.79 | 5个特征 | MAE 略高但 R² 可接受，特征少2个，更可解释 |

如果你决定选 5 特征，下面的流程告诉你如何找到、加载并使用对应的模型。

---

## 第一步：从 Path Summary 读取各特征数的全部指标

训练日志控制台中每个模型结束后会打印：

```
==============================================================================================================
  SHAP-RFECV Path Summary for SVR
  Feat  RKfold MAE ± std         RKfold R² ± std        LOOCV R²   LOOCV MAE   100-spl MAE  Test R²   Test MAE
  --------------------------------------------------------------------------------------------------------------
  9     2.3400 ± 0.1500          0.7700 ± 0.0400        0.8112     2.1800      2.3100      0.8007    2.3700
  8     2.2800 ± 0.1300          0.7850 ± 0.0350        0.8201     2.0500      2.2500      0.8100    2.2800
  7     2.1500 ± 0.1100          0.8100 ± 0.0300        0.8450     1.9200      2.1200      0.8350    2.1500    ← 自动选择 (MAE最低)
  6     2.3100 ± 0.1400          0.7800 ± 0.0380        0.8180     2.1400      2.2800      0.8050    2.3200
  5     2.5200 ± 0.1700          0.7400 ± 0.0450        0.7900     2.3500      2.4800      0.7700    2.5100    ← 你觉得更好
  4     2.8800 ± 0.2000          0.6700 ± 0.0550        0.7500     2.6500      2.8500      0.7000    2.8800
  3     3.1500 ± 0.2300          0.6100 ± 0.0650        0.6900     2.9800      3.1200      0.6400    3.1500
  --------------------------------------------------------------------------------------------------------------
  ★ Auto-selected optimal: 7 features (min RKfold MAE)
```

同样的表格也保存在 `models/SVR/SVR_final_*_metrics.txt` 的 `SHAP-RFECV Path` 部分。

---

## 第二步：并排对比两个特征数（辅助决策）

使用 `compare_feature_counts()` 函数，直观看出每个指标的优劣：

```python
from example.manual_feature_selection import compare_feature_counts

compare_feature_counts('models/SVR', 7, 5)
```

输出示例：

```
======================================================================
  Manual Comparison: 7 features vs 5 features
  Model directory: models/SVR
======================================================================
  Metric                     7 feat              5 feat   Favors
  --------------------------------------------------------------------
  100-split MAE                  2.1200              2.4800   7 feat
  Test R²                        0.8350              0.7700   7 feat
  RKfold MAE                     2.1500              2.5200   7 feat
  RKfold MAE std                 0.1100              0.1700   7 feat
  RKfold R²                      0.8100              0.7400   7 feat
  LOOCV R²                       0.8450              0.7900   7 feat
  LOOCV MAE                      1.9200              2.3500   7 feat

  Features (7): ['reaction_energy', 'lumo_energy', 'VBur_C', 'pka', 'C_Polarization', 'B_s', 'NPA_charge_C']
  Features (5): ['reaction_energy', 'lumo_energy', 'VBur_C', 'pka', 'C_Polarization']
  Only in 7 feat: {'B_s', 'NPA_charge_C'}
```

7 特征在所有指标上都优于 5 特征，但这意味着你多用了 2 个特征。你可以权衡：
- **选 7 特征**如果预测精度是第一优先级
- **选 5 特征**如果你追求更简约的模型、更容易的化学解释、或担心 7 特征中存在冗余/噪声特征

---

## 第三步：加载对应特征数的最佳 checkpoint

训练过程中每个迭代都会保存一个 checkpoint（`*_iteration_N_<timestamp>.joblib`），内含该轮迭代的完整模型、scaler 和特征列表。

同一特征数可能有多个迭代的 checkpoint（因为被移除的特征不同）。`load_by_feature_count()` 会自动找到 MAE 最低的那个：

```python
from example.manual_feature_selection import load_by_feature_count

info, source_file = load_by_feature_count('models/SVR', 5)
```

输出：

```
Selected: SVR_iteration_8_20260511_120000.joblib
Features (5): ['reaction_energy', 'lumo_energy', 'VBur_C', 'pka', 'C_Polarization']
Metrics: MAE_mean=2.4800, R²_test=0.7700, MAE_test=2.5100
Best hyperparameters: {'C': 73.31, 'epsilon': 0.017, 'gamma': 8.04}
```

`info` 字典包含以下字段：

| 键 | 类型 | 说明 |
|---|---|---|
| `model` | sklearn 模型实例 | 已训练的模型 |
| `scaler_X` | MinMaxScaler | 特征缩放器（在该轮训练集上拟合） |
| `scaler_y` | MinMaxScaler(0,100) | 目标缩放器（在该轮训练集上拟合） |
| `features` | list | 该轮保留的特征名称 |
| `hyperparameters` | dict | 最优超参数 |
| `metrics` | dict | 全部性能指标 |

---

## 第四步：用选定模型预测外部数据

```python
from example.manual_feature_selection import predict_external

# 用 5 特征的模型预测外部数据
result = predict_external(info, 'external_data.csv')
```

输出：

```
External data: 20 samples from external_data.csv
Predictions saved to: external_data_predicted_5feat.csv
Predicted range: 12.34 ~ 35.67 kcal/mol
Mean prediction: 22.31 kcal/mol
```

函数会自动：
1. 从外部 CSV 中提取模型需要的特征列
2. 用 checkpoint 中保存的 `scaler_X` 缩放特征
3. 用 checkpoint 中保存的 `model` 预测
4. 用 `scaler_y` 逆变换回 kcal/mol
5. 将预测结果追加为新列并保存

---

## 第五步（可选）：一行代码加载

如果你熟悉 checkpoint 结构，可以用一行代码快速加载：

```python
import joblib, glob

model_dir, target_n = 'models/SVR', 5

mi = min(
    [joblib.load(f) for f in glob.glob(f'{model_dir}/*_iteration_*.joblib')
     if len(joblib.load(f)['features']) == target_n],
    key=lambda x: x['metrics']['mae_mean']
)

model, sX, sY, feats = mi['model'], mi['scaler_X'], mi['scaler_y'], mi['features']
```

---

## 关键提示

1. **迭代 checkpoint vs 最终模型**：
   - `*_iteration_N_*.joblib`：每轮迭代的快照，包含该轮的特征集。**用于手动选择不同特征数**
   - `*_final_*.joblib`：自动选择（或 `--force_n_features` 指定）的最终模型。**直接用这个如果你的选择与自动一致**

2. **scaler 是特定于该轮训练集的**：加载迭代 checkpoint 时，`scaler_X` 是在该轮的 `X_train`（特定特征子集）上拟合的。预测时确保传入的特征子集与 `features` 列表一致。

3. **特征顺序自动匹配**：`scaler_X.transform(X_new[features])` 中 pandas 的列索引会自动按 `features` 列表的顺序对齐，无需手动排列。

4. **如果外部数据缺特征列**：`predict_external()` 会抛出明确的 `ValueError`，列出缺少哪些特征。如果外部数据是通过相同 DFT 方法计算的，通常所有特征都有。

5. **如果不确定选哪个特征数**：用 `compare_feature_counts()` 并排对比，重点关注 RKfold MAE（PRIMARY）和特征列表的化学合理性。
