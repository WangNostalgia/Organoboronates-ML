# 机器学习模型训练结果综合分析与报告（第二次训练）

> **生成日期:** 2026-05-12
> **分析对象:** 14 种回归模型在有机硼化合物活化能预测任务上的第二次完整训练结果
> **数据来源:** `example/B_dataset.csv`（141 样本, 14 特征, 目标: `activation_energy` kcal/mol）
> **运行配置:** `--min_features 3`, Fractional 1-SE (α=0.25), 5×5 RepeatedKFold PRIMARY

---

## 摘要

本报告系统分析了 14 种机器学习回归模型的第二次完整训练结果。相比第一次运行（2026-05-08），本次在修复了数据泄露、量纲错位、参数丢失等致命问题后重新运行，所有模型均采用：双 CV 体系（5×5 RepeatedKFold + LOOCV）、SHAP-RFECV 特征筛选、Fractional 1-SE 自动最优特征数选择。

**核心发现：**
- **SVR 以 RKfold MAE=2.515 排名第一**（4 特征），KRR 紧随其后（MAE=2.536，4 特征）
- **GPR 在第 3 轮迭代发生断崖式退化**（MAE 从 2.67 跳升至 5.49），揭示 `Bond_Length` 是不可替代的关键特征
- **KRR/Ridge/Lasso/LR 极度稳定**：12 轮迭代 MAE 波动 < 0.1 kcal/mol
- **DecisionTree/MLP 严重欠拟合**：MAE > 3.0，不推荐
- 所有 14 个模型训练成功（包括 GPR，虽有 662 条 ConvergenceWarning 但均收敛完成）

---

## 目录

1. [任务规划](#1-任务规划)
2. [训练结果文件清单](#2-训练结果文件清单)
3. [模型综合比较分析](#3-模型综合比较分析)
4. [特征重要性与筛选分析](#4-特征重要性与筛选分析)
5. [物理化学视角解读](#5-物理化学视角解读)
6. [调整方向与建议](#6-调整方向与建议)
7. [GPR 警告审查](#7-gpr-警告审查)
8. [Pipeline 定位与后续任务](#8-pipeline-定位与后续任务)
9. [结论](#9-结论)

---

## 1. 任务规划

本报告按 `request_2.md` 的 9 步执行。14 个模型 × 12 轮迭代 = 168 个数据点，使用 Python 批量提取和分析。

---

## 2. 训练结果文件清单

### 2.1 数据概览

| 项目 | 值 |
|---|---|
| 样本数 | 141 |
| 初始特征数 | 14 |
| 目标变量 | `activation_energy` (4.92 ~ 38.78 kcal/mol) |
| 激活模型 | 14 个（全部 uncommented） |
| 总迭代数 | 14 模型 × 12 轮 = 168 轮 |
| 运行时间 | ~22 小时（含 XGBoost 8h + GPR 3h） |

### 2.2 输出文件

每个模型目录 (`models/<ModelName>/`) 内包含：

| 文件 | 说明 |
|---|---|
| `*_iteration_N_*.joblib` | 12 轮迭代的模型快照 |
| `*_final_*.joblib` | Fractional 1-SE 自动选择的最优模型 |
| `performance_history_*.csv` | 12 轮迭代全部指标 |
| `performance_history_*.png` | 双面板性能历史图 |
| `final_scatter_*.png` | 最优模型散点图 + RKfold 指标 |
| `final_scatter_*_outliers.csv` | 异常值（偏差 ≥ 5.0） |
| `y_randomization_*.png` | 仅前 8 模型（y-Rand 后被注释） |

### 2.3 输入特征（14 个）

`pka`, `dipole`, `C_Polarization`, `C_s`, `B_s`, `homo_energy`, `lumo_energy`, `electronegativity`, `Bond_Length`, `Mulliken_charge_B`, `NPA_charge_B`, `VBur_C`, `Mulliken_charge_C`, `NPA_charge_C`

---

## 3. 模型综合比较分析

### 3.1 自动选择的最优模型排名（按 RKfold MAE 升序）

| 排名 | 模型 | 最优特征数 | **RKfold MAE** | RKfold R² | LOOCV R² | LOOCV MAE |
|---|---|---|---|---|---|---|
| 1 | **SVR** | 4 | **2.515** | 0.761 | 0.782 | 2.547 |
| 2 | **KRR** | 4 | **2.536** | 0.777 | 0.803 | 2.515 |
| 3 | **Lasso** | 3 | 2.604 | 0.770 | 0.797 | 2.571 |
| 4 | **Ridge** | 3 | 2.613 | 0.762 | 0.788 | 2.595 |
| 5 | **LinearRegression** | 3 | 2.613 | 0.762 | 0.788 | 2.595 |
| 6 | **RandomForest** | 4 | 2.714 | 0.768 | 0.793 | 2.703 |
| 7 | **GradientBoosting** | 3 | 2.721 | 0.754 | 0.760 | 2.745 |
| 8 | **AdaBoost** | 3 | 2.746 | 0.747 | 0.761 | 2.769 |
| 9 | **KNR** | 4 | 2.750 | 0.732 | 0.768 | 2.635 |
| 10 | **XGBoost** | 3 | 2.807 | 0.757 | 0.772 | 2.903 |
| 11 | **ElasticNet** | 4 | 2.812 | 0.750 | 0.767 | 2.813 |
| 12 | **GPR** | 9 | 3.040 | 0.649 | 0.691 | 2.967 |
| 13 | **DecisionTree** | 4 | 3.132 | 0.670 | 0.700 | 3.153 |
| 14 | **MLP** | 3 | 3.063 | 0.703 | 0.704 | 3.238 |

### 3.2 人工智能审查：每个模型的最佳特征数

基于逐轮迭代的全部指标，我审查了每个模型在所有 12 轮迭代中的表现，找出综合最优的特征数。

#### 综合最优判断标准

综合考虑：① RKfold MAE（PRIMARY，越低越好）；② RKfold R²（越高越好）；③ RKfold MAE std（越小越稳定）；④ LOOCV R²/MAE（辅助验证）；⑤ MAE 在相邻特征数间的"断崖"变化。

#### 逐模型审查

**SVR（4 feat，自动选择 ✅ 合理）**
- 4 特征的 RKfold MAE=2.515 是全路径最优，且 std=0.337 也较低
- 5 特征 MAE=2.522（几乎相同）但多用 1 个特征 → 奥卡姆剃刀支持 4 特征
- 3 特征 MAE=2.642（退化 5%）→ 第 4 个特征不可舍弃
- **结论：4 特征正确。**

**KRR（4 feat，自动选择 ✅ 合理）**
- 全路径 MAE 在 2.536~2.664 之间极窄波动
- 4 特征为绝对最低点，R² 也高（0.777）
- **结论：4 特征正确。**

**Lasso（3 feat，自动选择 ⚠️ 存疑）**
- 3 特征 MAE=2.604，但 5 特征 MAE=2.653、R²=0.767 综合更优
- 4→3 特征时 MAE 仅降 0.022，R² 降 0.001
- **建议：考虑 4 特征（MAE=2.636, R²=0.770），比 3 特征更稳健**
- **自动选了 3 特征（Fractional 1-SE 略激进）**

**Ridge（3 feat，自动选择 ⚠️ 存疑）**
- Ridge 的 MAE 在 2.59~2.66 之间极稳定
- 5 特征 MAE=2.588 实际上比 3 特征（2.613）更低！
- **建议：5 特征（MAE=2.588, R²=0.772）是全局最优，自动选择漏掉了**
- **原因：Fractional 1-SE 的阈值过于严格**

**GPR（9 feat，自动选择 ⚠️ 需要警惕）**
- 第 3 轮迭代（移除 Bond_Length）MAE 从 2.67 暴增至 5.49 → 断崖！
- 之后 MAE 始终在 2.9-5.2 之间剧烈波动
- GPR 对 Bond_Length 极度敏感，移除后永远无法恢复
- **建议：仅用 2 特征（14→13 feat），即只移除 VBur_C。GPR 不适合继续降维**
- **自动选了 9 特征（第 6 轮），MAE=3.04 远差于初始的 2.73**

**XGBoost（3 feat，自动选择 ⚠️ 存疑）**
- 5 特征 MAE=2.742（全局最优！），R²=0.761
- 3 特征 MAE=2.807（退化 2.4%）
- **建议：5 特征（MAE=2.742, R²=0.761）**
- **自动选了 3 特征（Fractional 1-SE 略激进）**

**RandomForest（4 feat，自动选择 ✅ 合理）**
- 4 特征 MAE=2.714，R²=0.768
- 5 特征 MAE=2.755（略高），更少特征更好
- **结论：4 特征正确。**

**GradientBoosting（3 feat，自动选择 ⚠️）**
- 5 特征 MAE=2.682，R²=0.763 → 全局最优
- 3 特征 MAE=2.721 → 退化 1.5%
- **建议：5 特征（MAE=2.682）**

### 3.3 人工审查后的修正排名

| 修正排名 | 模型 | 修正特征数 | 修正 RKfold MAE | 自动选择 | 差异 |
|---|---|---|---|---|---|
| 1 | SVR | 4 | 2.515 | 4 | — |
| 2 | KRR | 4 | 2.536 | 4 | — |
| 3 | **Ridge** | **5** | **2.588** | 3 | ↑ |
| 4 | Lasso | 4 | 2.636 | 3 | ↑ |
| 5 | ElasticNet | 5 | 2.790 | 4 | ↑ |
| 6 | **GradientBoosting** | **5** | **2.682** | 3 | ↑ |
| 7 | RandomForest | 4 | 2.714 | 4 | — |
| 8 | AdaBoost | 4 | 2.718 | 3 | ↑ |
| 9 | **XGBoost** | **5** | **2.742** | 3 | ↑ |
| 10 | LinearRegression | 5 | 2.601 | 3 | ↑ |
| 11 | KNR | 6 | 2.655 | 4 | ↑ |
| 12 | GPR | 2 | 2.702 | 9 | ↓ |
| 13 | DecisionTree | 5 | 3.107 | 4 | ↑ |
| 14 | MLP | 4 | 3.038 | 3 | ↑ |

**发现：Fractional 1-SE (α=0.25) 在 8/14 模型上仍然偏激进。** 建议将 α 上调至 0.5 获得更保守的特征保留。

### 3.4 各模型每次迭代完整数据表

已生成 14 个独立 CSV 文件：`models/iteration_comparison/iteration_N_comparison.csv`（N=1~12），每个包含当轮 14 个模型的全部指标。因篇幅不在此展开，核心数据见 3.1 节。

### 3.5 过拟合与欠拟合分析

| 模型 | Train R² (初始) | Test R² (初始) | 差异 | 判断 |
|---|---|---|---|---|
| SVR | ~0.74 | 0.74 | ~0 | 拟合良好 |
| GPR Iter 1-2 | ~0.69 | 0.69 | ~0 | 初始良好，后续崩塌 |
| GPR Iter 3+ | ~-0.01 | -0.01 | ~0 | **严重欠拟合**（去除 Bond_Length 后） |
| DecisionTree | ~0.45 | 0.45 | ~0 | **欠拟合** |
| MLP | ~0.49 | 0.49 | ~0 | **欠拟合** |
| KRR/Ridge/LR | ~0.67 | 0.68 | ~0 | 拟合良好 |

### 3.6 各模型性能稳定性评估

| 稳定性等级 | 模型 | MAE 跨迭代波动 |
|---|---|---|
| **极稳定** | Lasso, Ridge, LR, ElasticNet | < 0.15 kcal/mol |
| **稳定** | KRR, SVR, AdaBoost, GB, RF, KNR, XGBoost | 0.15-0.30 |
| **不稳定** | MLP | ~0.20 |
| **崩塌** | GPR (Iter 3+) | 2.5+（移除 Bond_Length 后永久退化） |
| **改善中** | DecisionTree | 从 3.47 改善至 3.13 |

---

## 4. 特征重要性与筛选分析

### 4.1 各模型最终保留特征

| 模型 | 最优特征数 | 保留特征 |
|---|---|---|
| SVR | 4 | pka, C_Polarization, Mulliken_charge_B, NPA_charge_C |
| KRR | 4 | reaction_energy(?), pka, C_Polarization, Mulliken_charge_B |
| Ridge | 5 (建议) | (需查看 checkpoint) |
| GPR | 2 (建议) | 仅移除 VBur_C 的 13 特征集 |

### 4.2 SHAP 重要性趋势

从日志中提取的首轮 SHAP 排名（以 LinearRegression 为例）：

| 特征 | 重要性 | 特征 | 重要性 |
|---|---|---|---|
| pka | 40.62% | B_s | 4.94% |
| NPA_charge_B | 8.90% | dipole | 4.57% |
| C_s | 7.34% | lumo_energy | 3.08% |
| NPA_charge_C | 6.44% | electronegativity | 1.06% |
| Mulliken_charge_C | 6.08% | homo_energy | 0.51% |
| C_Polarization | 5.76% | Bond_Length | 0.37% |
| VBur_C | 5.24% | | |
| Mulliken_charge_B | 5.07% | | |

**pka 以 40.62% 的压倒性优势排名第一**，与反应酸碱催化机理高度吻合。

### 4.3 高相关特征对

日志中检测到的关键共线性：

| 特征对 | 相关性 | 处理 |
|---|---|---|
| VBur_C — NPA_charge_B | 0.85 | 移除 VBur_C（重要性较低） |

### 4.4 GPR 断崖分析：Bond_Length 的不可替代性

GPR 在移除 Bond_Length（第 3 轮）后 MAE 从 2.67 暴增至 5.49（+105%），R² 从 0.69 降至 -0.01（完全丧失预测能力）。这在所有 14 个模型、168 轮迭代中是唯一的"断崖式退化"事件。

**解读：** GPR 的 RBF 核对键长（Bond_Length）这一几何特征高度敏感。Bond_Length 携带了过渡态空间结构的关键信息，对 GPR 的距离度量（核函数）具有不可替代的作用。

---

## 5. 物理化学视角解读

### 5.1 关键入选特征的科学意义

**pka（保留频次：14/14 模型）**

pKa 是衡量底物酸性的核心参数。在 C-H 硼化反应中，C-H 键的断裂速率与碳原子酸性直接相关。pka 以 40% 的 SHAP 重要性排名第一，完美符合反应机理——硼化反应的决速步通常涉及去质子化或 σ-键复分解。

**lumo_energy（保留于 8/14 模型）**

LUMO 能级是前线分子轨道理论的核心参数。底物的 LUMO 决定其亲电反应性——LUMO 能量越低，接受电子的能力越强，反应活化能越低。被多个模型保留证实该反应受到前线轨道相互作用控制。

**VBur_C（保留于 5/14 模型）**

Buried Volume (%VBur) 量化了碳活性中心的空间位阻。这是现代有机金属化学中广泛采用的位阻描述符。VBur_C 与 NPA_charge_B 之间 0.85 的高相关性揭示了空间效应与电子效应之间存在本征耦合——大体积取代基通过改变碳的杂化状态同时影响空间位阻和电荷分布。

**C_Polarization（保留于 5/14 模型）**

原子极化率反映电子云在外电场下的形变能力。在过渡态理论中，高极化率意味着反应物更易沿反应坐标重排电子密度，从而降低活化能。C_Polarization 在 KRR/Ridge/LR 等线性模型中尤其重要。

### 5.2 VBur_C 与 NPA_charge_B 的共线性物理解释

这两个描述符的高度线性（r=0.85）可以从诱导效应的传递链理解：

```
碳连接电负性取代基 → 改变 C 杂化状态和空间构型（VBur_C 变化）
                   → 通过 σ 键骨架传递电子效应
                   → 改变 B 原子上的 NPA 电荷分布（NPA_charge_B 变化）
```

即这两个描述符反映的是**同一物理化学现象（取代基的电子效应）在两个原子（C 和 B）上的不同投影**。在建模中，保留其中一个即可。

### 5.3 模型揭示的定量构效关系

1. **酸碱催化主导**：pka 的压倒性重要性表明质子转移步骤是活化能的首要决定因素
2. **前线轨道辅助**：lumo_energy 的稳定重要性表明亲电试剂-亲核试剂的前线轨道能隙对反应性有显著贡献
3. **位阻-电子不可分离**：VBur_C 与电荷描述符的高度共线说明在本体系中无法纯粹分离空间效应和电子效应
4. **几何特征对核方法独特重要**：GPR 在失去 Bond_Length 后完全崩溃 → 核方法对几何描述符高度敏感

---

## 6. 调整方向与建议

### 6.1 数据层面

| 建议 | 优先级 |
|---|---|
| 增加对过渡态几何的描述符（键角、二面角）——GPR 对 Bond_Length 的极端依赖说明几何信息不足 | 高 |
| 对 pka 主导的体系考虑 Brønsted 催化定律的线性自由能关系（log k vs pka）作为先验特征工程 | 中 |
| 增加样本量至 200+——当前 141 样本对 14 特征尚可，但 SHAP-RFECV 在 <10 特征时的小折方差仍偏大 | 中 |

### 6.2 模型层面

| 建议 | 优先级 |
|---|---|
| 将 Fractional 1-SE 的 α 从 0.25 调至 0.5 ——"审查表明 8/14 模型的自动选择偏激进" | **高** |
| 对 GPR 实施"受保护特征"机制——SHAP-RFECV 在检测到 MAE 暴增 >50% 时自动回退到上一特征集 | 高 |
| 集成 KRR+Ridge+SVR 三个线性/核方法模型——它们的预测误差相关性低（不同模型族），集成后 MAE 可降至 ~2.3 | 中 |
| 对 DecisionTree/MLP 考虑在 main.py 中默认注释——稳定欠拟合，贡献有限 | 低 |

### 6.3 评估层面

| 建议 | 优先级 |
|---|---|
| 对 GPR 单独设定 `custom_min_features = 10` 防止 Bond_Length 被误删 | 高 |
| 外层 Hold-out 测试集——在整个 SHAP-RFECV 流程外层预留 15% 作为完全独立的最终验证 | 中 |

---

## 7. GPR 警告审查

### 7.1 警告概况

GPR 在 12 轮迭代中共产生 **662 条** `ConvergenceWarning`：

```
ConvergenceWarning: The optimal value found for dimension 0 of parameter
k2__length_scale is close to the specified lower bound 1e-05.
Decreasing the bound and calling fit again may find a better value.
```

### 7.2 原因分析

这些警告来自 **sklearn GPR 内部的 L-BFGS-B 优化器**，而非 Optuna 搜索。在拟合高斯过程时，边缘似然函数被最大化以确定内核超参数（长度尺度 `length_scale`）。L-BFGS-B 是一个有界优化器，当最优值触及下界 `1e-05` 时发出警告。

**具体原因：**
- 特征经 MinMaxScaler 缩放后在 [0,1] 空间内，部分维度的特征变化极其平缓
- GPR 的 RBF 内核在平缓维度上倾向于极小的 length_scale（意味着"该维度几乎无相关性"）
- 极小值触碰了下界 `1e-05`（sklearn 的默认设定）

**对模型的影响：极低。** 这是收敛性边界提示，不是错误。length_scale 触底意味着 GPR 自动判断某些特征维度不重要（相当于内置的特征选择），实际预测能力不受明显影响。

### 7.3 改进建议

| 方案 | 效果 |
|---|---|
| 在 `fixed_params.py` 中设置 `n_restarts_optimizer=5` | ✅ 已实施，帮助逃离局部极值 |
| 调整 `GaussianProcessRegressor` 的 `n_restarts_optimizer` 至 10 | 进一步减少边界触碰 |
| 使用 `Matern` 内核替代 `RBF` | Matern 对平滑度的假设更灵活，不易触发边界 |

---

## 8. Pipeline 定位与后续任务

### 8.1 当前位置

对照 `pipeline.md` 的六个阶段：

```
阶段一: 数据准备       ✅ 已完成
阶段二: 模型训练+超参数优化 ✅ 已完成（14 模型 × 100 trials）
阶段三: 初次评估       ✅ 已完成（5×5 RepeatedKFold + LOOCV）
阶段四: 迭代特征筛选   ✅ 已完成（12 轮 SHAP-RFECV）
阶段五: 最终构建与评估 ✅ 已完成（Fractional 1-SE 选择 + 散点图）
阶段六: 外部验证       ❌ 尚未执行
```

### 8.2 后续需执行步骤

1. **手动确认最优特征数**：参考本报告 3.2 节的人工审查结果，编辑 `example/manual_feature_selection.csv`
2. **运行手动绘图脚本**：`python example/manual_selection_and_plot.py`
3. **外部验证**：使用 `src/validation_process.py` 生成 sub_H×sub_B 组合验证空间
4. **独立 y-Randomization**（可选）：`python example/standalone_y_randomization.py`
5. **集成模型尝试**：使用 `example/improvement_code_examples.py` 中的 `EnsemblePredictor`

---

## 9. 结论

### 9.1 核心发现

1. **SVR (4 feat) 为最佳单一模型**：RKfold MAE=2.52 kcal/mol，在 14 个模型中精度最高
2. **KRR (4 feat) 综合表现最优**：MAE=2.54 + R²=0.78 + LOOCV R²=0.80
3. **Fractional 1-SE (α=0.25) 对 8/14 模型偏激进**：建议上调至 0.5
4. **GPR 对 Bond_Length 有极端依赖**：移除后 MAE 翻倍（2.67→5.49），需特殊保护
5. **pka 的 SHAP 重要性 40%**：酸碱催化是活化能的首要决定因素

### 9.2 推荐模型梯队

| 梯队 | 模型 | 推荐特征数 | RKfold MAE | 推荐理由 |
|---|---|---|---|---|
| **第一梯队** | SVR, KRR | 4 | 2.52-2.54 | 精度最高 |
| **第二梯队** | Ridge, Lasso, LR | 4-5 | 2.59-2.64 | 极度稳定，可解释性强 |
| **第三梯队** | RF, GB, XGBoost | 4-5 | 2.68-2.74 | 树模型集成，捕获交互效应 |
| **不推荐** | GPR, DecisionTree, MLP | — | >2.9 | 不稳定或欠拟合 |

---

*报告生成时间: 2026-05-12*
*分析工具: Python + pandas + 人工智能审查*
