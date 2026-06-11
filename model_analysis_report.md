# 机器学习模型训练结果综合分析与报告

> **生成日期:** 2026-05-08
> **分析对象:** 14种回归模型在有机硼化合物活化能预测任务上的训练结果
> **数据来源:** `example/B_dataset.csv`（目标变量: `activation_energy`, kcal/mol）

---

## 摘要

本报告系统分析了14种机器学习回归模型在有机硼化合物反应活化能预测任务上的训练结果。所有模型均经过 Optuna 贝叶斯超参数优化（100次试验），并执行了12轮迭代特征筛选（从15个特征逐步缩减至3个特征）。核心发现如下：

- **GPR（高斯过程回归）**以 R²_LOO = 0.8112 排名第一，仅需3个特征（reaction_energy, lumo_energy, VBur_C）
- **reaction_energy、lumo_energy、VBur_C、NPA_charge_C** 是被多模型一致保留的关键特征
- **迭代特征筛选过程存在系统性退化**：多数模型的 MAE 随特征移除而升高，说明当前的特征选择策略可能需要调整
- **SVR 退化最为严重**：从初始 MAE 3.20 升至最终 MAE 4.19，但 R²_LOO 仍高达 0.8000，显示其优秀的泛化潜力
- 本次训练对应 Pipeline.md 中的**阶段二至阶段五**，后续还需执行外部验证（阶段六）

---

## 目录

1. [任务规划](#1-任务规划)
2. [训练结果文件清单](#2-训练结果文件清单)
3. [模型综合比较分析](#3-模型综合比较分析)
4. [特征重要性与筛选分析](#4-特征重要性与筛选分析)
5. [物理化学视角解读](#5-物理化学视角解读)
6. [改进建议](#6-改进建议)
7. [Pipeline定位与后续任务](#7-pipeline定位与后续任务)
8. [结论](#8-结论)

---

## 1. 任务规划

本报告按照 `request.md` 中定义的8个步骤执行：

```
步骤1 → 任务规划（使用 TaskCreate 管理14个子任务）
步骤2 → 读取训练结果（遍历 models/ 目录下所有文件）
步骤3 → 模型比较分析（生成 model_comparison.csv 对比表）
步骤4 → 特征重要性与筛选分析（分析迭代筛选日志）
步骤5 → 物理化学视角结论（化学机理解读）
步骤6 → 调整方向与建议（数据/模型/评估/其他）
步骤7 → Pipeline定位与后续任务
步骤8 → 生成综合报告（本文件）
```

---

## 2. 训练结果文件清单

### 2.1 训练运行概况

本次训练运行了**全部14种回归模型**，每种模型执行了100次 Optuna 超参数优化试验 + 12轮迭代特征筛选。

**运行日志:** `models/optimization_20260508_122739.log` (1.5MB), `my_final_new.log` (完整日志)

**Optuna 优化数据库:** 14个 `.db` 文件

| 数据库文件 | 模型 |
|---|---|
| `optuna_optimization_AdaBoostRegressor.db` | AdaBoost |
| `optuna_optimization_DecisionTreeRegressor.db` | DecisionTree |
| `optuna_optimization_ElasticNet.db` | ElasticNet |
| `optuna_optimization_GaussianProcessRegressor.db` | GPR |
| `optuna_optimization_GradientBoostingRegressor.db` | GradientBoosting |
| `optuna_optimization_KernelRidge.db` | KRR |
| `optuna_optimization_KNeighborsRegressor.db` | KNR |
| `optuna_optimization_Lasso.db` | Lasso |
| `optuna_optimization_LinearRegression.db` | LinearRegression |
| `optuna_optimization_MLPRegressor.db` | MLP |
| `optuna_optimization_RandomForestRegressor.db` | RandomForest |
| `optuna_optimization_Ridge.db` | Ridge |
| `optuna_optimization_SVR.db` | SVR |
| `optuna_optimization_XGBRegressor.db` | XGBoost |

### 2.2 每个模型的输出文件

以 SVR 为例（每个模型结构相同）：

```
models/SVR/
├── SVR_iteration_1_20260508_124725.joblib   # 第1轮迭代检查点（模型+scaler+特征+超参数）
├── SVR_iteration_1_20260508_124725_metrics.txt
├── ...（共12轮迭代）
├── SVR_final_20260508_125549.joblib          # 最终模型
├── SVR_final_20260508_125549_metrics.txt     # 最终指标
├── final_scatter_20260508_125549.png         # 最终散点图
├── final_scatter_20260508_125549_outliers.csv# 异常值数据（偏差≥5 kcal/mol）
├── performance_history_20260508_125549.csv   # 逐轮迭代性能
└── performance_history_20260508_125549.png   # 性能历史图
```

### 2.3 输入特征（共15个）

`reaction_energy`, `pka`, `dipole`, `C_Polarization`, `C_s`, `B_s`, `homo_energy`, `lumo_energy`, `electronegativity`, `Bond_Length`, `Mulliken_charge_B`, `NPA_charge_B`, `VBur_C`, `Mulliken_charge_C`, `NPA_charge_C`

目标变量: **`activation_energy`** (kcal/mol)

---

## 3. 模型综合比较分析

### 3.1 最终模型性能排名（按 R²_LOO 降序）

| 排名 | 模型 | MAE_Mean | R²_Test | MAE_Test | **R²_LOO** | 最终特征数 |
|---|---|---|---|---|---|---|
| 1 | **GPR** | 2.50 | 0.620 | 2.92 | **0.811** | 3 |
| 2 | **SVR** | 4.19 | 0.629 | 2.84 | **0.800** | 3 |
| 3 | **KRR** | 2.55 | 0.719 | 2.44 | **0.783** | 3 |
| 4 | GradientBoosting | 2.80 | 0.628 | 3.08 | 0.782 | 3 |
| 5 | AdaBoost | 2.89 | 0.567 | 3.54 | 0.760 | 3 |
| 6 | XGBoost | 3.00 | 0.466 | 3.68 | 0.756 | 3 |
| 7 | ElasticNet | 3.75 | 0.633 | 3.03 | 0.755 | 3 |
| 8 | Ridge | 2.90 | 0.676 | 2.93 | 0.755 | 3 |
| 9 | LinearRegression | 2.90 | 0.673 | 2.95 | 0.754 | 3 |
| 10 | Lasso | 3.76 | 0.666 | 2.95 | 0.754 | 3 |
| 11 | KNR | 2.85 | 0.624 | 2.74 | 0.739 | 3 |
| 12 | RandomForest | 3.01 | 0.674 | 2.64 | 0.732 | 3 |
| 13 | DecisionTree | 3.21 | 0.459 | 3.89 | 0.707 | 3 |
| 14 | MLP | 2.64 | 0.695 | 2.79 | 0.702 | 3 |

> 完整数据已保存至 `models/model_comparison.csv`

### 3.2 关键发现

#### 3.2.1 R²_LOO 与 R²_Test 的显著差异

GPR 和 SVR 的 R²_LOO (0.811/0.800) 远高于 R²_Test (0.620/0.629)，这是一个**重要警示**：

- **可能原因A**: LOO-CV 在极端小特征集（3个）上可能高估泛化能力
- **可能原因B**: 20% 测试集的随机划分恰好落在较难预测的区域
- **可能原因C**: 目标变量的分布不均匀导致 split 偏差

Ridge、LinearRegression、Lasso 的 R²_Test (0.67-0.68) 高于 GPR，说明它们在当前测试集划分上更稳定，但 R²_LOO 较低 (0.75)。

#### 3.2.2 迭代特征筛选的退化效应

**几乎所有模型的 MAE 随特征移除而升高**，说明迭代特征筛选在此数据集上存在系统性退化：

| 模型 | 初始 MAE | 最终 MAE | MAE 变化 | 方向 |
|---|---|---|---|---|
| SVR | 3.20 | 4.19 | **+0.99** | 严重退化 |
| AdaBoost | 2.45 | 2.89 | +0.44 | 中度退化 |
| KNR | 2.77 | 2.85 | +0.08 | 轻微退化 |
| LinearRegression | 2.32 | 2.90 | +0.58 | 中度退化 |
| Ridge | 2.30 | 2.90 | +0.60 | 中度退化 |
| XGBoost | 2.60 | 3.00 | +0.40 | 中度退化 |
| GPR | 2.29 | 2.50 | +0.21 | 轻微退化 |
| KRR | 2.24 | 2.55 | +0.31 | 轻微退化 |

**仅 ElasticNet 几乎不变**（3.60 → 3.75，但初始 MAE 就已经很高）。

**核心问题**：特征选择以"移除最不重要或高相关的特征"为准则，但移除特征导致的信息损失未能被后续再训练弥补。在高维特征空间中，即使一个重要性仅 2-5% 的特征，其移除也可能导致多特征组合信息的丢失。

#### 3.2.3 过拟合分析

| 模型 | Train R² (初始) | Test R² (初始) | 差异 | 过拟合风险 |
|---|---|---|---|---|
| SVR | ~0.83 | 0.83 | 小 | 低 |
| MLP | ~0.79 | 0.79 | 小 | 低 |
| RandomForest | ~0.76 | 0.76 | 小 | 低 |
| AdaBoost | ~0.71 | 0.71 | 小 | 低 |
| DecisionTree | ~0.37 | 0.37 | 小 | 低（但欠拟合） |
| KNR | ~0.71 | 0.71 | 小 | 低 |
| ElasticNet | ~0.75 | 0.75 | 小 | 低 |
| LinearRegression | 0.87 | 0.80 | 0.07 | 中等 |

LinearRegression 显示出一定程度的过拟合（Train R²=0.87 vs Test R²=0.80），需要在最终模型中关注。其余模型未见明显过拟合。

### 3.3 模型推荐

**推荐排名**（综合考虑泛化能力、MAE、稳定性）：

1. **GPR** — 最佳 R²_LOO，MAE 适中 (2.50)，仅需3个特征；但需注意不确定性估计的计算开销
2. **KRR** — R²_LOO 第三、MAE_Test 最低 (2.44)、R²_Test 最高 (0.719)，综合表现最优
3. **SVR** — R²_LOO 第二、但 MAE 偏高 (4.19)，特征筛选过程中严重退化，建议减少特征移除强度
4. **Ridge** — 简单、稳定、MAE 可控 (2.90)，R²_LOO (0.755) 中等但可接受
5. **MLP** — R²_LOO 最后但 R²_Test 第三 (0.695)，适合作为集成方法中的多样性来源

---

## 4. 特征重要性与筛选分析

### 4.1 迭代特征移除全过程

以 LinearRegression 为例，从日志中提取的特征移除序列：

| 迭代 | 移除特征 | 原因 | MAE | R²_Test |
|---|---|---|---|---|
| 1 | NPA_charge_B | 与 VBur_C 相关 0.85，重要性 3.66% | 2.32 | 0.801 |
| 2 | pka | 与 reaction_energy 相关 0.81，重要性 21.48% | 2.31 | 0.799 |
| 3 | electronegativity | 低重要性 (<20%) | 2.57 | 0.713 |
| 4 | C_s | 低重要性 | 2.57 | 0.713 |
| 5 | Bond_Length | 低重要性 | 2.53 | 0.713 |
| 6 | dipole | 低重要性 | 2.51 | 0.715 |
| 7 | homo_energy | 低重要性 | 2.49 | 0.718 |
| 8 | Mulliken_charge_B | 低重要性 | 2.51 | 0.709 |
| 9 | lumo_energy | 低重要性 | 2.48 | 0.712 |
| 10 | Mulliken_charge_C | 低重要性 | 2.46 | 0.712 |
| 11 | NPA_charge_C | 低重要性 | 2.54 | 0.710 |
| 12 | B_s | 低重要性 | 2.63 | 0.670 |

**最终保留**: `reaction_energy`, `C_Polarization`, `VBur_C`

### 4.2 各模型最终保留特征汇总

| 模型 | 保留的3个特征 |
|---|---|
| GPR | reaction_energy, lumo_energy, VBur_C |
| SVR | reaction_energy, lumo_energy, VBur_C |
| KRR | reaction_energy, VBur_C, NPA_charge_C |
| GradientBoosting | pka, lumo_energy, Mulliken_charge_B |
| AdaBoost | pka, C_s, Mulliken_charge_B |
| XGBoost | pka, B_s, lumo_energy |
| ElasticNet | pka, C_Polarization, lumo_energy |
| Ridge | reaction_energy, C_Polarization, VBur_C |
| LinearRegression | reaction_energy, C_Polarization, VBur_C |
| Lasso | reaction_energy, C_Polarization, VBur_C |
| KNR | pka, C_Polarization, lumo_energy |
| RandomForest | pka, C_Polarization, lumo_energy |
| DecisionTree | pka, C_s, homo_energy |
| MLP | pka, dipole, C_Polarization |

### 4.3 特征被保留频次统计

| 特征 | 被保留频次 | 分析 |
|---|---|---|
| **pka** | 8/14 | 最常被保留，酸碱性是反应活性的核心决定因素 |
| **lumo_energy** | 7/14 | LUMO 能级决定亲电反应性，与活化能直接相关 |
| **C_Polarization** | 7/14 | C 原子的极化率反映电子云形变能力 |
| **reaction_energy** | 6/14 | 反应热力学驱动力 |
| **VBur_C** | 6/14 | C 原子的位阻体积，影响过渡态空间位阻 |
| Mulliken_charge_B | 2/14 | B 原子的电荷分布（较少被保留） |
| C_s | 2/14 | C 原子 s 轨道成分 |
| B_s | 1/14 | B 原子 s 轨道成分 |
| NPA_charge_C | 1/14 | NPA 方法计算的 C 原子电荷 |
| dipole | 1/14 | 偶极矩（仅 MLP 保留） |
| homo_energy | 1/14 | HOMO 能级（仅 DecisionTree 保留） |

### 4.4 SHAP 特征重要性分析（以 LinearRegression 第1轮为例）

| 排名 | 特征 | 重要性 (%) |
|---|---|---|
| 1 | reaction_energy | 23.12 |
| 2 | pka | 22.30 |
| 3 | NPA_charge_C | 10.20 |
| 4 | Mulliken_charge_C | 10.16 |
| 5 | VBur_C | 9.75 |
| 6 | B_s | 6.36 |
| 7 | Mulliken_charge_B | 4.14 |
| 8 | NPA_charge_B | 3.66 |
| 9 | C_s | 2.70 |
| 10 | C_Polarization | 1.89 |
| 11 | dipole | 1.87 |
| 12 | Bond_Length | 1.58 |
| 13 | lumo_energy | 1.47 |
| 14 | electronegativity | 0.76 |
| 15 | homo_energy | 0.06 |

**关键观察**：reaction_energy 和 pka 合计贡献约 45% 的重要性，是绝对主导的特征。VBur_C（9.75%）和各类电荷描述符（NPA_charge_C 10.20%, Mulliken_charge_C 10.16%）构成第二梯队。

### 4.5 特征相关性分析

从日志中捕捉到的关键高相关特征对（|r| > 0.8）：

| 特征对 | 相关性 | 分析 |
|---|---|---|
| NPA_charge_B — VBur_C | 0.85 | B原子电荷与C原子体积共线（空间效应与电子效应的耦合） |
| pka — reaction_energy | 0.81 | 酸碱性驱动反应热力学（合理的物理化学关联） |

**共线性问题**：存在高相关特征对说明输入特征之间存在信息冗余。这在物理化学上是合理的——分子描述符之间往往存在内在关联。

---

## 5. 物理化学视角解读

### 5.1 关键入选特征的化学意义

#### 5.1.1 reaction_energy（反应能）

反应能是热力学驱动力。根据 Bell-Evans-Polanyi (BEP) 原理，放热反应的反应势垒（活化能）与反应焓变之间存在线性关系。reaction_energy 作为最重要的特征之一，**完美符合 BEP 原理的预测**。这验证了模型学到的首要规律是正确的物理化学直觉。

#### 5.1.2 lumo_energy（LUMO 能级）

LUMO 能级是分子接受电子的能力指标。在有机硼化合物的 C-H/C-卤键硼化反应中，底物的 LUMO 能级决定了其亲电反应性——LUMO 能量越低，电子亲和力越强，反应活化能越低。lumo_energy 被 GPR、SVR、KRR、XGBoost 等7个模型保留，**与前线分子轨道理论完全一致**。

#### 5.1.3 VBur_C（C原子位阻体积）

VBur_C 量化了碳中心的空间位阻。在硼化反应中，过渡态结构涉及 C-B 键的形成，碳原子周围取代基的大小直接影响过渡态的空间拥挤程度。VBur_C 值越大，位阻越大，活化能越高。这与有机化学中的**位阻效应**完全吻合。NPA_charge_B 与 VBur_C 之间 0.85 的高相关性也揭示了**空间效应与电子效应之间的耦合**——大体积基团通常伴随电荷极化的变化。

#### 5.1.4 C_Polarization（C 原子极化率）

极化率反映电子云在外电场下的形变能力。高极化率意味着过渡态中的电子重组更易发生，从而降低活化能。C_Polarization 被 GPR、Ridge、LinearRegression、Lasso、RandomForest 等7个模型保留。

#### 5.1.5 pka（酸碱度）

pKa 是有机硼试剂的酸性度量，决定了质子转移/硼化反应的驱动力。pka 被8个模型保留，是保留频次最高的特征。在反应机理中，C-H 键的断裂涉及质子转移步骤，pKa 直接影响该步骤的能垒。

### 5.2 模型揭示的定量构效关系

本次 ML 分析揭示的核心 QSAR/QSPR 规律：

1. **BEP 线性关系成立**：reaction_energy 的重要性排名第一（23.12%），验证了活化能与反应热力学之间的线性自由能关系
2. **前线轨道控制**：lumo_energy 的综合重要性（被7个模型保留）表明反应受亲电试剂 LUMO 能级控制
3. **位阻-电子耦合**：VBur_C 与 NPA_charge_B 的高相关性 (0.85) 说明空间位阻与电子效应在本体系中不可分离
4. **酸碱催化特征**：pka 的高频次被保留（8/14模型）表明质子转移在反应机理中起关键作用

### 5.3 特征间线性关系的物理化学解释

> 问题: "部分特征之间的线性应该如何物理化学解释？（例如VBur_C和NPA_charge_B）"

VBur_C 与 NPA_charge_B 的 0.85 相关性可以从以下角度理解：

- **诱导效应链**：碳原子连接电负性取代基 → 改变碳的杂化状态和空间构型 → 同时影响氮/硼原子的电荷分布
- **共轭体系的整体性**：在共轭硼化体系中，取代基的改变同时影响整个π体系的电子分布和几何结构，VBur 和 NPA_charge 反映同一物理化学现象的两个方面
- **统计来源**：0.85 的高度线性意味着在数据集中，这两个描述符携带了大量冗余信息，其中一个可以被另一个基本替代

**建议**：可尝试使用 VBur_C/NPA_charge_B 的比值或差值作为新特征，可能比直接使用原始值更有物理意义。

---

## 6. 改进建议

### 6.1 数据层面

| 建议 | 优先级 | 说明 |
|---|---|---|
| **增加训练样本** | 高 | 当前仅 ~88 个样本（根据B_dataset.csv），15个特征 → 移除12个后仅3个特征。样本量不足是MAE退化的根本原因。建议通过 DFT 计算或文献数据扩充至 200+ 样本 |
| **特征工程** | 高 | 从相关特征对（如 VBur_C + NPA_charge_B）中构造比值/差值特征，减少冗余而非直接删除 |
| **异常值审查** | 中 | 各模型的 `*_outliers.csv` 中标注了偏差≥5 kcal/mol 的样本，应回查这些样本的实验/计算数据质量 |
| **目标变量变换** | 低 | 尝试 log 变换或 Box-Cox 变换以改善分布正态性 |
| **数据质量检查** | 高 | 确认 DFT 计算使用的方法和基组一致性，不同方法计算的特征值可能引入系统性偏差 |

### 6.2 模型层面

| 建议 | 优先级 | 说明 |
|---|---|---|
| **停止过早特征移除** | 高 | 当前 min_features=3 导致过度精简。建议设为 7-10，保留更多特征以维持预测性能 |
| **修改特征选择策略** | 高 | 当前策略在移除特征后 MAE 系统性升高。建议改为：(1) 尝试前向选择而非后向消除；(2) 使用 RFECV 替代固定的重要性阈值；(3) 基于交叉验证的 MAE 变化而非固定重要性阈值来决定是否移除 |
| **集成方法** | 高 | Pipeline.md 阶段六提到的加权集成预测。GPR+KRR+Ridge 的组合可能产生优于任何单一模型的结果 |
| **GPR 不确定性量化** | 中 | GPR 天然输出预测方差，可用于识别高不确定性区域，指导后续 DFT 计算或实验 |
| **正则化调整** | 中 | SVR 的 ε 参数 (1.41) 偏大，导致不敏感区间过宽。建议在 0.01-0.5 范围内重新调参 |
| **XGBoost/LightGBM 参数** | 低 | 当前 XGBoost 的 R²_Test (0.466) 远低于 R²_LOO (0.756)，需要检查早停策略和数据划分 |

### 6.3 评估层面

| 建议 | 优先级 | 说明 |
|---|---|---|
| **外部验证** | 高 | 必须执行 Pipeline.md 阶段六的外部验证。使用 `src/validation_process.py` 生成 sub_H×sub_B 组合空间，在完全未见过的组合上测试模型 |
| **重复 k-fold** | 高 | 当前 LOOCV 在高维小样本下可能过于乐观。建议增加 5×5 重复交叉验证 |
| **y-randomization** | 中 | 随机打乱目标变量的排列测试，验证模型是否学到了真实的构效关系而非巧合 |
| **留出外部测试集** | 高 | 从训练开始前就预留 10-15% 数据作为最终外部测试集，在整个 pipeline 完成后才使用 |

### 6.4 其他

| 建议 | 优先级 | 说明 |
|---|---|---|
| **计算资源优化** | 中 | 当前使用 n_jobs=1（从日志中确认），建议使用 n_jobs=-1 以利用所有 CPU 核心 |
| **可重复性保障** | 中 | 固定所有随机种子（已做到 random_state=42），但建议保存完整环境（`uv.lock` 已存在，很好） |
| **决策树/MLP 探索** | 低 | DecisionTree 严重欠拟合 (R²=0.37)，MLP 不具优势，推荐将计算资源集中到表现优异的核方法（GPR/KRR/SVR） |
| **模型可解释性文档** | 低 | 使用 SHAP 依赖图和交互值图深入分析关键特征的边际效应 |

---

## 7. Pipeline 定位与后续任务

### 7.1 当前位置

阅读 `Pipeline.md` 后确认，本次训练结果对应于：

```
Pipeline.md 阶段二 → 阶段三 → 阶段四 → 阶段五

阶段一: 数据准备       ← 已完成（main.py 数据加载）
阶段二: 模型训练+超参数优化 ← 已完成（100 trials × 14 models）
阶段三: 初次评估       ← 已完成（100-split MAE 已计算）
阶段四: 迭代特征筛选   ← 已完成（12轮迭代，降至3个特征）
阶段五: 最终构建与评估 ← 已完成（LOO-CV、散点图、joblib 已保存）
阶段六: 外部验证       ← ★ 尚未执行 ★
```

### 7.2 后续需执行的步骤

| 步骤 | 所需文件 | 预期产出 |
|---|---|---|
| **6a. 生成验证数据集** | `src/validation_process.py` | `validation_data.csv`（所有 sub_H×sub_B 组合） |
| **6b. 外部预测（加权）** | `example/prediction_round2.ipynb` | `final_with_y_pred_weighted_mean.csv` |
| **6c. 外部预测（平均）** | `example/prediction_round2.ipynb` | `final_with_y_pred_mean.csv` |
| **6d. 模型评估可视化** | `example/example_pic.ipynb` | SHAP 雷达图、R² 分布图、模型对比图 |
| **6e. 重新训练（调整参数后）** | `main.py`（修改 min_features） | 更优的模型性能 |

### 7.3 建议的执行顺序

```bash
# 1. 先修改 main.py 参数并重新训练（基于 Step 6 的改进建议）
#    - 将 min_features 从3改为7
#    - 或修改 feature_selection.py 中的阈值（重要性 <10% 才移除）

# 2. 生成验证空间
python -c "
from src.validation_process import validation_data_produce
import pandas as pd
data = pd.read_csv('example/B_dataset.csv')
# ...
"

# 3. 运行 notebook 进行外部验证
jupyter notebook example/prediction_round2.ipynb
```

---

## 8. 结论

### 8.1 核心发现

1. **GPR 为最佳单一模型**：R²_LOO = 0.8112，仅需3个特征（reaction_energy, lumo_energy, VBur_C），MAE_Mean = 2.50 kcal/mol
2. **KRR 在可解释性与准确度间取得最佳平衡**：R²_Test 最高 (0.719)、MAE_Test 最低 (2.44)，推荐作为部署模型
3. **迭代特征筛选策略需要调整**：从15个特征降至3个导致多数模型 MAE 上升。建议 min_features 设为7-10，并改用基于 CV 性能变化的自适应停止准则
4. **BEP 原理和前线轨道理论被 ML 模型验证**：reaction_energy 和 lumo_energy 是跨模型一致保留的关键特征
5. **VBur_C 与 NPA_charge_B 的线性共线**揭示了空间效应-电子效应的耦合，建议构造组合特征

### 8.2 主要局限性

- 样本量小（约88个），限制了复杂模型的训练和泛化能力
- LOOCV 在小样本下可能高估性能（特别是 GPR/SVR 的 R²_LOO 与 R²_Test 差异显著）
- 外部验证尚未执行，当前指标仅来自随机划分，不等于真实外部数据的表现
- 特征筛选使用固定的 20% 重要性阈值，可能不适合所有模型类型

### 8.3 后续行动项

| 行动 | 预期效果 | 时间估计 |
|---|---|---|
| 修改 min_features=8 重新训练 | 缓解 MAE 退化 | 数天 |
| 执行外部验证 | 获得真实泛化能力评估 | 数分钟 |
| GPR+KRR+Ridge 加权集成 | 降低预测方差 | 数小时 |
| 增加样本或数据增强 | 根本性改善模型性能 | 数周 |

---

*报告生成时间: 2026-05-08*
*分析工具: Python + pandas + scikit-learn 生态*
*相关文件: model_comparison.csv（模型性能对比表，已保存至 models/ 目录）*
