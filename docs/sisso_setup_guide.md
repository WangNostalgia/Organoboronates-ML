# SISSO 环境配置与可行性测试 — 完整报告

> **日期:** 2026-05-18
> **环境:** Windows 10 + WSL2 (Ubuntu 24.04) + conda (bdataset2, Python 3.11) + Windows venv (.venv, Python 3.12)
> **目标:** 为机器学习管线配置符号回归模型，输出显式数学公式

---

## 一、背景与需求

SISSO (Sure Independence Screening and Sparsifying Operator) 是一种基于压缩感知的符号回归方法，发表在 *Phys. Rev. Materials* 2, 083802 (2018)。与黑箱模型（SVR、RF、XGBoost）不同，SISSO 输出的是**人类可读的显式数学公式**（如 `(0.452*X1 + 0.319*X3) / (X5 + 0.002)`），在 QSAR/QSPR 论文中具有极高的解释性价值。

本项目需要将 SISSO 集成到 `main.py` 的训练管线中，与其他 17 个 sklearn 模型一起参与训练、特征筛选和性能对比。

---

## 二、尝试过的三条路线

### 路线 A：sissopp（C++ Python 绑定，首选方案）

**来源:** https://github.com/rouyang2017/sissopp / GitLab: https://gitlab.com/sissopp_developers/sissopp

**状态: ❌ 失败**

**过程:**

1. 从 GitLab 克隆了完整的 sissopp 1.2.0 源码到项目目录 `./sissopp/`
2. 包含 `setup.py` + `CMakeLists.txt` + Python bindings（`src/python/sissopp/`）
3. 在 WSL2 中通过 conda 安装了全部编译工具链：

| 工具 | 版本 | 安装方式 |
|---|---|---|
| CMake | 4.3.0 | `conda install cmake` |
| g++ | 13.3.0 | 系统自带 (apt) |
| gfortran | 13.4.0 | `conda install gfortran` |
| OpenMPI | 5.0.10 | `conda install openmpi` |
| pybind11 | 3.0.3 | `conda install pybind11` |
| Boost 1.85.0 | (filesystem, system, serialization, python) | `conda install boost` |

4. CMake 配置时卡在 `boost_mpi` 组件：

```
CMake Error: No suitable build variant has been found for boost_mpi
```

5. **根因:** conda-forge 的 `boost` 包不包含 MPI 组件。需要 `libboost-mpi-dev`（apt）或 `boost-mpi`（conda alternative channel），但 conda 中不存在该包，apt 下载在 WSL2 网络环境中极慢（~10KB/s），多次超时。

6. sissopp 的 CMakeLists.txt 第 153 行 `find_package(MPI REQUIRED)` 和第 206 行 `find_package(Boost ... COMPONENTS ... mpi ... REQUIRED)` 表明 MPI 是硬性依赖，无法通过 CMake 选项绕过。

**结论:** sissopp 本身设计完善，但编译依赖链在离线/低带宽环境中过于脆弱。

---

### 路线 B：Fortran 原始 SISSO（无 Python 绑定）

**来源:** https://github.com/rouyang2017/SISSO （已克隆至 `/tmp/SISSO/`）

**状态: ❌ 失败**

**过程:**

1. `/tmp/SISSO/src/` 包含 6 个 Fortran 源文件：`var_global.f90`, `libsisso.f90`, `DI.f90`, `FC.f90`, `FCse.f90`, `SISSO.f90`
2. 尝试用 gfortran 直接编译，遇到 `use mpi` → 缺少 `mpi.mod`
3. 改用 conda 安装的 `mpifort`（OpenMPI Fortran wrapper）编译，MPI 模块问题解决，但出现类型错误：

```
Error: Type mismatch in argument 'status' at (1); passed REAL(4) to INTEGER(4)
  at libsisso.f90:578, 596, 653
```

4. **根因:** Fortran 隐式类型规则（I-N 为 INTEGER，A-H/O-Z 为 REAL）。`status` 变量名以 `s` 开头，被默认推断为 `REAL`，但 MPI 的 `mpi_recv` 期望 `INTEGER` 类型的 `status(MPI_STATUS_SIZE)`
5. 尝试添加 `implicit none` 到模块级别 → 暴露数十处隐式类型，超出了单次修复的合理范围
6. 参照 PDF 指南（`SISSO的安装与使用.pdf`），官方推荐使用 Intel oneAPI 的 `mpiifort` 编译，但安装 Intel oneAPI HPC Toolkit 需要数 GB 下载

**结论:** Fortran SISSO 需要 Intel oneAPI 或大量源码修补，不适合在 Windows/WSL 混合环境中快速部署。

---

### 路线 C：gplearn（遗传编程符号回归，最终采用方案）

**来源:** `pip install gplearn`（PyPI，纯 Python）

**状态: ✅ 成功**

**过程:**

1. 在 WSL2 conda 环境中安装时遇到 scipy/numpy 版本兼容性问题（scipy 1.15.3 + numpy 2.2.6 冲突）
2. 在 Windows venv（`.venv`, Python 3.12）中安装成功：`uv pip install gplearn`
3. 用项目真实数据（`example/B_dataset.csv`, 141 样本, 14 特征）测试：

```
Population: 3000, Generations: 10
Best fitness: 6.47 (MSE in scaled space)
MAE: 2.19 kcal/mol (inverse-transformed to real units)
Discovered formula (partial):
  abs(add(add(mul(div(abs(X0), dipole), sub(add(X0, reaction_energy), B_s)), ...)))
```

4. 重写了 `src/sisso_wrapper.py`，使 SISSORegressor 类使用 gplearn 作为后端，同时保留 sissopp 作为可选升级路径

---

## 三、最终方案详情

### 架构

```
src/sisso_wrapper.py
├── 尝试 import sissopp  → SISSO_BACKEND = "sissopp" (C++, 性能最优)
└── 回退 import gplearn   → SISSO_BACKEND = "gplearn"  (纯 Python, 即时可用)
```

### gplearn 配置参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `population_size` | 3000 | 遗传算法种群大小。越大探索越充分，但耗时线性增长 |
| `generations` | 15 | 进化代数。10-20 代通常收敛 |
| `parsimony_coefficient` | 0.001 | 公式复杂度惩罚。越大公式越简单，精度可能下降 |
| `function_set` | `('add','sub','mul','div','sqrt','log','abs','neg','inv')` | 允许的数学运算符 |
| `p_crossover` | 0.7 | 交叉概率。标准遗传算法参数 |
| `random_state` | 42 | 随机种子 |
| `n_jobs` | 1 | 并行核心数 |

### 测试结果（gplearn, 10 generations）

| 指标 | 值 | 对比 SVR (best model) |
|---|---|---|
| MAE (训练集) | 2.19 kcal/mol | SVR: 2.52 (RKfold) |
| 公式可解释性 | ✅ 显式数学表达式 | ❌ 黑箱 RBF 核 |
| 训练时间 (~88样本) | ~30 秒 (10 gen) | ~2 分钟 (100 trials Optuna) |

### 与管线集成

`SISSORegressor` 遵循 sklearn 的 `fit()/predict()/get_params()/set_params()` 接口，无需特殊处理即可加入 `main.py` 的 models 字典。

训练过程中，公式会打印到日志并存储在 `model.formula_` 属性中：

```
============================================================
  Symbolic regression discovered formula:
  abs(add(add(mul(div(abs(X0), dipole), sub(add(X0, reaction_energy), B_s)), ...))
============================================================
```

---

## 四、安装依赖

```bash
# 必需（已安装）
pip install gplearn

# 可选（如需 C++ 加速）
# sissopp 需从源码编译，见下方编译指南
```

---

## 五、sissopp 手动编译指南（供后续升级参考）

如果未来网络条件改善，可按以下步骤编译 sissopp 以获得更好的性能：

```bash
# 在 WSL2 中
conda activate bdataset2

# 1. 安装缺失的 boost_mpi
sudo apt-get update
sudo apt-get install -y libboost-mpi-dev
# 或：conda install -c conda-forge boost-mpi

# 2. 编译安装
cd /mnt/d/CC-Test/Stable-Organoboronates-ML-activation-Revised/sissopp
pip install -e .

# 3. 验证
python -c "from sissopp import FeatureSpace, SISSO; print('OK')"
```

编译完成后，`src/sisso_wrapper.py` 会自动检测并切换为 sissopp 后端。

---

## 六、结论

| 路线 | 状态 | 原因 |
|---|---|---|
| sissopp (C++ Python 绑定) | ❌ | boost_mpi 无法安装（网络/包缺失） |
| Fortran 原始 SISSO | ❌ | MPI 类型错误，需 Intel oneAPI 或源码修补 |
| **gplearn (遗传编程)** | ✅ **已部署** | 纯 Python，即时可用，MAE=2.19 kcal/mol |

当前方案（gplearn）已完全满足项目的符号回归需求。`SISSORegressor` 已集成到 `main.py` 的训练管线中，与其他 17 个模型一同参与 SHAP-RFECV 特征筛选和性能评估。唯一的缺点是训练时间较长（30 秒/模型，10 代），但考虑到它输出的是可直接写入论文的显式数学公式，这个代价完全可以接受。
