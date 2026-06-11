你是一名精通机器学习流程调试与评估的计算科学家。请使用 **superpowers** 技能规划并管理整个诊断与修复任务。过程中可自主调用其他 skills 辅助完成任务。

结合以下评估，判断是否正确，修复其余代码，给出的代码可供参考：

---
. 我指出的 GPR 崩溃问题是否修改完全？效果如何？
GPR 内核泄漏的根因已正确修复。
在 train_and_evaluate.py 中增加了 GPR 特殊处理块，从 best_params 中剥离出字符串 kernel、length_scale、noise_level，并根据这些值重建真实的 kernel 对象，再将其注入回参数集。这避免了“仅 pop 导致 kernel 丢失”的陷阱，也保证了调优得到的内核配置被完整保留。同时在 fixed_params.py 中为 GPR 增加了 n_restarts_optimizer=5，有助于减轻核函数边缘似然的局部最优问题。

但本次修改引入了一个新的不兼容错误：
iterative_optimization.py 中调用 plot_scatter 时新增了 rkf_mae=... 和 rkf_r2=... 两个关键字参数，但 visualization.py 中的 plot_scatter 函数签名并未包含这两个参数（根据最后一次提供的版本，仅有 r2_loo=None）。运行时会抛出 TypeError: plot_scatter() got unexpected keyword argument 'rkf_mae'，导致所有模型完成迭代后无法绘制最终散点图。这是未同步修改导致的新缺陷。

2. 是否还有其他值得修改的问题？修改是否引入其他新问题？
plot_scatter 参数不兼容（见上）是必须修复的新问题。

除此以外，整体修复质量较高，未见其他逻辑错误或数据泄露。

轻微的代码清洁问题：iterative_optimization.py 仍导入了 y_randomization_test 但已注释掉其调用，这无功能影响，但可保留作为快速恢复的标记。

Ridge/Lasso 分支现在使用外部传入的 random_state（而非硬编码 42）是一个积极的改进，保证了数据划分与优化的一致性。


---

### 二、 终极综合修改建议（三把精确手术刀）


#### 刀法 2：贯通 `src/train_and_evaluate.py` 尾部残留的数据切分种子

解决我此前指出的内部种子割裂问题。请打开 `src/train_and_evaluate.py`，定位到 Optuna 寻优结束后（约第 304～318 行处）用于输出传统对照指标的代码块，将写死的 `42` 替换为动态透传的变量：

```python
    # 修正前：残留硬编码 42
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 修正后：使用顶层透传的 random_state 参数保证全局一致性
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # ... 中间标准化与拟合代码保持不变 ...

    # 修正打印日志，消除视觉误导
    logger.info(f"random_state = {random_state}")

```

#### 刀法 3：GPR 寻优空间的自适应扩容（针对 `ConvergenceWarning`）

若运行日志中 GPR 依然提示收敛触碰边界，可直接在 `src/train_and_evaluate.py` 内部的 `objective` 函数中，对 `length_scale` 的采样下探空间进行拓宽：

```
python
        elif model_class == GaussianProcessRegressor:
            from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
            kernel_choice = trial.suggest_categorical("kernel", ["RBF", "Matern", "RBF+White"])
            
            # 将 length_scale 范围从原先的 0.1~10 放宽至 1e-3~1e3，赋予核函数更大弹性
            l_scale = trial.suggest_float("length_scale", 1e-3, 1e3, log=True)
            
            if kernel_choice == "RBF":
                kernel = ConstantKernel(1.0) * RBF(length_scale=l_scale)
            elif kernel_choice == "Matern":
                kernel = ConstantKernel(1.0) * Matern(length_scale=l_scale)
            else:
                n_level = trial.suggest_float("noise_level", 1e-5, 1.0, log=True)
                kernel = ConstantKernel(1.0) * RBF(length_scale=l_scale) + WhiteKernel(noise_level=n_level)
            
            params = {**base_params, "kernel": kernel, "alpha": trial.suggest_float("alpha", 1e-10, 1e-1, log=True)}


---
```

### 三、 架构层面的最终建议回应

关于**y-Randomization 的算力策略**，极其赞同外部评估的“终态独立验证”方案。
由于树模型庞大的时间开销（XGBoost 长达 8 小时），建议您在日常特征工程迭代中完全无视被注释的置换模块。当您最终选定模型并导出最优参数字典后，可编写一个几十行的极简独立脚本，直接加载最终选定的特征子集与固定参数，单纯调用 `y_randomization_test` 跑完一整夜的盲态置换流。这既确保了发文所需严谨的 $p$ 值背书，又最大化释放了开发联调期的宝贵算力。

也即，帮我写一个独立的代码，类似manual_selection_and_plot.py，读取指定的模型和特征数，进行`y_randomization_test`，并给出简单的使用指南


#### 任务三
完成以上所有分析和代码修改后，创建一个新的 Markdown 文件 **`debug_log_and_revisions_12.md`**，内容需包括：
1. **问题修正**：针对上述给出的问题，逐个思考修正，列出所有详细修正思路、修正代码、修正后的效果.
2. **修正后的代码使用指导**：如有命令行参数变化、配置文件字段变化等，说明如何运行新版本。
3. **遗留问题与建议**：如果在修复过程中仍存在无法完全解决的疑虑，或对未来运行的建议，一并列出。
4. 根据新代码，更新README、Pipeline.md、CLAUDE.md、user_manual.md等相关说明文档

#### 任务四
- 汇总全部修改的文件列表。
- 确认所有修改已保存，导入和语法无误（至少静态检查关键脚本）。
- 将总结告知用户，指明输出的核心文档为 `debug_log_and_revisions_12.md`。

---
**注意**：
- 所有修改均需基于对实际输出文件的完整阅读，不可臆测。
- 若某疑惑需要进一步运行小规模实验验证，可创建临时脚本并说明。
- 确保所有代码修改有详细**英文注释**，解释为何如此修改。