# 代码审查修复报告（最终复审）

日期：2026-06-19
分支：`codex/code-review-remediation`
对比基线：`master`（merge-base `30f919b`）
前序 final-wave 提交：`a5ca47d fix: close final review edge cases`
最终剩余问题修复提交：`c069d43 fix: enforce exact run family retention`

## 1. 最终结论

最终复审列出的剩余问题均已修复，并保留了分支上的既有正确改动：

- `clean_old_versions()` 现在只识别当前模型目录 basename 对应的 canonical 产物。
- `keep_versions` 在成功完成新 run 后再次执行，最终只保留最新的完整 family。
- checkpoint 特征数 metadata 只接受非 bool 的 Python/NumPy integral。
- `list_available_models()` 优先读取 current nested metric schema，并兼容 current direct/legacy flat schema。
- 重复且未引用的 `docs/superpowers/plans/2026-06-19-final-review-edge-cases.md` 已删除。
- 本报告已按最终行为、测试数、文件清单和真实 lock 验证结果更新，文件编码为 UTF-8。

## 2. 最终行为

### 2.1 精确且 model-scoped 的版本清理

`clean_old_versions(model_dir, keep_versions)` 使用：

```python
model_name = os.path.basename(os.path.normpath(model_dir))
escaped_model_name = re.escape(model_name)
```

并通过 anchored regex 仅识别以下白名单：

- `<Model>_final_<timestamp>.joblib`
- `<Model>_final_<timestamp>_metrics.txt`
- `<Model>_iteration_<N>_<timestamp>.joblib`
- `<Model>_iteration_<N>_<timestamp>_metrics.txt`
- `performance_history_<timestamp>.csv`
- `performance_history_<timestamp>.png`
- `final_scatter_<timestamp>.png`
- `final_scatter_<timestamp>_outliers.csv`

因此：

- `manual_notes`、备份文件、未知扩展和未知命名保持不变。
- foreign-model checkpoint/final 保持不变，且 foreign final 不参与当前模型 retention 排序。
- 当前模型旧 family 的 canonical joblib、metrics、history、scatter 和 outlier CSV 会一起删除。
- 如果目录尚无 final，则只用当前模型 canonical iteration timestamp 做兼容 retention。

### 2.2 `keep_versions` 的实际运行顺序

训练开始前仍保留一次清理，用于处理既有旧 family/orphan。

新 run 完成以下写入后，再执行一次：

```python
clean_old_versions(model_dir, keep_versions)
```

第二次清理位于 final joblib、final metrics 和 `plot_scatter()` 成功返回之后。结果是：

- 预置两个 family、完成一个新 run 且 `keep_versions=2` 时，只剩最新两个完整 family。
- 如果 scatter 或此前的最终写入失败，第二次清理不会执行，旧完整 family 不会被未完成 run 淘汰。

### 2.3 严格的 checkpoint 特征数 metadata

`_loaded_feature_count()` 以 `len(features)` 为真实值。

`optimal_n_features` 存在时：

- 接受 Python `int` 和 NumPy integer。
- 拒绝 bool。
- 拒绝所有 float，包括整数值 float。
- 拒绝字符串及其他可被 `int()` 强制转换的类型。
- metadata 与 `len(features)` 不一致时同样拒绝。

所有无效类型或不一致均抛出统一 `ValueError`，消息包含：

- checkpoint filepath；
- metadata 的 `repr`；
- 实际 `len(features)`。

### 2.4 模型列表 metric schema 优先级

`list_available_models()` 对 MAE/R² 分别按以下顺序读取首个非 `None` 值：

1. `metrics.secondary.internal_cv.rkf_*_mean`
2. `metrics.internal_cv.rkf_*_mean`
3. flat `metrics.rkf_*_mean`
4. legacy flat `metrics.rkf_*_opt_mean`

current nested schema 因此不会再被 legacy alias 覆盖。

### 2.5 文档清理

本分支新引入但重复、未引用的：

```text
docs/superpowers/plans/2026-06-19-final-review-edge-cases.md
```

已删除。最终整分支相对 `master` 的净修改文件数由 36 个变为 35 个。

## 3. TDD RED / GREEN

| 项目 | RED（修复前实际结果） | GREEN（修复后实际结果） |
|---|---|---|
| model-scoped 白名单 | 1 failed；foreign 2099 final 进入 retention，当前最新 family 被删 | `clean_old_versions` 相关 2 passed |
| 成功 run 后 retention | 1 failed；预置 2 个 family 后运行新 run，最终仍有 3 个 family | 清理/集成组合 3 passed |
| 异常路径 | 增加保护测试，验证 scatter 失败时旧 family 不被清理 | 通过 |
| strict Integral metadata | 1 failed、1 passed；`2.9`、`'2'`、`True` 可被旧 `int()` 接受 | metadata 相关 3 passed |
| 模型列表 schema | 2 failed；nested current 被 legacy 覆盖，direct/flat current 得到 NaN | 2 passed |

新增/强化测试覆盖：

- escaped model basename（模型名含 `+`）。
- manual notes、unknown files、foreign checkpoint 保留。
- canonical final/iteration joblib、metrics、history、scatter、outlier CSV。
- lightweight 新 run 的两个完整 family retention。
- scatter 异常时不做完成后清理。
- `2.9`、`'2'`、bool、Python int、NumPy integer。
- nested current、direct current、flat current、legacy flat metrics。

## 4. 最终验证

### 4.1 目标测试

```powershell
python -m pytest tests/test_iterative_boundary.py tests/test_external_validation.py tests/test_applicability_domain.py tests/test_examples_and_cli.py tests/test_y_randomization.py -q
```

结果：`68 passed in 5.37s`

其中：

- `tests/test_iterative_boundary.py`：`14 passed`
- `tests/test_external_validation.py`：`20 passed`

### 4.2 全套 pytest

```powershell
python -m pytest -q
```

结果：`83 passed in 7.84s`

### 4.3 unittest

```powershell
python -m unittest discover -s tests -q
```

结果：`Ran 83 tests in 5.634s`，`OK`。

输出中的奇异矩阵 pseudo-inverse fallback 是预期应用日志，不是测试 warning/failure。

### 4.4 编译、导入与 CLI

以下命令均 exit code 0：

```powershell
python -m compileall -q main.py src example tests
python -c "import src.external_validation; import src.applicability_domain; import src.y_randomization; print('available core imports ok')"
python -c "from tests.test_iterative_boundary import install_optional_dependency_stubs; install_optional_dependency_stubs(); import src.iterative_optimization; print('training import with optional-dependency stubs ok')"
python main.py --help
python src/external_validation.py --help
```

### 4.5 diff 与静态检查

- `git diff --check`：exit code 0，无 whitespace error。
- Windows Git 仅提示工作区文件未来可能进行 LF→CRLF 转换。
- ensemble 实现不再命中：
  - `all_predictions[label]`
  - `model_weights[label]`
  - ensemble 对 `metrics.mae_mean` 的 fallback
- 活跃文档/示例不再命中 `models/<ModelName>/y_randomization...`。
- `git ls-files` 大小写精确检查只跟踪 canonical `pipeline.md`。

### 4.6 `uv.lock`

主代理已使用真实 `uv 0.11.22` 执行 lock check，结果通过：

```text
Resolved 77 packages
```

因此：

- `uv.lock` 已经过 resolver 验证。
- greenlet 的 s390x wheel 差异保留为 resolver 结果。
- 未手工恢复或伪造这些 wheel 条目。
- 不再将 lock 状态描述为“环境无 uv、未检查”。

## 5. 最终修改文件清单

整条 remediation 分支相对 `master` 的净修改文件共 35 个：

1. `AGENTS.md`
2. `CLAUDE.md`
3. `README.md`
4. `README_CN.md`
5. `code-review-report-new.md`
6. `docs/superpowers/specs/2026-06-18-code-review-remediation-design.md`
7. `example/diagnose_lasso.py`
8. `example/improvement_code_examples.py`
9. `example/load_checkpoint_guide.py`
10. `example/standalone_y_randomization.py`
11. `main.py`
12. `pipeline.md`
13. `pyproject.toml`
14. `requirements.txt`
15. `src/applicability_domain.py`
16. `src/evaluation.py`
17. `src/external_validation.py`
18. `src/fixed_params.py`
19. `src/hyperparameter_optimization_and_training.py`
20. `src/iterative_optimization.py`
21. `src/model_utils.py`
22. `src/train_and_evaluate.py`
23. `src/visualization.py`
24. `src/y_randomization.py`
25. `tests/__init__.py`
26. `tests/test_applicability_domain.py`
27. `tests/test_evaluation.py`
28. `tests/test_examples_and_cli.py`
29. `tests/test_external_validation.py`
30. `tests/test_iterative_boundary.py`
31. `tests/test_model_configuration.py`
32. `tests/test_training_selection.py`
33. `tests/test_y_randomization.py`
34. `user_manual.md`
35. `uv.lock`

本次最终剩余问题的代码提交 `c069d43` 修改 4 个文件：

- `src/iterative_optimization.py`
- `src/external_validation.py`
- `tests/test_iterative_boundary.py`
- `tests/test_external_validation.py`

后续 documentation 提交包含：

- 删除 `docs/superpowers/plans/2026-06-19-final-review-edge-cases.md`
- 更新 `code-review-report-new.md`

## 6. 遗留关注项

- 未执行需要数天的完整 Optuna/SHAP/所有默认模型训练。
- CatBoost、LightGBM、XGBoost、GPlearn 等可选包的全部真实训练组合仍需后续长跑验证。
- timestamp 仍为秒级；同一模型被两个进程在同一秒并发启动时仍可能命名冲突。
- 非 canonical 历史 checkpoint 会被发现逻辑忽略；需要使用时应先迁移并校验 metadata。
- `keep_versions` 的 family 完整性由成功完成后的第二次清理保证；进程被强制终止时可能留下未完成产物，但不会在该失败调用中触发完成后清理。
