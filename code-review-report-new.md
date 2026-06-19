# 代码审查修复报告（最终复审）

日期：2026-06-20
分支：`codex/code-review-remediation`
对比基线：`master`（merge-base `30f919b`）
前序 final-wave 提交：`a5ca47d fix: close final review edge cases`
前序 family-retention 提交：`c069d43 fix: enforce exact run family retention`
本轮最终门禁代码提交：见本次提交（final 原子写与 checkpoint 反序列化隔离收口）

## 1. 最终结论

最终复审列出的剩余问题均已修复，并保留了分支上的既有正确改动：

- `clean_old_versions()` 现在只识别当前模型目录 basename 对应的 canonical 产物。
- `keep_versions` 在 API/helper/CLI 三层均严格要求正整数，bool 无效，非法 API 调用在 split/getcwd/logger/write 前原子失败，CLI 返回 exit code 2 且无运行时副作用。
- history、scatter 和 outlier 成功后，final metrics/joblib 会先写同目录临时文件，再用 `os.replace()` 原子落位；任一步失败都会清理临时/新 canonical final 文件，final joblib 仍是完整 family 的最后完成标记。
- 无 final joblib 的 iteration/history/scatter 都按 orphan 处理；下次开头 cleanup 会删除 orphan，同时保留旧完整 families。
- 损坏 checkpoint 的反序列化异常会记录 warning 并跳过单个文件；可读但 metadata/schema 非法的 checkpoint 仍明确抛出 `ValueError`。
- checkpoint `features` 必须是非字符串 sequence，元素为非空字符串且唯一；`optimal_n_features` 只接受非 bool 的 Python/NumPy integral。
- `list_available_models()` 优先读取 current nested metric schema，并兼容 current direct/legacy flat schema。
- 重复且未引用的 `docs/superpowers/plans/2026-06-19-final-review-edge-cases.md` 已删除。
- 本报告已按最终行为、实际测试数和文件清单更新，文件编码为 UTF-8。

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
- 只有存在 canonical final joblib 的 timestamp 才是完整 family。
- 如果目录尚无 final，所有 canonical iteration/history/scatter/final-metrics 都是 orphan，cleanup 会删除它们。

### 2.2 `keep_versions` 正整数与原子性

`iterative_optimization()` 的首个可执行语句校验 `keep_versions`：

- 接受 Python/NumPy integral 且值 `>= 1`；
- 拒绝 `0`、负数、bool、float、字符串和 `None`；
- 校验发生在 `train_test_split()`、`os.getcwd()`、logger 初始化和任何写入之前。

`clean_old_versions()` 自身重复执行同一防御性校验。`main.py` 的
`--keep_versions` 使用 positive-int argparse type，因此 CLI 的 `0`/`-1`
在 runtime setup 前以 exit code 2 失败，不创建 `models/` 或 optimization log。

### 2.3 完成标记与实际运行顺序

训练开始前仍保留一次清理，用于处理既有旧 family/orphan。

新 run 的写入顺序为：

1. performance history PNG；
2. performance history CSV；
3. final scatter，以及存在 outlier 时对应的 outlier CSV；
4. final metrics 临时文件；
5. final joblib 临时文件；
6. `os.replace()` final metrics；
7. `os.replace()` final joblib（最后的完整 family 标记）；
8. `clean_old_versions(model_dir, keep_versions)`。

结果是：

- 预置两个 family、完成一个新 run 且 `keep_versions=2` 时，只剩最新两个完整 family。
- 如果 scatter 失败，本次 timestamp 不会留下 final joblib/final metrics，且末尾 cleanup 不执行。
- 如果 final metrics/joblib 任一步失败，本次 timestamp 不会留下 canonical final 或临时文件，且末尾 cleanup 不执行。
- 失败 run 可能暂留 iteration/history orphan；下一次开头 cleanup 会删除这些 orphan，同时旧两个完整 family 保持不变。

### 2.4 严格的 checkpoint features schema 与特征数 metadata

`_loaded_feature_count()` 统一执行 schema 和 metadata 校验，并以
`len(features)` 为真实值。

`features` 必须：

- 是非字符串 `Sequence`（当前写入格式为 list）；
- 每一项都是非空、非纯空白字符串；
- 特征名唯一。

`optimal_n_features` 存在时：

- 接受 Python `int` 和 NumPy integer。
- 拒绝 bool。
- 拒绝所有 float，包括整数值 float。
- 拒绝字符串及其他可被 `int()` 强制转换的类型。
- metadata 与 `len(features)` 不一致时同样拒绝。

所有无效 `optimal_n_features` 类型或不一致均抛出统一 `ValueError`，消息包含：

- checkpoint filepath；
- metadata 的 `repr`；
- 实际 `len(features)`。

### 2.5 损坏 checkpoint 的隔离

`_discover_model_checkpoints()` 只在 `joblib.load()` 周围捕获
`Exception`（因此覆盖 EOF、pickle、KeyError、IndexError、ValueError 等单文件反序列化失败），
记录包含路径和异常类型的 warning 后跳过该文件。

- 健康 final 与截断 iteration 并存时，discovery/list/load 继续使用健康 final。
- 如果 canonical checkpoints 全部损坏，选择层按正常的 “No final or iteration checkpoints found” 路径失败。
- 成功反序列化后发生的 features schema 或 `optimal_n_features` 不一致不在捕获范围内，继续明确抛出包含路径的 `ValueError`。

### 2.6 模型列表 metric schema 优先级

`list_available_models()` 对 MAE/R² 分别按以下顺序读取首个非 `None` 值：

1. `metrics.secondary.internal_cv.rkf_*_mean`
2. `metrics.internal_cv.rkf_*_mean`
3. flat `metrics.rkf_*_mean`
4. legacy flat `metrics.rkf_*_opt_mean`

current nested schema 因此不会再被 legacy alias 覆盖。

### 2.7 文档清理

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
| `keep_versions` 正整数 | API/helper/CLI 新测试失败；非法值进入 split/runtime | API/helper/CLI 原子性测试通过 |
| scatter 后完成标记 | 失败 run 仍留下 final joblib/metrics | 失败 run 无 final，后续 cleanup 只删 orphan |
| final 原子写 | partial final dump 后残留 canonical final/临时文件 | partial final dump 后旧健康 family 保留、无 canonical final/临时文件 |
| 损坏 checkpoint | 截断 iteration 或 `KeyError` 损坏 iteration 阻断健康 final discovery/list/load | 损坏文件 warning+skip；全损坏走 No checkpoints |
| features schema | string、非 str 项、空项、重复项可通过 `len()` | 统一 `ValueError` 且包含 checkpoint path |

新增/强化测试覆盖：

- escaped model basename（模型名含 `+`）。
- manual notes、unknown files、foreign checkpoint 保留。
- canonical final/iteration joblib、metrics、history、scatter、outlier CSV。
- lightweight 新 run 的两个完整 family retention。
- scatter 异常时无 final joblib/metrics、不做完成后清理；显式后续 cleanup 保留旧两个完整 family 并删除失败 run orphan。
- partial final dump 时旧健康 family 保留、无 canonical final/临时文件。
- `2.9`、`'2'`、bool、Python int、NumPy integer。
- nested current、direct current、flat current、legacy flat metrics。
- API/helper 的 `keep_versions` 非法值与 CLI `0`/`-1`。
- 有效 final + 截断 iteration、有效 final + `KeyError` 损坏 iteration、全部 checkpoint 损坏。
- features 为字符串、包含非字符串、空/空白名称、重复名称。

## 4. 最终验证

### 4.1 目标测试

```powershell
python -m pytest tests/test_iterative_boundary.py tests/test_external_validation.py tests/test_applicability_domain.py tests/test_examples_and_cli.py tests/test_y_randomization.py -q
```

结果：`76 passed in 6.83s`

### 4.2 全套 pytest

```powershell
python -m pytest -q
```

结果：`91 passed in 10.66s`

### 4.3 unittest

```powershell
python -m unittest discover -s tests -q
```

结果：`Ran 91 tests in 8.080s`，`OK`。

输出中的奇异矩阵 pseudo-inverse fallback、损坏 checkpoint skip warning
均为测试覆盖的预期应用日志，不是测试失败。

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
- AST/源码顺序断言确认 `keep_versions` 是 `iterative_optimization()` 的首个可执行校验，且写入顺序为 scatter → final metrics temp → final joblib temp → replace metrics → replace joblib → cleanup。
- Windows Git 仅提示工作区文件未来可能进行 LF→CRLF 转换。
- ensemble 实现不再命中：
  - `all_predictions[label]`
  - `model_weights[label]`
  - ensemble 对 `metrics.mae_mean` 的 fallback
- 活跃文档/示例不再命中 `models/<ModelName>/y_randomization...`。
- `git ls-files` 大小写精确检查只跟踪 canonical `pipeline.md`。

### 4.6 `uv.lock`

前序复审曾使用真实 `uv 0.11.22` 执行 lock check并记录：

```text
Resolved 77 packages
```

本轮当前 Anaconda shell 中 `uv` 不在 PATH，`python -m uv` 也不可用，因此
未重复执行 lock check。本轮代码/报告提交均未修改 `uv.lock`。前序结论保持为：

- `uv.lock` 已经过 resolver 验证。
- greenlet 的 s390x wheel 差异保留为 resolver 结果。
- 未手工恢复或伪造这些 wheel 条目。
- 本轮不把前序验证冒充为当前 shell 的重复验证。

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

本轮当前提交将更新 5 个文件：

- `src/iterative_optimization.py`
- `src/external_validation.py`
- `tests/test_iterative_boundary.py`
- `tests/test_external_validation.py`
- `code-review-report-new.md`

## 6. 遗留关注项

- 未执行需要数天的完整 Optuna/SHAP/所有默认模型训练。
- CatBoost、LightGBM、XGBoost、GPlearn 等可选包的全部真实训练组合仍需后续长跑验证。
- timestamp 仍为秒级；同一模型被两个进程在同一秒并发启动时仍可能命名冲突。
- 非 canonical 历史 checkpoint 会被发现逻辑忽略；需要使用时应先迁移并校验 metadata。
- 进程在 final joblib 完成标记前被强制终止时可能留下 canonical orphan；下一次开头 cleanup 会删除它们。
