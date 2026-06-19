# 代码审查修复报告（最终整分支波次）

日期：2026-06-19
分支：`codex/code-review-remediation`
对比基线：`master`（merge-base `30f919b`）
本轮实现提交：`a5ca47d fix: close final review edge cases`

## 1. 审查范围与排除项

### 审查范围

- 整条 remediation 分支相对 `master` 的代码、测试、示例、依赖元数据和活跃文档。
- 本轮 final-wave 的全部 Important：
  - 一次模型执行的 run-family 时间戳与版本清理。
  - external validation checkpoint 的真实特征数、canonical filename 解析与唯一发现。
  - ensemble 的成员身份、权重 MAE、重复 DataFrame index、空成员和空交集。
- 本轮全部 Minor：
  - AD summary 和 prediction-only 解释。
  - atomic CLI 精确退出码。
  - canonical `pipeline.md`。
  - standalone y-randomization 实际输出路径。
  - loky warning 的窄范围消除。
  - Task 7 formatter/canonical 修复保持。
  - `uv.lock` greenlet s390x 三条 wheel 的可验证性检查。

### 排除项

- 未运行需要数天的完整 Optuna/SHAP/多模型训练。
- 未下载或安装缺失工具/可选依赖。
- 未修改 `sissopp` 第三方内容。
- 未对真实外部实验数据做科学结论复核。
- 未做与审查项无关的大规模重构或格式化。
- 本环境没有 `uv` 可执行文件，因此未做无法通过 `uv lock --check` 证实的纯文本 lock 恢复。

## 2. 原问题清单与严重度

现有设计文档只保留了 `CR-1..3 / H-1..4 / M-1..5 / L-1` 的范围编号，没有保留逐条原始标题。下表依据实际提交历史和设计章节恢复对应关系；本轮新增 Important/Minor 在下一节单列。

| 编号 | 严重度 | 问题摘要 | 状态 |
|---|---|---|---|
| CR-1 | Critical | final test 参与模型/特征选择或被重复评估，导致泛化指标污染 | 已修复 |
| CR-2 | Critical | CV/超参数选择中的缩放边界不一致，存在折间数据泄漏风险 | 已修复 |
| CR-3 | Critical | 最终持久化 estimator、scaler、特征和参数可能与实际评估对象不一致 | 已修复 |
| H-1 | High | 固定参数、调优参数和模型重建分散，可能丢失真实配置 | 已修复 |
| H-2 | High | `force_n_features` 对不支持模型可能在运行副作用发生后才失败 | 已修复 |
| H-3 | High | checkpoint 查找/精确特征数/ensemble 行对齐不确定 | 已修复并在本轮强化 |
| H-4 | High | AD 残差尺度和 y-randomization 统计流程不可靠 | 已修复 |
| M-1 | Medium | external validation 存在重复实现和旧路径 | 已修复 |
| M-2 | Medium | 依赖声明与默认模型注册表不一致 | 已修复 |
| M-3 | Medium | CLI 默认值、废弃参数和帮助文本不一致 | 已修复 |
| M-4 | Medium | 文档、示例、metric schema 和实际实现不一致 | 已修复 |
| M-5 | Medium | 更广泛 CI/跨环境基础设施 | 本次明确排除 |
| L-1 | Low | formatter、canonical 文件名和历史示例标识问题 | 已修复 |

## 3. 本轮 Important：原因、方案与实际修改

### A. run family 与版本清理

原因：

- `src/iterative_optimization.py` 在每个 iteration 和 final 阶段分别调用 `datetime.now()`。
- `clean_old_versions()` 又独立按 iteration timestamp 排序，无法知道哪些 iteration 属于被保留的 final run。

方案与修改：

- 每个模型进入执行时只生成一次 `run_timestamp`。
- iteration checkpoint、final checkpoint、performance history CSV/PNG、final scatter 及其衍生产物统一复用该 timestamp。
- `clean_old_versions()` 以 final timestamp 为 run-family 真值，保留最新 `keep_versions` 个 final family，并同步保留完整 iteration/metrics/history/scatter；孤立或过期 timestamp 产物删除。
- 没有 final 的中断目录仍可按 iteration timestamp 做兼容保留。

行为变化：

- 一次 3→2 feature path 的两个 iteration 和 final filename 具有同一 timestamp。
- `keep_versions=2` 保留两个完整 run family，不再受一个更新的 orphan iteration 干扰。

### B. checkpoint 真实特征数与 filename 分类

原因：

- 原实现优先信任 `optimal_n_features`，可能用错误 metadata 完成 exact match。
- final/iteration 使用两个重叠 glob，再通过 substring 分类；模型名含 `_final_` 时，一个 iteration 文件可能被发现两次并误判。

方案与修改：

- 实际特征数永远使用 `len(features)`。
- 若 `optimal_n_features` 存在且不一致，发现/加载时抛出包含文件路径、metadata 值和真实长度的 `ValueError`。
- 基于 model directory canonical name 构造 `re.escape()` 后的 anchored regex：
  - `<Model>_final_<timestamp>.joblib`
  - `<Model>_iteration_<N>_<timestamp>.joblib`
- 每个模型目录只枚举一次 `*.joblib`，每个文件最多发现一次。
- `list_available_models()` 复用同一发现逻辑。

行为变化：

- 假 metadata 不再参与 exact/closest 选择。
- 非 canonical checkpoint filename 被忽略。
- 模型名包含 `_final_` 或 `_iteration_` token 时仍按尾部 canonical suffix 正确分类。

### C. ensemble 成员身份、权重和样本对齐

原因：

- 展示 label 同时充当 prediction/weight 内部键，重复 spec 或 closest 收敛可能覆盖成员。
- 权重错误地优先读取 legacy flat 值，并可能回退到 stability `mae_mean`。
- 原始 DataFrame index 重复时，`.loc` 会扩张行数，造成预测长度与 index 长度不一致。
- 成员筛选后 0 行仍调用 `scaler.transform()`。

方案与修改：

- 为每个 spec row 创建唯一 `member_id`，展示 `label` 单独保存。
- 对 `_loaded_from` 做 `abspath + realpath + normcase`，重复 resolved checkpoint 作为 member error 跳过。
- 权重 MAE 顺序：
  1. final：`metrics.secondary.internal_cv.rkf_mae_mean`
  2. iteration：`metrics.internal_cv.rkf_mae_mean`
  3. 兼容：flat `rkf_mae_mean`
  4. 兼容：flat `rkf_mae_opt_mean`
- 明确不读取 stability `mae_mean`；值和最终权重都必须 finite 且大于 0。
- 输入表增加唯一内部 `_ensemble_row_id`，原始 index 保存到 `original_index`；成员交集只使用内部 row id。
- 输出恢复原始 index，同时保留 `original_index`、ID、target 和逐成员预测的正确一一对应。
- 新增 `EnsembleValidationError`：
  - 所有成员无效。
  - 成员无完整行。
  - 成员预测交集为空。
  - 权重数组无效。

行为变化：

- 重复 spec/closest 收敛不会覆盖已有成员，错误会进入 `ensemble_errors`。
- 无效 MAE 成员被跳过，ensemble 不再生成 NaN 权重。
- duplicate index 输入不会扩张或错配。
- 0 行成员在 scaler 前失败。

## 4. 本轮 Minor

### Applicability Domain summary

- 恢复 total external samples。
- 输出 flagged count 和 percentage。
- 增加简洁 interpretation。
- prediction-only interpretation 仅描述 leverage/k-NN criteria。
- `NearestNeighbors` 固定 `n_jobs=1`。
- AD 测试类使用窄范围 `LOKY_MAX_CPU_COUNT=1`，没有全局吞 warning。

### CLI / canonical / 文档

- atomic `--force_n_features` 测试从“非 0”强化为精确 `exit code == 2`。
- 设计文档只声明 canonical `pipeline.md`，不再声称维护两个大小写文件。
- standalone y-randomization 的说明和结束日志改为真实路径：
  - `models/y_randomization_<ModelName>.png`
- README、中文 README、user manual、pipeline、AGENTS、CLAUDE 同步真实路径。
- Task 7 formatter/current-metric/canonical 修复保持通过。

### uv.lock

- 分支较早的 resolver 更新添加 `gplearn==0.4.2`，同时删除了 greenlet 的三个 s390x wheel 条目。
- 本轮确认当前环境没有 `uv`，且用户要求不能下载工具。
- 因此没有做无法通过 `uv lock --check` 验证的文本恢复；`uv.lock` 在提交 `a5ca47d` 中未修改。
- 后续在装有与项目一致版本 `uv` 的环境中应执行 `uv lock --check`；若 resolver 仍删除这些 wheel，应保留 resolver 结果。

## 5. TDD RED / GREEN 记录

| 项目 | RED（修复前实际结果） | GREEN（修复后实际结果） |
|---|---|---|
| Run timestamp + family clean | 2 failed：3 个不同 timestamp；orphan iteration 导致完整 family 被删 | 2 passed |
| Checkpoint metadata + anchored discovery | 2 failed：不拒绝假 metadata；模型名含 `_final_` 时发现 3 次而非 2 次 | 4 passed（含既有 latest/ambiguity） |
| Ensemble edge cases | 7 failed：duplicate checkpoint、weight precedence、NaN MAE、duplicate index、0-row scaler、disjoint error | 7 passed；全 external 文件 16 passed |
| AD summary + loky | 2 failed 且 1 loky warning | AD 全文件 12 passed，0 warnings |
| Standalone y-randomization path | 1 failed：仍指向 `models/<ModelName>/...` | 1 passed |
| Atomic CLI code 2 | 测试强化项；生产行为原本已返回 2 | 精确断言通过 |
| 文档 canonical | 既有 canonical 测试保留 | 通过 |

## 6. 精确验证结果

### 目标测试

```powershell
python -m pytest tests/test_iterative_boundary.py tests/test_external_validation.py tests/test_applicability_domain.py tests/test_examples_and_cli.py tests/test_y_randomization.py -q
```

结果：`61 passed in 5.93s`

### 全套 pytest

```powershell
python -m pytest -q
```

结果：`76 passed in 9.20s`，无 warnings summary。

### unittest discover

```powershell
python -m unittest discover -s tests -v
```

结果：`Ran 76 tests in 6.693s`，`OK`。
日志中有一个预期的奇异矩阵 pseudo-inverse fallback 应用日志，不是 Python warning。

### 语法与导入

```powershell
python -m compileall -q main.py src example tests
```

结果：exit code 0。

```powershell
python -c "import src.external_validation; import src.applicability_domain; import src.y_randomization; print('available core imports ok')"
```

结果：通过。

```powershell
python -c "from tests.test_iterative_boundary import install_optional_dependency_stubs; install_optional_dependency_stubs(); import src.iterative_optimization; print('training import with optional-dependency stubs ok')"
```

结果：通过。

直接导入完整训练栈时，本机缺少 `optuna`，实际得到：

```text
ModuleNotFoundError: No module named 'optuna'
```

这是当前可选依赖环境限制；没有按要求下载依赖。

### CLI

```powershell
python main.py --help
```

结果：exit code 0，显示当前 `--n_jobs`、`--n_trials`、`--keep_versions`、`--min_features`、`--force_n_features`。

### 静态检查

- 生产代码不再命中：
  - `all_predictions[label]`
  - `model_weights[label]`
  - ensemble 对 `metrics.mae_mean` 的 fallback
  - `models/<ModelName>/y_randomization...`
- `git ls-files` 大小写精确检查：只跟踪 `pipeline.md`。
- `src/iterative_optimization.py` 仅有一次 run timestamp 生成，所有运行产物复用 `run_timestamp`。
- `git diff --check`：无 whitespace error；Windows Git 仅提示未来可能进行 LF→CRLF 转换。

### Synthetic smoke

1. 两成员 ensemble：
   - final/current MAE=1，iteration/current MAE=2。
   - duplicate index `[9, 9, 10]`。
   - weighted prediction 均为 `1.4`。
   - 原始 index 完整保留。
   - 结果：`synthetic ensemble smoke ok`。

2. 三个 run family、每个两个 iteration，`keep_versions=2`：
   - 最旧 family 全部删除。
   - 两个最新 family 的 final、两个 iteration、history 全部保留。
   - 结果：`synthetic run-family cleanup smoke ok`。

## 7. 使用和配置变化

- 无新增 CLI 参数。
- 本轮不改变训练算法或默认超参数。
- ensemble 输出新增 `original_index` 列，并恢复返回 DataFrame 的原始 index。
- ensemble 对无效 member 使用显式错误/跳过语义；调用方应检查 `ensemble_errors`。
- checkpoint 文件必须遵循 model directory canonical filename。
- `optimal_n_features` 与 `features` 不一致的历史 checkpoint 现在会被拒绝。
- AD summary 新增总数、比例和 interpretation。
- y-randomization 图仍由 `src/y_randomization.py` 写入根 `models/` 目录；文档现已与实现一致。

## 8. 关键文件

- `src/iterative_optimization.py`：run-family timestamp、family-aware cleanup。
- `src/external_validation.py`：checkpoint parser/validation、ensemble identity/weights/row alignment。
- `src/applicability_domain.py`：AD summary 和窄范围并行设置。
- `tests/test_iterative_boundary.py`：3→2 path 与完整 family retention。
- `tests/test_external_validation.py`：metadata、canonical filename、duplicate checkpoint、weight、duplicate index、空集。
- `tests/test_applicability_domain.py`：summary、prediction-only interpretation、loky warning。
- `tests/test_examples_and_cli.py`：精确 CLI code 2 和 canonical docs。
- `tests/test_y_randomization.py`：standalone 输出路径。

## 9. 遗留风险

- 未执行完整多日训练；真实 Optuna/SHAP/所有默认模型组合仍需后续长跑验证。
- 当前环境缺少 `optuna`，训练模块直接导入受限；正式运行前应按项目依赖安装。
- CatBoost、LightGBM、XGBoost、GPlearn 等可选包的真实二进制兼容性未在本轮环境逐一验证。
- `uv.lock` 的 greenlet s390x resolver 差异未在无 `uv` 环境伪造处理。
- timestamp 仍沿用秒级 filename 格式；同一模型被两个进程在同一秒并发启动仍可能产生命名冲突。
- 非 canonical 历史 checkpoint filename 现在会被忽略；需要使用时应先迁移为当前格式并校验 metadata。

## 10. 完整修改文件列表

### 整条 remediation 分支相对 `master`（最终 36 个）

1. `AGENTS.md`
2. `CLAUDE.md`
3. `README.md`
4. `README_CN.md`
5. `docs/superpowers/plans/2026-06-19-final-review-edge-cases.md`
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
36. `code-review-report-new.md`

### 本轮实现提交 `a5ca47d`（17 个）

1. `AGENTS.md`
2. `CLAUDE.md`
3. `README.md`
4. `README_CN.md`
5. `docs/superpowers/plans/2026-06-19-final-review-edge-cases.md`
6. `docs/superpowers/specs/2026-06-18-code-review-remediation-design.md`
7. `example/standalone_y_randomization.py`
8. `pipeline.md`
9. `src/applicability_domain.py`
10. `src/external_validation.py`
11. `src/iterative_optimization.py`
12. `tests/test_applicability_domain.py`
13. `tests/test_examples_and_cli.py`
14. `tests/test_external_validation.py`
15. `tests/test_iterative_boundary.py`
16. `tests/test_y_randomization.py`
17. `user_manual.md`

本报告为第 18 个本轮文件，将在独立 documentation commit 中提交。
