# -*- coding: utf-8 -*-
"""
LassoCV 诊断脚本：排查为什么 Lasso 在 SHAP-RFECV 迭代中产生恒定不变的指标。

运行方式:
    uv run python example/diagnose_lasso.py
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import LeaveOneOut, RepeatedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score

print("=" * 70)
print("  LassoCV 诊断：选择行为与收敛分析")
print("=" * 70)

# ── 1. 加载数据 ──
data = pd.read_csv('example/B_dataset.csv').dropna(axis=1, how='all')
features = data.select_dtypes(include=[np.number]).columns
X = data[features].drop('activation_energy', axis=1)
y = data['activation_energy']
print(f"\n[数据] N={len(X)}, 特征数={X.shape[1]}")
print(f"[数据] y 范围: {y.min():.2f} ~ {y.max():.2f} kcal/mol")

# ── 2. 模拟 train_and_evaluate 中的 LassoCV 调用方式 ──
# (与 src/train_and_evaluate.py 中 LassoCV 配合 LOO 的配置一致)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler(feature_range=(0, 100))
X_tr_s = scaler_X.fit_transform(X_train)
y_tr_s = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

# LassoCV 参数：与代码中完全一致
print("\n" + "-" * 50)
print("  LassoCV 配置")
print("-" * 50)
alphas = np.logspace(-4, 1, 50)
print(f"  alpha 搜索范围: {alphas[0]:.6f} ~ {alphas[-1]:.4f} (50 个候选)")
print(f"  CV: LeaveOneOut ({len(X_train)} folds)")
print(f"  max_iter: 10000")

lasso_cv = LassoCV(
    alphas=alphas,
    cv=LeaveOneOut(),
    max_iter=10000,
    tol=1e-4,
    selection='random',
    random_state=42,
    n_jobs=1
)
lasso_cv.fit(X_tr_s, y_tr_s)

print(f"\n  ★ LassoCV 选择的 alpha: {lasso_cv.alpha_:.6f}")
print(f"  ★ alpha_ 在候选列表中的索引: {np.argmin(np.abs(alphas - lasso_cv.alpha_))} / {len(alphas)}")
print(f"  ★ 非零系数数: {np.sum(lasso_cv.coef_ != 0)} / {len(lasso_cv.coef_)}")

# ── 3. 检查 alpha 路径上的 MSE ──
print("\n" + "-" * 50)
print("  Alpha 路径分析 (前 10 和后 5 个 alpha)")
print("-" * 50)
mse_path = lasso_cv.mse_path_  # shape: (n_alphas, n_folds)
mean_mse = mse_path.mean(axis=1)

# 按 alpha 从大到小
print(f"  {'alpha':<14} {'mean MSE':<12} {'n_nonzero':<10} {'备注'}")
print(f"  {'-'*45}")
# 用不同 alpha 值测试非零系数数
for i, a in enumerate(alphas):
    l = Lasso(alpha=a, max_iter=10000, tol=1e-4, random_state=42)
    l.fit(X_tr_s, y_tr_s)
    nz = np.sum(l.coef_ != 0)
    marker = " ← SELECTED" if abs(a - lasso_cv.alpha_) < 1e-10 else ""
    if i < 5 or i >= len(alphas) - 5 or abs(a - lasso_cv.alpha_) < 1e-10:
        print(f"  {a:<14.6f} {mean_mse[i]:<12.6f} {nz:<10}{marker}")

# ── 4. 模拟迭代：逐步移除特征，观察 LassoCV 行为 ──
print("\n" + "=" * 70)
print("  模拟 SHAP-RFECV 迭代（逐步移除最低 SHAP 重要性特征）")
print("=" * 70)

current_features = list(X.columns)
results = []

for it in range(1, 10):
    X_cur = X_train[current_features]
    X_tr_s = scaler_X.fit_transform(X_cur)
    y_tr_s = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

    lasso_cv = LassoCV(
        alphas=alphas, cv=LeaveOneOut(), max_iter=10000,
        tol=1e-4, selection='random', random_state=42, n_jobs=1
    )
    lasso_cv.fit(X_tr_s, y_tr_s)

    best_alpha = lasso_cv.alpha_
    nz = np.sum(lasso_cv.coef_ != 0)
    # 手动计算 5x5 RepeatedKFold MAE
    model = Lasso(alpha=best_alpha, max_iter=10000, tol=1e-4, random_state=42)
    rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
    mae_list, r2_list = [], []
    for tr_i, te_i in rkf.split(X_tr_s):
        fsy = MinMaxScaler(feature_range=(0, 100))
        yt_s = fsy.fit_transform(y_tr_s[tr_i].reshape(-1, 1)).ravel()
        ye_s = fsy.transform(y_tr_s[te_i].reshape(-1, 1)).ravel()
        model.fit(X_tr_s[tr_i], yt_s)
        yp_s = model.predict(X_tr_s[te_i])
        yp_o = fsy.inverse_transform(yp_s.reshape(-1, 1)).ravel()
        mae_list.append(mean_absolute_error(y_tr_s[te_i], yp_o))  # 注意：这里 y_tr_s 是 scaled
        r2_list.append(r2_score(y_tr_s[te_i], yp_o))

    # 修正：用逆变换后的值
    mae_list_orig = []
    r2_list_orig = []
    sy_full = MinMaxScaler(feature_range=(0, 100))
    y_full_s = sy_full.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    for tr_i, te_i in rkf.split(X_tr_s):
        fsy = MinMaxScaler(feature_range=(0, 100))
        yt_s = fsy.fit_transform(y_full_s[tr_i].reshape(-1, 1)).ravel()
        ye_s = fsy.transform(y_full_s[te_i].reshape(-1, 1)).ravel()
        model.fit(X_tr_s[tr_i], yt_s)
        yp_s = model.predict(X_tr_s[te_i])
        yp_o = fsy.inverse_transform(yp_s.reshape(-1, 1)).ravel()
        ye_o = sy_full.inverse_transform(ye_s.reshape(-1, 1)).ravel()
        mae_list_orig.append(mean_absolute_error(ye_o, yp_o))
        r2_list_orig.append(r2_score(ye_o, yp_o))

    results.append({
        'iteration': it,
        'n_features': len(current_features),
        'best_alpha': best_alpha,
        'n_nonzero': nz,
        'rkf_mae': np.mean(mae_list_orig),
        'rkf_r2': np.mean(r2_list_orig),
    })

    print(f"  Iter {it}: {len(current_features):2d} features | alpha={best_alpha:.6f} | "
          f"nonzero={nz:2d} | RKfold MAE={np.mean(mae_list_orig):.4f} | RKfold R²={np.mean(r2_list_orig):.4f}")

    # 移除"最不重要"的特征（用 Lasso 系数的绝对值模拟重要性）
    importances = np.abs(lasso_cv.coef_)
    if importances.sum() > 0:
        importance_pct = importances / importances.sum() * 100
        worst_idx = np.argmin(importance_pct)
        current_features.pop(worst_idx)
    else:
        break

# ── 5. 诊断结论 ──
print("\n" + "=" * 70)
print("  诊断结论")
print("=" * 70)

df = pd.DataFrame(results)
if len(df) > 0:
    unique_mae = df['rkf_mae'].nunique()
    unique_r2 = df['rkf_r2'].nunique()
    unique_alpha = df['best_alpha'].nunique()
    print(f"  指标变化情况:")
    print(f"    RKfold MAE: {unique_mae} 个唯一值 / {len(df)} 次迭代")
    print(f"    RKfold R²:  {unique_r2} 个唯一值 / {len(df)} 次迭代")
    print(f"    best_alpha:  {unique_alpha} 个唯一值 / {len(df)} 次迭代")

    if unique_alpha == 1:
        print(f"\n  ⚠ LassoCV 在所有特征子集上都选择了相同的 alpha = {df['best_alpha'].iloc[0]:.6f}")
        print(f"  可能原因:")
        print(f"    1. Alpha 候选范围下限 ({alphas[0]:.6f}) 不够小，导致最优 alpha 始终是最小值")
        print(f"    2. LOO CV 路径在所有特征子集上对 alpha 不敏感")
        print(f"    3. 数据噪声大，CV 无法区分不同 alpha")
        print(f"\n  建议:")
        print(f"    1. 将 alpha 候选范围下探到 1e-6: alphas = np.logspace(-6, 1, 50)")
        print(f"    2. 检查 mse_path_ 来确认不同 alpha 的 CV MSE 是否有差异")
        print(f"    3. 使用 RepeatedKFold 替代 LOO 来降低 CV 方差")

    # 打印 mse_path_ 的最小值位置
    print(f"\n  Alpha 候选范围的最小值: {alphas[0]:.6f}")
    best_alpha_is_min = lasso_cv.alpha_ <= alphas[0] * 1.01
    print(f"  最优 alpha 是否接近最小值: {'是' if best_alpha_is_min else '否'}")
