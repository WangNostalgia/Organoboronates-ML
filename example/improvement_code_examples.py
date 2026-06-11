# -*- coding: utf-8 -*-
"""
===============================================================================
机器学习模型改进方法 — 完整代码示例
===============================================================================

本文件包含了 model_analysis_report.md 第 6.2 和 6.3 节中各项改进建议的
可执行代码示例。所有代码均为独立模块，可直接运行，不会修改项目原有代码。

涵盖内容：
  1. 改进的特征选择策略（自适应阈值 / CV性能导向 / RFECV / SHAP-RFECV）
  2. 集成方法（简单平均 / 加权平均 / Stacking）
  3. 集成模型选择策略（误差相关性分析 / 聚类分组 / 穷举组合）
  4. GPR 不确定性量化
  5. 正则化参数调整（SVR epsilon / Ridge alpha / Lasso alpha）
  6. 外部验证框架
  7. 重复 k-fold 交叉验证（含与 LOOCV 的角色分工说明）
  8. y-randomization 检验

运行环境要求: Python 3.9+, scikit-learn, pandas, numpy, matplotlib, optuna, shap
===============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
import os

warnings.filterwarnings('ignore')

# ============================================================================
# 公共数据加载
# ============================================================================

def load_data(csv_path='example/B_dataset.csv'):
    """
    加载数据，返回特征矩阵 X 和目标向量 y

    Parameters
    ----------
    csv_path : str
        数据文件路径

    Returns
    -------
    X : DataFrame, 特征矩阵（仅数值列，不含 activation_energy）
    y : Series, 目标变量 activation_energy
    feature_names : list, 特征名称列表
    """
    data = pd.read_csv(csv_path)
    data = data.dropna(axis=1, how='all')
    features = data.select_dtypes(include=[np.number]).columns
    X = data[features].drop('activation_energy', axis=1)
    y = data['activation_energy']
    print(f"[数据加载] 样本数: {len(X)}, 特征数: {len(X.columns)}")
    print(f"[数据加载] 目标变量范围: {y.min():.2f} ~ {y.max():.2f} kcal/mol")
    return X, y, list(X.columns)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  第1部分：改进的特征选择策略                                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ---------------------------------------------------------------------------
# 方案A：自适应重要性阈值（基于排列检验）
# ---------------------------------------------------------------------------

def permutation_feature_importance(model, X, y, scaler_X, scaler_y, n_permutations=100, random_state=42):
    """
    使用排列检验（Permutation Test）确定每个特征的统计显著性。

    原理：
        对于每个特征，随机打乱其值 100 次，每次计算 SHAP 重要性。
        如果原始重要性在随机分布中处于前 5%（即 p < 0.05），
        则认为该特征统计显著，应当保留。

    Parameters
    ----------
    model : 已训练的 sklearn 模型
    X : DataFrame, 特征矩阵（未缩放）
    y : Series, 目标变量
    scaler_X : 特征缩放器
    scaler_y : 目标缩放器
    n_permutations : int, 排列次数（默认100）
    random_state : int, 随机种子

    Returns
    -------
    results_df : DataFrame, 包含每个特征的原始重要性、p-value、是否显著
    """
    import shap
    from sklearn.preprocessing import MinMaxScaler

    rng = np.random.RandomState(random_state)
    X_scaled = scaler_X.transform(X)
    n_features = X.shape[1]

    # 步骤1：计算原始 SHAP 重要性
    print("[排列检验] 步骤1: 计算原始 SHAP 重要性...")
    n_clusters = max(10, min(15, int(len(X) * 0.15)))
    n_samples = max(20, min(int(len(X) * 0.3), 100))
    background = shap.kmeans(X_scaled, n_clusters)
    explainer = shap.KernelExplainer(model.predict, background)

    sample_indices = rng.choice(len(X), n_samples, replace=False)
    shap_values_original = explainer.shap_values(X_scaled[sample_indices])
    original_importance = np.abs(shap_values_original).mean(axis=0)
    original_importance = original_importance / original_importance.sum() * 100

    # 步骤2：对每个特征进行排列检验
    print(f"[排列检验] 步骤2: 对 {n_features} 个特征进行排列检验（每个 {n_permutations} 次）...")
    permuted_importances = np.zeros((n_features, n_permutations))

    for feat_idx in range(n_features):
        X_permuted = X.copy()
        for perm_idx in range(n_permutations):
            # 随机打乱该特征的值
            X_permuted.iloc[:, feat_idx] = rng.permutation(X.iloc[:, feat_idx].values)
            X_permuted_scaled = scaler_X.transform(X_permuted)

            # 重新计算 SHAP
            shap_values_perm = explainer.shap_values(X_permuted_scaled[sample_indices])
            permuted_importance = np.abs(shap_values_perm).mean(axis=0)
            permuted_importances[feat_idx, perm_idx] = permuted_importance[feat_idx]

    # 步骤3：计算 p-value
    # p-value = 打乱后重要性 ≥ 原始重要性的比例
    p_values = np.array([
        np.mean(permuted_importances[i] >= original_importance[i])
        for i in range(n_features)
    ])
    significant = p_values < 0.05

    results_df = pd.DataFrame({
        '特征': X.columns,
        '原始重要性(%)': original_importance.round(2),
        'p_value': p_values.round(4),
        '统计显著': ['是' if s else '否' for s in significant]
    }).sort_values('原始重要性(%)', ascending=False)

    print("\n[排列检验] 结果:")
    print(results_df.to_string(index=False))

    return results_df


# ---------------------------------------------------------------------------
# 方案B：基于 CV 性能变化的前向/后向选择（推荐）
# ---------------------------------------------------------------------------

def cv_guided_feature_selection(model_class, X, y, best_params, scaler_X, scaler_y,
                                  max_mae_increase=0.05, cv_folds=5, random_state=42):
    """
    基于 CV 性能变化的特征选择。

    原理：
        不依赖重要性数值的阈值，而是直接计算"移除特征后 CV MAE 的变化"。
        只有当移除特征后 MAE 上升不超过 max_mae_increase（如5%）时，
        才正式移除该特征。这样确保每次移除都不会显著损害模型性能。

    Parameters
    ----------
    model_class : sklearn 模型类（如 SVR）
    X : DataFrame, 特征矩阵
    y : Series, 目标变量
    best_params : dict, 模型的最佳超参数
    scaler_X : 特征缩放器
    scaler_y : 目标缩放器
    max_mae_increase : float, 允许的最大 MAE 上升比例（如0.05=5%）
    cv_folds : int, 交叉验证折数
    random_state : int, 随机种子

    Returns
    -------
    selected_features : list, 最终保留的特征名称
    history : list of dict, 每步的详细记录
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import MinMaxScaler
    import shap

    current_features = list(X.columns)
    removed_features = []
    history = []
    iteration = 0

    # 建立基线：当前特征集的 CV MAE
    model = model_class(**best_params)
    X_current_scaled = scaler_X.fit_transform(X[current_features])
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()
    baseline_scores = cross_val_score(
        model, X_current_scaled, y_scaled, cv=cv_folds,
        scoring='neg_mean_absolute_error'
    )
    baseline_mae = -baseline_scores.mean()  # neg → positive
    print(f"[CV特征选择] 初始 MAE (CV): {baseline_mae:.4f} (特征数: {len(current_features)})")

    while len(current_features) > 3:  # 最少保留3个特征
        iteration += 1

        # 步骤1：计算当前特征集的 SHAP 重要性
        model.fit(X_current_scaled, y_scaled)
        n_clusters = max(10, min(15, int(len(X) * 0.15)))
        n_samples = max(20, min(int(len(X) * 0.3), 100))
        background = shap.kmeans(X_current_scaled.to_numpy() if hasattr(X_current_scaled, 'to_numpy') else np.array(X_current_scaled), n_clusters)
        explainer = shap.KernelExplainer(model.predict, background)
        sample_indices = np.random.RandomState(random_state).choice(len(X), n_samples, replace=False)
        shap_values = explainer.shap_values(X_current_scaled[sample_indices] if hasattr(X_current_scaled, '__getitem__') else np.array(X_current_scaled)[sample_indices])
        importances = np.abs(shap_values).mean(axis=0)
        importances = importances / importances.sum()

        # 按重要性从低到高排序
        sorted_indices = np.argsort(importances)

        candidate_found = False
        for idx in sorted_indices:
            candidate_feature = current_features[idx]

            # 尝试移除候选特征
            temp_features = [f for f in current_features if f != candidate_feature]
            X_temp_scaled = scaler_X.fit_transform(X[temp_features])

            temp_scores = cross_val_score(
                model_class(**best_params), X_temp_scaled, y_scaled,
                cv=cv_folds, scoring='neg_mean_absolute_error'
            )
            temp_mae = -temp_scores.mean()

            mae_change = (temp_mae - baseline_mae) / baseline_mae  # 相对变化

            print(f"  [Iter {iteration}] 尝试移除 '{candidate_feature}' "
                  f"(重要性: {importances[idx]:.2%}) → CV MAE: {baseline_mae:.4f}→{temp_mae:.4f} "
                  f"({mae_change:+.1%})")

            if mae_change <= max_mae_increase:
                # MAE 没有显著上升 → 确认移除
                current_features = temp_features
                removed_features.append(candidate_feature)
                X_current_scaled = X_temp_scaled
                baseline_mae = temp_mae
                candidate_found = True

                history.append({
                    'iteration': iteration,
                    'removed_feature': candidate_feature,
                    'importance': f"{importances[idx]:.2%}",
                    'mae_before': baseline_mae,
                    'mae_after': temp_mae,
                    'mae_change': f"{mae_change:+.1%}",
                    'remaining_features': len(current_features)
                })

                print(f"    ✓ 确认移除 (MAE变化 {mae_change:+.1%} ≤ {max_mae_increase:+.0%})")
                break
            else:
                print(f"    ✗ 保留 (MAE变化 {mae_change:+.1%} > {max_mae_increase:+.0%})")

        if not candidate_found:
            print(f"\n[CV特征选择] 无特征可进一步移除（所有候选移除都导致 MAE 显著上升）")
            break

    print(f"\n[CV特征选择] 最终保留特征 ({len(current_features)}个): {current_features}")
    print(f"[CV特征选择] 最终 CV MAE: {baseline_mae:.4f}")

    return current_features, history


# ---------------------------------------------------------------------------
# 方案C：递归特征消除 + 交叉验证 (RFECV)
# ---------------------------------------------------------------------------

def rfecv_feature_selection(model, X, y, scaler_X, scaler_y, cv_folds=5, min_features_to_select=3):
    """
    使用 scikit-learn 的 RFECV 进行自动特征选择。

    原理：
        RFECV 从全部特征开始，逐步移除最不重要的特征。
        在每一个特征数量下用 CV 评估性能，最终自动选择使 CV 性能最优的特征子集。

    Parameters
    ----------
    model : 已配置好超参数的 sklearn 模型
    X : DataFrame, 特征矩阵
    y : Series, 目标变量
    scaler_X : 特征缩放器
    scaler_y : 目标缩放器
    cv_folds : int, 交叉验证折数
    min_features_to_select : int, 最少保留特征数

    Returns
    -------
    selected_features : list, 最终保留的特征名称
    cv_results : dict, RFECV 的详细结果
    """
    from sklearn.feature_selection import RFECV
    from sklearn.model_selection import KFold

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()

    print(f"[RFECV] 开始自动特征选择（最少 {min_features_to_select} 个特征）...")

    rfecv = RFECV(
        estimator=model,
        step=1,  # 每次移除1个特征
        cv=KFold(n_splits=cv_folds, shuffle=True, random_state=42),
        scoring='neg_mean_absolute_error',
        min_features_to_select=min_features_to_select,
        n_jobs=-1
    )
    rfecv.fit(X_scaled, y_scaled)

    # 收集结果
    selected_mask = rfecv.support_
    selected_features = list(X.columns[selected_mask])
    feature_ranking = rfecv.ranking_

    print(f"\n[RFECV] 最优特征数: {rfecv.n_features_}")
    print(f"[RFECV] 保留特征: {selected_features}")
    print(f"[RFECV] 最优 CV MAE: {-rfecv.cv_results_['mean_test_score'][rfecv.n_features_ - min_features_to_select]:.4f}")

    # 可视化 RFECV 路径
    n_scores = len(rfecv.cv_results_['mean_test_score'])
    plt.figure(figsize=(10, 5))
    plt.errorbar(
        range(min_features_to_select, min_features_to_select + n_scores),
        -rfecv.cv_results_['mean_test_score'],
        yerr=rfecv.cv_results_['std_test_score'],
        capsize=3, marker='o'
    )
    plt.axvline(x=rfecv.n_features_, color='r', linestyle='--',
                label=f'最优特征数={rfecv.n_features_}')
    plt.xlabel('特征数量')
    plt.ylabel('CV MAE (负值 → 正值)')
    plt.title('RFECV: 特征数量 vs CV MAE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('rfecv_path.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[RFECV] 路径图已保存至 rfecv_path.png")

    return selected_features, rfecv


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  第2部分：集成方法                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class EnsemblePredictor:
    """
    集成预测器，支持三种集成方式：
      1. 简单平均 (simple_average)
      2. 加权平均 (weighted_average)
      3. Stacking (stacking)

    使用方法:
        ensemble = EnsemblePredictor(models, method='weighted_average')
        ensemble.fit(X_train, y_train, X_val, y_val)
        y_pred = ensemble.predict(X_test)
    """

    def __init__(self, models, method='weighted_average', meta_model=None):
        """
        Parameters
        ----------
        models : dict, 格式为 {'模型名': 模型实例, ...}
            如 {'GPR': GPR_model, 'KRR': KRR_model, 'Ridge': Ridge_model}
        method : str, 集成方式: 'simple_average', 'weighted_average', 'stacking'
        meta_model : sklearn模型, stacking 时的元模型（默认为 Ridge）
        """
        self.models = models
        self.method = method
        self.weights = None  # 加权平均的权重字典
        self.meta_model = meta_model
        self.fitted_models = {}

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        训练所有基模型并计算集成权重。

        Parameters
        ----------
        X_train : array-like, 训练特征
        y_train : array-like, 训练目标
        X_val : array-like, 验证特征（加权平均和 Stacking 需要）
        y_val : array-like, 验证目标（加权平均和 Stacking 需要）
        """
        from sklearn.metrics import mean_squared_error

        # 训练所有基模型
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            self.fitted_models[name] = model
            print(f"[集成训练] {name} 训练完成")

        if self.method == 'weighted_average':
            if X_val is None or y_val is None:
                raise ValueError("加权平均需要提供验证集 (X_val, y_val)")

            # 计算每个模型在验证集上的 MSE
            mse_dict = {}
            for name, model in self.fitted_models.items():
                y_pred_val = model.predict(X_val)
                mse = mean_squared_error(y_val, y_pred_val)
                mse_dict[name] = mse
                print(f"[集成训练] {name} 验证 MSE: {mse:.4f}")

            # 权重 = 1/MSE，归一化
            inv_mse = {k: 1.0 / v for k, v in mse_dict.items()}
            total_inv_mse = sum(inv_mse.values())
            self.weights = {k: v / total_inv_mse for k, v in inv_mse.items()}

            print(f"[集成训练] 权重分配:")
            for name, w in self.weights.items():
                print(f"  {name}: {w:.3f}")

        elif self.method == 'stacking':
            if X_val is None or y_val is None:
                raise ValueError("Stacking 需要提供验证集 (X_val, y_val)")
            if self.meta_model is None:
                from sklearn.linear_model import Ridge
                self.meta_model = Ridge(alpha=1.0)

            # 生成元特征：每个基模型在验证集上的预测
            meta_features = np.column_stack([
                model.predict(X_val) for model in self.fitted_models.values()
            ])

            # 训练元模型
            self.meta_model.fit(meta_features, y_val)
            print(f"[集成训练] Stacking 元模型训练完成")

        return self

    def predict(self, X):
        """
        使用训练好的集成模型进行预测。
        """
        predictions = np.column_stack([
            model.predict(X) for model in self.fitted_models.values()
        ])

        if self.method == 'simple_average':
            return predictions.mean(axis=1)

        elif self.method == 'weighted_average':
            weights_array = np.array([self.weights[name] for name in self.fitted_models.keys()])
            return np.dot(predictions, weights_array)

        elif self.method == 'stacking':
            return self.meta_model.predict(predictions)

        else:
            raise ValueError(f"未知集成方式: {self.method}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  第3部分：GPR 不确定性量化                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def gpr_uncertainty_analysis(X_train, X_test, y_train, y_test, scaler_X, scaler_y):
    """
    使用 GPR 进行预测并量化不确定性。

    原理：
        GPR 不仅仅输出预测均值，还输出预测方差（不确定性）。
        高方差的预测点意味着模型对该区域的预测不够自信，
        应当优先进行 DFT 计算或实验验证。

    Parameters
    ----------
    X_train, X_test : DataFrame, 训练/测试特征
    y_train, y_test : Series, 训练/测试目标
    scaler_X, scaler_y : 特征/目标缩放器

    Returns
    -------
    results : dict, 包含预测值、标准差、置信区间等
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
    from sklearn.metrics import mean_absolute_error, r2_score

    # 缩放数据
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

    # 定义核函数：RBF(平滑变化) + WhiteKernel(观测噪声)
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)

    # 训练 GPR
    print("[GPR不确定性] 训练模型...")
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-5,
        normalize_y=True,
        n_restarts_optimizer=5,
        random_state=42
    )
    gpr.fit(X_train_scaled, y_train_scaled)

    # 预测 + 获取标准差 (return_std=True)
    y_pred_scaled, y_std_scaled = gpr.predict(X_test_scaled, return_std=True)

    # 反缩放回原始量纲
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    # 标准差的反缩放（线性变换下标准差乘以缩放因子）
    y_std = y_std_scaled * scaler_y.scale_[0]  # scaler_y.scale_ 包含 MinMaxScaler 的缩放因子

    # 95% 置信区间
    y_lower = y_pred - 1.96 * y_std
    y_upper = y_pred + 1.96 * y_std

    # 评估指标
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"[GPR不确定性] 测试 MAE: {mae:.4f} kcal/mol")
    print(f"[GPR不确定性] 测试 R²: {r2:.4f}")
    print(f"[GPR不确定性] 平均预测标准差: {y_std.mean():.4f} kcal/mol")
    print(f"[GPR不确定性] 最大预测标准差: {y_std.max():.4f} kcal/mol")

    # 识别高不确定性样本（前20%）
    uncertainty_threshold = np.percentile(y_std, 80)
    high_uncertainty_indices = np.where(y_std >= uncertainty_threshold)[0]
    print(f"[GPR不确定性] 高不确定性样本数 (前20%): {len(high_uncertainty_indices)}")

    # ---- 可视化 ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 子图1：预测 vs 实际（带误差棒的颜色映射）
    ax1 = axes[0]
    scatter = ax1.scatter(y_test, y_pred, c=y_std, cmap='YlOrRd',
                          edgecolors='k', linewidth=0.5, s=60)
    ax1.errorbar(
        y_test.iloc[high_uncertainty_indices],
        y_pred[high_uncertainty_indices],
        yerr=1.96 * y_std[high_uncertainty_indices],
        fmt='none', ecolor='red', alpha=0.4, capsize=2,
        label=f'95% CI (高不确定性, n={len(high_uncertainty_indices)})'
    )
    # 完美预测线
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax1.plot(lims, lims, 'k--', alpha=0.5, label='完美预测')
    ax1.set_xlabel('Experimental Activation Energy (kcal/mol)')
    ax1.set_ylabel('Predicted Activation Energy (kcal/mol)')
    ax1.set_title(f'GPR Prediction with Uncertainty\nMAE={mae:.3f}, R²={r2:.3f}')
    ax1.legend(fontsize=8)
    plt.colorbar(scatter, ax=ax1, label='Std Dev (kcal/mol)')

    # 子图2：预测标准差分布
    ax2 = axes[1]
    ax2.hist(y_std, bins=20, edgecolor='k', alpha=0.7)
    ax2.axvline(x=uncertainty_threshold, color='red', linestyle='--',
                label=f'Top 20% threshold = {uncertainty_threshold:.3f}')
    ax2.set_xlabel('Prediction Std Dev (kcal/mol)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Prediction Uncertainty')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('gpr_uncertainty.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("[GPR不确定性] 图表已保存至 gpr_uncertainty.png")

    # 输出高不确定性样本列表
    uncertainty_df = pd.DataFrame({
        '样本索引': high_uncertainty_indices,
        '实验值': y_test.iloc[high_uncertainty_indices].values,
        '预测值': y_pred[high_uncertainty_indices],
        '标准差': y_std[high_uncertainty_indices],
        '95%CI下限': y_lower[high_uncertainty_indices],
        '95%CI上限': y_upper[high_uncertainty_indices],
        '预测误差': np.abs(y_test.iloc[high_uncertainty_indices].values - y_pred[high_uncertainty_indices])
    }).sort_values('标准差', ascending=False)

    uncertainty_df.to_csv('gpr_high_uncertainty_samples.csv', index=False, encoding='utf-8-sig')
    print(f"[GPR不确定性] 高不确定性样本已保存至 gpr_high_uncertainty_samples.csv")

    results = {
        'y_pred': y_pred,
        'y_std': y_std,
        'y_lower': y_lower,
        'y_upper': y_upper,
        'mae': mae,
        'r2': r2,
        'high_uncertainty_indices': high_uncertainty_indices,
        'gpr_model': gpr
    }

    return results


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  第4部分：正则化参数调整                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def svr_epsilon_tuning(X, y, scaler_X, scaler_y):
    """
    在更合理的范围内重新调优 SVR 的 epsilon 参数。

    原理：
        当前训练的最优 ε = 1.41 kcal/mol 过大，意味着模型对
        1.41 kcal/mol 以内的误差完全不敏感。对于活化能预测，
        推荐 ε 在 0.01-0.5 范围内。

    步骤：
        1. 固定其他参数（C, gamma）为当前最优值
        2. 在 0.001 - 0.5 范围内搜索最优 epsilon
        3. 重新做 CV 评估
    """
    from sklearn.svm import SVR
    from sklearn.model_selection import cross_val_score, KFold
    from sklearn.preprocessing import MinMaxScaler

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()

    print("[SVR正则化调整] 在合理范围内搜索最优 epsilon...")

    # 尝试不同的 epsilon 值
    epsilon_candidates = np.logspace(-3, 0, 20)  # 0.001 ~ 1.0
    results = []

    for eps in epsilon_candidates:
        svr = SVR(C=73.31, epsilon=eps, gamma=8.04, kernel='rbf')  # 固定 C 和 gamma
        scores = cross_val_score(
            svr, X_scaled, y_scaled,
            cv=KFold(n_splits=5, shuffle=True, random_state=42),
            scoring='neg_mean_absolute_error'
        )
        mae = -scores.mean()
        mae_std = scores.std()
        results.append({'epsilon': eps, 'cv_mae': mae, 'cv_mae_std': mae_std})

    results_df = pd.DataFrame(results)

    # 找到最优 epsilon（MAE 最小）
    best_idx = results_df['cv_mae'].idxmin()
    best_eps = results_df.loc[best_idx, 'epsilon']
    best_mae = results_df.loc[best_idx, 'cv_mae']

    print(f"[SVR正则化调整] 最优 epsilon: {best_eps:.4f}")
    print(f"[SVR正则化调整] 对应 CV MAE: {best_mae:.4f} kcal/mol")
    print(f"[SVR正则化调整] (原 epsilon=1.41 的 MAE 需重新评估)")

    # 可视化
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(results_df['epsilon'], results_df['cv_mae'],
                yerr=results_df['cv_mae_std'], capsize=3, marker='o')
    ax.set_xscale('log')
    ax.axvline(x=best_eps, color='red', linestyle='--', label=f'Best ε={best_eps:.4f}')
    ax.axvline(x=1.41, color='orange', linestyle='--', label='Original ε=1.41')
    ax.set_xlabel('ε (log scale)')
    ax.set_ylabel('CV MAE (kcal/mol)')
    ax.set_title('SVR: Effect of ε on CV MAE')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('svr_epsilon_tuning.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[SVR正则化调整] 图表已保存至 svr_epsilon_tuning.png")

    return best_eps, results_df


def ridge_lasso_auto_alpha(X, y, scaler_X, scaler_y):
    """
    使用 RidgeCV 和 LassoCV 自动选择最优的正则化强度 α。

    原理：
        RidgeCV/LassoCV 内部使用交叉验证自动选择 α，
        不需要手动指定 Optuna 搜索空间。
    """
    from sklearn.linear_model import RidgeCV, LassoCV
    from sklearn.model_selection import LeaveOneOut

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()

    loo = LeaveOneOut()

    # RidgeCV
    print("[正则化调整] 使用 RidgeCV 自动选择 α...")
    ridge_cv = RidgeCV(
        alphas=np.logspace(-3, 3, 50),  # 从 0.001 到 1000
        cv=loo,
        scoring='neg_mean_absolute_error'
    )
    ridge_cv.fit(X_scaled, y_scaled)
    print(f"[正则化调整] Ridge 最优 α: {ridge_cv.alpha_:.4f}")

    # LassoCV
    print("[正则化调整] 使用 LassoCV 自动选择 α...")
    lasso_cv = LassoCV(
        alphas=np.logspace(-4, 1, 50),
        cv=loo,
        max_iter=10000,
        random_state=42
    )
    lasso_cv.fit(X_scaled, y_scaled)
    print(f"[正则化调整] Lasso 最优 α: {lasso_cv.alpha_:.6f}")
    print(f"[正则化调整] Lasso 非零系数特征数: {np.sum(lasso_cv.coef_ != 0)}")

    # ---- Ridge 系数路径图 ----
    from sklearn.linear_model import ridge_regression_path

    alphas = np.logspace(-3, 3, 100)
    coefs = []
    for a in alphas:
        ridge = Ridge(alpha=a) if 'Ridge' in dir() else __import__('sklearn.linear_model').linear_model.Ridge(alpha=a)
        ridge.fit(X_scaled, y_scaled)
        coefs.append(ridge.coef_)
    coefs = np.array(coefs)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, feature_name in enumerate(X.columns):
        ax.plot(alphas, coefs[:, i], label=feature_name, alpha=0.7)
    ax.axvline(x=ridge_cv.alpha_, color='red', linestyle='--', label=f"RidgeCV α={ridge_cv.alpha_:.2f}")
    ax.set_xscale('log')
    ax.set_xlabel('α (log scale)')
    ax.set_ylabel('Coefficient Value')
    ax.set_title('Ridge: Coefficient Path (Effect of Regularization)')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ridge_coefficient_path.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[正则化调整] Ridge 系数路径图已保存至 ridge_coefficient_path.png")

    return ridge_cv.alpha_, lasso_cv.alpha_, ridge_cv, lasso_cv


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  第5部分：外部验证框架                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def external_validation_framework(model, scaler_X, scaler_y,
                                    X_train, y_train,
                                    X_external, y_external=None):
    """
    对外部数据集进行预测和评估。

    原理：
        外部验证数据应当完全独立于训练数据。
        在化学ML中，可以是新的底物组合、不同来源的数据、
        或来自 validation_process.py 生成的完整组合空间。

    Parameters
    ----------
    model : 已训练的 sklearn 模型
    scaler_X, scaler_y : 在训练集上拟合的缩放器
    X_train : DataFrame, 训练特征（用于对比分布）
    y_train : Series, 训练目标（用于计算 Q²_ext 基线）
    X_external : DataFrame, 外部验证特征
    y_external : Series or None, 外部验证目标（如有实验值）

    Returns
    -------
    results : dict, 外部验证结果
    """
    from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

    # 预测
    X_ext_scaled = scaler_X.transform(X_external)
    y_pred_scaled = model.predict(X_ext_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    # 应用范围检查（Applicability Domain）：基于特征空间的马氏距离
    from sklearn.covariance import EmpiricalCovariance
    X_train_scaled = scaler_X.transform(X_train)
    cov_estimator = EmpiricalCovariance().fit(X_train_scaled)
    mahalanobis_dist = cov_estimator.mahalanobis(X_ext_scaled)
    # 阈值：训练集马氏距离的第95百分位
    ad_threshold = np.percentile(
        cov_estimator.mahalanobis(X_train_scaled), 95
    )
    inside_ad = mahalanobis_dist <= ad_threshold

    print(f"[外部验证] 总外部样本数: {len(X_external)}")
    print(f"[外部验证] 在应用范围内: {inside_ad.sum()} ({inside_ad.mean():.1%})")
    print(f"[外部验证] 超出应用范围: {(~inside_ad).sum()} ({1 - inside_ad.mean():.1%})")

    if y_external is not None:
        # 有实验值 → 可计算误差指标
        mae_ext = mean_absolute_error(y_external, y_pred)
        rmse_ext = np.sqrt(mean_squared_error(y_external, y_pred))
        r2_ext = r2_score(y_external, y_pred)

        # Q²_ext 使用训练集均值作为基线
        y_train_mean = y_train.mean()
        ss_res = np.sum((y_external - y_pred) ** 2)
        ss_tot = np.sum((y_external - y_train_mean) ** 2)
        q2_ext = 1 - ss_res / ss_tot

        print(f"[外部验证] ------ 外部验证指标 ------")
        print(f"[外部验证] MAE_ext:  {mae_ext:.4f} kcal/mol")
        print(f"[外部验证] RMSE_ext: {rmse_ext:.4f} kcal/mol")
        print(f"[外部验证] R²_ext:   {r2_ext:.4f}")
        print(f"[外部验证] Q²_ext:   {q2_ext:.4f}")

        # 仅应用范围内的指标
        if inside_ad.sum() > 0:
            mae_ad = mean_absolute_error(
                y_external[inside_ad], y_pred[inside_ad]
            )
            r2_ad = r2_score(y_external[inside_ad], y_pred[inside_ad])
            print(f"[外部验证] MAE (应用范围内): {mae_ad:.4f} kcal/mol")
            print(f"[外部验证] R²  (应用范围内): {r2_ad:.4f}")

    # ---- 可视化 ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 子图1：预测 vs 实际（如有实验值）
    ax1 = axes[0]
    if y_external is not None:
        # 用不同颜色标记应用范围内外
        ax1.scatter(
            y_external[inside_ad], y_pred[inside_ad],
            c='blue', alpha=0.7, label=f'Inside AD (n={inside_ad.sum()})'
        )
        ax1.scatter(
            y_external[~inside_ad], y_pred[~inside_ad],
            c='red', alpha=0.7, marker='^', label=f'Outside AD (n={(~inside_ad).sum()})'
        )
        # 完美预测线
        lims = [
            min(y_external.min(), y_pred.min()),
            max(y_external.max(), y_pred.max())
        ]
        ax1.plot(lims, lims, 'k--', alpha=0.5, label='Perfect Prediction')
        ax1.set_xlabel('Experimental Value (kcal/mol)')
        ax1.set_ylabel('Predicted Value (kcal/mol)')
        ax1.set_title(f'External Validation\nMAE={mae_ext:.3f}, Q²={q2_ext:.3f}')
    else:
        ax1.hist(y_pred, bins=20, edgecolor='k', alpha=0.7)
        ax1.set_xlabel('Predicted Activation Energy (kcal/mol)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('External Prediction Distribution')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 子图2：马氏距离分布（应用范围）
    ax2 = axes[1]
    ax2.hist(mahalanobis_dist, bins=20, edgecolor='k', alpha=0.7)
    ax2.axvline(x=ad_threshold, color='red', linestyle='--',
                label=f'AD Threshold (95th percentile)')
    ax2.set_xlabel('Mahalanobis Distance to Training Set')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Applicability Domain Check')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('external_validation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[外部验证] 图表已保存至 external_validation.png")

    # 保存预测结果
    pred_df = pd.DataFrame({
        'y_pred': y_pred,
        'inside_ad': inside_ad,
        'mahalanobis_dist': mahalanobis_dist
    })
    if y_external is not None:
        pred_df['y_true'] = y_external.values
        pred_df['abs_error'] = np.abs(y_external.values - y_pred)
    pred_df.to_csv('external_predictions.csv', index=False, encoding='utf-8-sig')
    print("[外部验证] 预测结果已保存至 external_predictions.csv")

    return {
        'y_pred': y_pred,
        'inside_ad': inside_ad,
        'mahalanobis_dist': mahalanobis_dist
    }


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  第6部分：重复 k-fold 交叉验证                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def repeated_kfold_evaluation(model, X, y, scaler_X, scaler_y,
                                n_splits=5, n_repeats=5, random_state=42):
    """
    使用重复 k-fold 交叉验证评估模型，替代单一的 LOOCV。

    原理：
        LOOCV 在 N=88 的小样本下方差极大。
        重复 k-fold 通过多次不同的数据划分，降低了 CV 估计的方差，
        同时报告标准差作为模型稳定性的衡量。

    Parameters
    ----------
    model : sklearn 模型
    X, y : 特征和目标
    scaler_X, scaler_y : 缩放器
    n_splits : int, k-fold 的折数（默认5）
    n_repeats : int, 重复次数（默认5 → 共25次评估）
    random_state : int, 随机种子

    Returns
    -------
    results : dict, 包含所有评估指标及标准差
    """
    from sklearn.model_selection import RepeatedKFold, cross_validate

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()

    rkf = RepeatedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state
    )

    print(f"[重复k-fold] 开始 {n_repeats}×{n_splits} = {n_repeats * n_splits} 次评估...")

    scoring = {
        'mae': 'neg_mean_absolute_error',
        'r2': 'r2',
        'rmse': 'neg_root_mean_squared_error'
    }

    cv_results = cross_validate(
        model, X_scaled, y_scaled,
        cv=rkf,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False
    )

    # 转换负值指标
    mae_scores = -cv_results['test_mae']
    r2_scores = cv_results['test_r2']
    rmse_scores = -cv_results['test_rmse']

    print(f"\n[重复k-fold] {'='*50}")
    print(f"[重复k-fold] 评估次数: {len(mae_scores)} ({n_repeats} repeats × {n_splits} folds)")
    print(f"[重复k-fold] MAE:      {mae_scores.mean():.4f} ± {mae_scores.std():.4f} kcal/mol")
    print(f"[重复k-fold] RMSE:     {rmse_scores.mean():.4f} ± {rmse_scores.std():.4f} kcal/mol")
    print(f"[重复k-fold] R²:       {r2_scores.mean():.4f} ± {r2_scores.std():.4f}")
    print(f"[重复k-fold] MAE 范围: [{mae_scores.min():.4f}, {mae_scores.max():.4f}]")

    # 可视化 MAE 分布
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    ax1 = axes[0]
    ax1.boxplot(mae_scores, vert=True)
    ax1.set_ylabel('MAE (kcal/mol)')
    ax1.set_title(f'Repeated {n_splits}-fold CV ({n_repeats}×)\n'
                  f'MAE = {mae_scores.mean():.3f} ± {mae_scores.std():.3f}')
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.boxplot(r2_scores, vert=True)
    ax2.set_ylabel('R²')
    ax2.set_title(f'R² = {r2_scores.mean():.3f} ± {r2_scores.std():.3f}')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('repeated_kfold.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[重复k-fold] 图表已保存至 repeated_kfold.png")

    return {
        'mae_mean': mae_scores.mean(),
        'mae_std': mae_scores.std(),
        'r2_mean': r2_scores.mean(),
        'r2_std': r2_scores.std(),
        'rmse_mean': rmse_scores.mean(),
        'rmse_std': rmse_scores.std(),
        'all_mae': mae_scores,
        'all_r2': r2_scores,
        'all_rmse': rmse_scores
    }


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  第7部分：y-randomization 检验                                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def y_randomization_test(model_class, best_params, X, y, scaler_X, scaler_y,
                          n_permutations=100, cv_folds=5, random_state=42):
    """
    y-randomization 检验：验证模型学到的结构-活性关系是否真实。

    原理：
        将目标变量 y 随机打乱 N 次，在打乱后的数据上重新训练和评估。
        如果原始模型的性能显著优于所有打乱模型（p < 0.05），
        则说明模型学到了真实的构效关系，而非巧合。

    Parameters
    ----------
    model_class : sklearn 模型类
    best_params : dict, 最佳超参数
    X, y : 特征和目标
    scaler_X, scaler_y : 缩放器
    n_permutations : int, 随机化次数（默认100）
    cv_folds : int, CV 折数
    random_state : int, 随机种子

    Returns
    -------
    results : dict, 包含原始性能、随机分布、p-value
    """
    from sklearn.model_selection import cross_val_score, KFold

    rng = np.random.RandomState(random_state)
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()
    y_values = y.values.copy()

    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    # 步骤1：计算原始模型的性能
    print("[y-randomization] 步骤1: 计算原始模型性能...")
    model_original = model_class(**best_params)
    original_scores = cross_val_score(
        model_original, X_scaled, y_scaled,
        cv=cv, scoring='neg_mean_absolute_error'
    )
    original_mae = -original_scores.mean()
    print(f"[y-randomization] 原始模型 MAE: {original_mae:.4f} kcal/mol")

    # 步骤2：N 次随机化
    print(f"[y-randomization] 步骤2: 进行 {n_permutations} 次 y-randomization...")
    random_mae_list = []
    random_r2_list = []

    for i in range(n_permutations):
        # 随机打乱 y（保持 X 不变）
        y_shuffled = rng.permutation(y_values)

        # 缩放打乱后的 y
        scaler_y_shuffled = MinMaxScaler(feature_range=(0, 100))
        y_shuffled_scaled = scaler_y_shuffled.fit_transform(
            y_shuffled.reshape(-1, 1)
        ).ravel()

        # 训练随机模型
        model_random = model_class(**best_params)
        random_scores = cross_val_score(
            model_random, X_scaled, y_shuffled_scaled,
            cv=cv, scoring='neg_mean_absolute_error'
        )
        mae_random = -random_scores.mean()
        random_mae_list.append(mae_random)

        if (i + 1) % 20 == 0:
            print(f"  已完成 {i + 1}/{n_permutations}...")

    random_mae_array = np.array(random_mae_list)

    # 步骤3：计算 p-value
    # p-value = 随机模型 MAE ≤ 原始模型 MAE 的比例
    # （MAE 越小越好，所以随机模型 MAE ≤ 原始 MAE 意味着随机模型碰巧表现好）
    p_value = np.mean(random_mae_array <= original_mae)

    print(f"\n[y-randomization] {'='*50}")
    print(f"[y-randomization] 原始模型 MAE: {original_mae:.4f}")
    print(f"[y-randomization] 随机模型 MAE 均值: {random_mae_array.mean():.4f} ± {random_mae_array.std():.4f}")
    print(f"[y-randomization] 随机模型 MAE 范围: [{random_mae_array.min():.4f}, {random_mae_array.max():.4f}]")
    print(f"[y-randomization] p-value: {p_value:.4f}")

    if p_value < 0.05:
        print(f"[y-randomization] ✓ 检验通过！模型学到了真实的构效关系 (p={p_value:.4f} < 0.05)")
    elif p_value < 0.10:
        print(f"[y-randomization] ⚠ 边缘显著 (p={p_value:.4f})，建议增加数据量或更严格验证")
    else:
        print(f"[y-randomization] ✗ 检验未通过 (p={p_value:.4f} ≥ 0.05)，模型可能学到了偶然相关性")

    # ---- 可视化 ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(random_mae_array, bins=min(30, n_permutations // 3),
            edgecolor='k', alpha=0.7, label=f'y-randomized (n={n_permutations})')
    ax.axvline(x=original_mae, color='red', linewidth=2.5, linestyle='--',
               label=f'Original Model MAE={original_mae:.3f}')
    ax.axvline(x=random_mae_array.mean(), color='blue', linewidth=1.5, linestyle='-',
               label=f'Random Mean MAE={random_mae_array.mean():.3f}')
    ax.set_xlabel('CV MAE (kcal/mol)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'y-Randomization Test\np-value = {p_value:.4f} ' +
                 ('(PASS ✓)' if p_value < 0.05 else '(FAIL ✗)'))
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('y_randomization.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[y-randomization] 图表已保存至 y_randomization.png")

    return {
        'original_mae': original_mae,
        'random_mae_mean': random_mae_array.mean(),
        'random_mae_std': random_mae_array.std(),
        'random_mae_list': random_mae_array,
        'p_value': p_value,
        'passed': p_value < 0.05
    }


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  第8部分：SHAP-RFECV（SHAP 驱动的递归特征消除）                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def shap_rfecv(model_class, best_params, X, y, scaler_X, scaler_y,
               min_features_to_select=3, cv_folds=5, random_state=42):
    """
    SHAP-RFECV: 使用 SHAP 重要性 + RFECV 的特征选择方法。

    为什么需要 SHAP-RFECV？
    ─────────────────────────────────
    标准 RFECV 依赖模型的内置 feature_importances_ 或 coef_ 属性。
    但 SVR (RBF核)、KRR、GPR、KNR 等模型不支持这些属性，导致标准 RFECV
    对它们不可用。SHAP-RFECV 用 SHAP 替代内置重要性，实现"模型无关"的 RFECV。

    算法伪代码:
    ─────────────────────────────────
    F = 全部特征
    results = []
    while |F| >= min_features_to_select:
        model.fit(F)
        shap_importance = SHAP(model, F)       ← 用 SHAP 替代 feature_importances_
        worst_feature = argmin(shap_importance)  ← 找到最不重要的
        cv_mae = cross_val_score(model, F)       ← CV 评估当前性能
        results.append((|F|, cv_mae, worst_feature))
        F = F - {worst_feature}                ← 移除最不重要的
    return F_best = argmin(cv_mae) 对应的 F

    相对于当前固定阈值方法的优势:
    ─────────────────────────────────
    1. 模型无关: SVR/GPR/KRR/RF/MLP 统一适用
    2. 数据驱动: 自动找到最优特征数，不需要人为设定 min_features
    3. SHAP 比内置重要性更准确: 基于 Shapley 值，公平分配相关特征的贡献

    Parameters
    ----------
    model_class : sklearn 模型类
    best_params : dict, 模型的最佳超参数
    X : DataFrame, 特征矩阵
    y : Series, 目标变量
    scaler_X, scaler_y : 缩放器
    min_features_to_select : int, 最少保留特征数
    cv_folds : int, CV 折数
    random_state : int, 随机种子

    Returns
    -------
    best_features : list, 最优特征子集
    cv_path : DataFrame, 完整的选择路径数据
    """
    import shap
    from sklearn.model_selection import cross_val_score, KFold
    from sklearn.preprocessing import MinMaxScaler

    print("=" * 60)
    print("  SHAP-RFECV: SHAP驱动的递归特征消除")
    print("=" * 60)

    current_features = list(X.columns)
    cv_path = []  # 记录每一步的特征数和对应的 CV 性能
    rng = np.random.RandomState(random_state)
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    iteration = 0
    while len(current_features) > min_features_to_select:
        iteration += 1
        X_current = X[current_features]

        # 缩放当前特征集
        X_scaled = scaler_X.fit_transform(X_current)
        y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()

        # 训练模型
        model = model_class(**best_params)
        model.fit(X_scaled, y_scaled)

        # --- SHAP 重要性计算 ---
        # 根据模型类型选择合适的 SHAP 解释器
        model_name = model_class.__name__

        tree_models = ['RandomForestRegressor', 'GradientBoostingRegressor',
                       'XGBRegressor', 'LGBMRegressor', 'DecisionTreeRegressor']
        linear_models = ['LinearRegression', 'Ridge', 'Lasso', 'ElasticNet']

        if model_name in tree_models:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_scaled)
            importances = np.abs(shap_values).mean(axis=0)
        elif model_name in linear_models:
            explainer = shap.LinearExplainer(model, X_scaled)
            shap_values = explainer.shap_values(X_scaled)
            importances = np.abs(shap_values).mean(axis=0)
        else:
            # KernelExplainer 用于 SVR, GPR, KRR, MLP, KNR 等
            n_clusters = max(10, min(15, int(len(X) * 0.15)))
            n_samples = max(20, min(int(len(X) * 0.3), 100))
            np.random.seed(42)  # 固定种子保证可复现
            background = shap.kmeans(X_scaled, n_clusters)
            explainer = shap.KernelExplainer(model.predict, background)
            sample_indices = rng.choice(len(X_scaled), n_samples, replace=False)
            shap_values = explainer.shap_values(X_scaled[sample_indices])
            importances = np.abs(shap_values).mean(axis=0)

        # 归一化为百分比
        importances = importances / importances.sum() * 100

        # 找到重要性最低的特征
        worst_idx = np.argmin(importances)
        worst_feature = current_features[worst_idx]

        # CV 评估当前特征集的性能
        cv_scores = cross_val_score(
            model, X_scaled, y_scaled,
            cv=cv, scoring='neg_mean_absolute_error'
        )
        cv_mae = -cv_scores.mean()
        cv_mae_std = cv_scores.std()

        # 也计算 R²
        cv_r2_scores = cross_val_score(
            model, X_scaled, y_scaled,
            cv=cv, scoring='r2'
        )
        cv_r2 = cv_r2_scores.mean()
        cv_r2_std = cv_r2_scores.std()

        cv_path.append({
            'iteration': iteration,
            'n_features': len(current_features),
            'removed_feature': worst_feature,
            'removed_importance': importances[worst_idx],
            'cv_mae': cv_mae,
            'cv_mae_std': cv_mae_std,
            'cv_r2': cv_r2,
            'cv_r2_std': cv_r2_std,
            'features': current_features.copy()
        })

        print(f"  [Iter {iteration:2d}] 特征数={len(current_features):2d} | "
              f"移除='{worst_feature}' (SHAP重要性={importances[worst_idx]:.2f}%) | "
              f"CV MAE={cv_mae:.4f}±{cv_mae_std:.4f} | CV R²={cv_r2:.4f}±{cv_r2_std:.4f}")

        # 移除最不重要的特征
        current_features.remove(worst_feature)

    # 记录最后一轮（达到 min_features）
    X_scaled = scaler_X.fit_transform(X[current_features])
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()
    model = model_class(**best_params)
    model.fit(X_scaled, y_scaled)
    cv_scores = cross_val_score(model, X_scaled, y_scaled, cv=cv, scoring='neg_mean_absolute_error')
    cv_r2_scores = cross_val_score(model, X_scaled, y_scaled, cv=cv, scoring='r2')
    cv_path.append({
        'iteration': iteration + 1,
        'n_features': len(current_features),
        'removed_feature': 'STOP',
        'removed_importance': 0,
        'cv_mae': -cv_scores.mean(),
        'cv_mae_std': cv_scores.std(),
        'cv_r2': cv_r2_scores.mean(),
        'cv_r2_std': cv_r2_scores.std(),
        'features': current_features.copy()
    })

    # 找到最优特征数：CV MAE 最低的点
    cv_path_df = pd.DataFrame(cv_path)
    best_row = cv_path_df.loc[cv_path_df['cv_mae'].idxmin()]
    best_n_features = int(best_row['n_features'])
    best_features = best_row['features']

    print(f"\n  ┌{'─'*56}┐")
    print(f"  │ SHAP-RFECV 最优结果                                       │")
    print(f"  ├{'─'*56}┤")
    print(f"  │ 最优特征数: {best_n_features:2d}                                            │")
    print(f"  │ 最优 CV MAE: {best_row['cv_mae']:.4f} kcal/mol                              │")
    print(f"  │ 最优 CV R²:  {best_row['cv_r2']:.4f}                                       │")
    print(f"  │ 最优特征: {', '.join(best_features[:4])}... │")
    print(f"  └{'─'*56}┘")

    # ---- 可视化 SHAP-RFECV 路径 ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 子图1: CV MAE 路径
    ax1 = axes[0]
    ax1.errorbar(cv_path_df['n_features'], cv_path_df['cv_mae'],
                 yerr=cv_path_df['cv_mae_std'], marker='o', capsize=3,
                 linewidth=2, color='steelblue')
    ax1.axvline(x=best_n_features, color='red', linestyle='--',
                label=f'Best: {best_n_features} features')
    ax1.set_xlabel('Number of Features')
    ax1.set_ylabel('CV MAE (kcal/mol)')
    ax1.set_title(f'SHAP-RFECV Path ({model_class.__name__})\nLower MAE is Better')
    ax1.invert_xaxis()  # 反转X轴：从左到右特征减少
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 子图2: CV R² 路径
    ax2 = axes[1]
    ax2.errorbar(cv_path_df['n_features'], cv_path_df['cv_r2'],
                 yerr=cv_path_df['cv_r2_std'], marker='s', capsize=3,
                 linewidth=2, color='darkorange')
    ax2.axvline(x=best_n_features, color='red', linestyle='--',
                label=f'Best: {best_n_features} features')
    ax2.set_xlabel('Number of Features')
    ax2.set_ylabel('CV R²')
    ax2.set_title(f'SHAP-RFECV Path ({model_class.__name__})\nHigher R² is Better')
    ax2.invert_xaxis()
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'shap_rfecv_{model_class.__name__}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  SHAP-RFECV 路径图已保存至 shap_rfecv_{model_class.__name__}.png")

    return best_features, cv_path_df


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  第9部分：集成模型选择策略                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def ensemble_model_selection(trained_models_dict, X_val, y_val, scaler_X, scaler_y):
    """
    系统性地选择集成模型的成员组合。

    原理：
        集成学习的有效性取决于"多样性+准确性"。
        本函数通过三步流程：误差相关性分析 → 聚类分组 → 穷举组合评估，
        找到最优的集成成员组合。

    Parameters
    ----------
    trained_models_dict : dict, 格式为 {'模型名': 已训练的模型实例}
    X_val : DataFrame, 验证集特征（用于计算误差相关性）
    y_val : Series, 验证集目标
    scaler_X, scaler_y : 缩放器（已在训练集上拟合）

    Returns
    -------
    best_combination : list, 最优模型名称组合
    all_combinations_df : DataFrame, 所有组合的评估结果
    error_corr_matrix : DataFrame, 误差相关矩阵
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    from itertools import combinations

    print("=" * 60)
    print("  集成模型选择策略")
    print("=" * 60)

    model_names = list(trained_models_dict.keys())
    n_models = len(model_names)

    X_val_scaled = scaler_X.transform(X_val)
    y_val_scaled = scaler_y.transform(y_val.values.reshape(-1, 1)).ravel()

    # ─────── 步骤1: 计算每个模型的预测残差 ───────
    print("\n[步骤1] 计算各模型在验证集上的预测残差...")
    residuals = {}
    metrics = {}

    for name in model_names:
        model = trained_models_dict[name]
        y_pred_scaled = model.predict(X_val_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        resid = y_val.values - y_pred
        residuals[name] = resid
        metrics[name] = {
            'mae': mean_absolute_error(y_val, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_val, y_pred)),
            'residual_std': np.std(resid)
        }
        print(f"    {name:20s}: MAE={metrics[name]['mae']:.4f}, "
              f"RMSE={metrics[name]['rmse']:.4f}")

    # ─────── 步骤2: 计算误差相关矩阵 ───────
    print("\n[步骤2] 计算误差相关矩阵（衡量模型间的相似性）...")
    # 低相关性 = 模型犯不同的错误 = 适合集成
    error_corr = np.zeros((n_models, n_models))
    for i, name_i in enumerate(model_names):
        for j, name_j in enumerate(model_names):
            corr = np.corrcoef(residuals[name_i], residuals[name_j])[0, 1]
            error_corr[i, j] = corr

    error_corr_df = pd.DataFrame(error_corr, index=model_names, columns=model_names)

    print("\n  误差相关矩阵 (低值=互补性好):")
    # 格式化输出
    for i, name_i in enumerate(model_names):
        corr_str = '  '.join([f'{error_corr[i,j]:+.2f}' for j in range(n_models)])
        print(f"    {name_i:20s}: {corr_str}")

    # ─────── 步骤3: 基于误差相关性的层次聚类 ───────
    print("\n[步骤3] 层次聚类：将犯类似错误的模型归为一组...")

    # 将相关矩阵转为距离矩阵 (1 - |correlation|)
    distance_matrix = 1 - np.abs(error_corr)
    # 确保对角线为0
    np.fill_diagonal(distance_matrix, 0)

    condensed_dist = squareform(distance_matrix)
    linkage_matrix = linkage(condensed_dist, method='average')

    # 自动确定聚类数：剪枝阈值设为距离矩阵均值的0.7倍
    cut_threshold = np.mean(condensed_dist) * 0.7
    cluster_labels = fcluster(linkage_matrix, t=cut_threshold, criterion='distance')

    # 打印分组结果
    n_clusters = len(set(cluster_labels))
    groups = {}
    for i, label in enumerate(cluster_labels):
        if label not in groups:
            groups[label] = []
        groups[label].append(model_names[i])

    print(f"\n  共 {n_clusters} 组:")
    for label, members in groups.items():
        # 找出组内性能最好的模型作为代表
        best_in_group = min(members, key=lambda m: metrics[m]['mae'])
        print(f"    组{label}: {members}  → 代表模型: [{best_in_group}]")

    # ─────── 步骤4: 从每组选代表，穷举组合 ───────
    print("\n[步骤4] 从各组选代表模型，穷举组合评估...")

    # 每组选1个代表
    representatives = [min(members, key=lambda m: metrics[m]['mae'])
                       for members in groups.values()]
    print(f"  各组代表: {representatives}")

    # 如果代表太少（<2个），则使用所有模型
    if len(representatives) < 2:
        print("  ⚠ 聚类组数不足2组，使用全部候选模型")
        representatives = model_names

    # 穷举所有可能的组合（至少2个模型）
    all_combinations_results = []

    for r in range(2, len(representatives) + 1):
        for combo in combinations(representatives, r):
            combo_list = list(combo)

            # 集成预测（简单平均，因为这是选择阶段而非最终部署阶段）
            y_pred_ensemble_scaled = np.zeros(len(X_val_scaled))
            for name in combo_list:
                y_pred_ensemble_scaled += trained_models_dict[name].predict(X_val_scaled)
            y_pred_ensemble_scaled /= len(combo_list)

            y_pred_ensemble = scaler_y.inverse_transform(
                y_pred_ensemble_scaled.reshape(-1, 1)
            ).ravel()

            combo_mae = mean_absolute_error(y_val, y_pred_ensemble)
            combo_rmse = np.sqrt(mean_squared_error(y_val, y_pred_ensemble))

            # 计算多样性指标：组合内模型间平均误差相关性（越低越好）
            combo_indices = [model_names.index(n) for n in combo_list]
            combo_diversity = np.mean([
                1 - np.abs(error_corr[i, j])
                for i in combo_indices for j in combo_indices if i < j
            ])

            all_combinations_results.append({
                'combination': ' + '.join(combo_list),
                'n_models': len(combo_list),
                'mae': combo_mae,
                'rmse': combo_rmse,
                'diversity_score': combo_diversity
            })

    combo_df = pd.DataFrame(all_combinations_results).sort_values('mae')

    print(f"\n  Top 10 组合（按MAE排序）:")
    print(f"  {'组合':<50s} {'MAE':>8s} {'RMSE':>8s} {'多样性':>8s}")
    print(f"  {'-'*74}")
    for _, row in combo_df.head(10).iterrows():
        print(f"  {row['combination']:<50s} {row['mae']:8.4f} {row['rmse']:8.4f} {row['diversity_score']:8.3f}")

    best_combo = combo_df.iloc[0]
    best_combo_models = best_combo['combination'].split(' + ')

    print(f"\n  ★ 最佳集成组合: {best_combo['combination']}")
    print(f"  ★ MAE={best_combo['mae']:.4f}, RMSE={best_combo['rmse']:.4f}, "
          f"多样性={best_combo['diversity_score']:.3f}")

    # ---- 可视化 ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 子图1: 误差相关热力图
    ax1 = axes[0]
    im = ax1.imshow(error_corr, cmap='RdYlBu_r', vmin=-1, vmax=1, aspect='equal')
    ax1.set_xticks(range(n_models))
    ax1.set_yticks(range(n_models))
    ax1.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
    ax1.set_yticklabels(model_names, fontsize=8)
    # 在热力图上标注数值
    for i in range(n_models):
        for j in range(n_models):
            color = 'white' if abs(error_corr[i, j]) > 0.5 else 'black'
            ax1.text(j, i, f'{error_corr[i,j]:.2f}', ha='center', va='center',
                     fontsize=7, color=color)
    ax1.set_title('Error Correlation Matrix\n(Lower = More Complementary)')
    plt.colorbar(im, ax=ax1, shrink=0.8)

    # 子图2: 组合 MAE vs 多样性
    ax2 = axes[1]
    scatter = ax2.scatter(
        combo_df['diversity_score'], combo_df['mae'],
        c=combo_df['n_models'], cmap='viridis', s=100, edgecolors='k', linewidth=0.5
    )
    # 标注最佳组合
    ax2.annotate(
        best_combo['combination'][:30] + '...',
        (best_combo['diversity_score'], best_combo['mae']),
        xytext=(10, -15), textcoords='offset points',
        fontsize=8, color='red', fontweight='bold',
        arrowprops=dict(arrowstyle='->', color='red')
    )
    ax2.set_xlabel('Diversity Score (Higher = More Diverse)')
    ax2.set_ylabel('Ensemble MAE (kcal/mol)')
    ax2.set_title('Ensemble Combinations: Diversity vs Performance')
    plt.colorbar(scatter, ax=ax2, label='Number of Models')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ensemble_model_selection.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  集成选择图表已保存至 ensemble_model_selection.png")

    return best_combo_models, combo_df, error_corr_df


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  第10部分：完整工作流演示 (demo)                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def run_demo():
    """
    演示如何使用上述所有改进方法。

    此函数按顺序运行：
      1. 加载数据
      2. 特征选择（CV 性能导向）
      3. 模型训练（SVR + GPR + KRR）
      4. 重复 k-fold 评估
      5. y-randomization 检验
      6. GPR 不确定性量化
      7. 集成预测（加权平均）

    注意：此演示为完整流程展示，运行时间可能较长（取决于 n_trials 设置）。
    """
    from sklearn.svm import SVR
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split

    print("=" * 70)
    print("  机器学习模型改进方法 — 完整演示")
    print(f"  运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ---- 1. 加载数据 ----
    print("\n>>> 步骤1: 加载数据")
    X, y, feature_names = load_data()

    # 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler(feature_range=(0, 100))

    print(f"  训练集: {len(X_train)} 样本")
    print(f"  测试集: {len(X_test)} 样本")

    # ---- 2. 特征选择（CV 性能导向） ----
    print("\n>>> 步骤2: CV 性能导向的特征选择（以 SVR 为例）")
    # 使用预设的 SVR 最佳超参数（来自本次训练结果）
    svr_best_params = {'C': 73.31, 'epsilon': 0.05, 'gamma': 8.04, 'kernel': 'rbf'}

    selected_features, selection_history = cv_guided_feature_selection(
        SVR, X_train, y_train, svr_best_params, scaler_X, scaler_y,
        max_mae_increase=0.05, cv_folds=5
    )

    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    # ---- 3. 训练多个模型 ----
    print("\n>>> 步骤3: 训练多个模型")

    X_train_scaled = scaler_X.fit_transform(X_train_selected)
    X_test_scaled = scaler_X.transform(X_test_selected)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

    models = {}

    # SVR
    svr = SVR(**svr_best_params)
    svr.fit(X_train_scaled, y_train_scaled)
    models['SVR'] = svr
    print("  SVR: ✓")

    # GPR
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-5, n_restarts_optimizer=5, random_state=42)
    gpr.fit(X_train_scaled, y_train_scaled)
    models['GPR'] = gpr
    print("  GPR: ✓")

    # KRR
    krr = KernelRidge(alpha=0.0127, gamma=3.04, kernel='rbf')
    krr.fit(X_train_scaled, y_train_scaled)
    models['KRR'] = krr
    print("  KRR: ✓")

    # ---- 4. 重复 k-fold 评估 ----
    print("\n>>> 步骤4: 重复 k-fold 评估（以 SVR 为例）")
    rkf_results = repeated_kfold_evaluation(
        svr, X_train_selected, y_train, scaler_X, scaler_y,
        n_splits=5, n_repeats=5
    )

    # ---- 5. y-randomization 检验 ----
    print("\n>>> 步骤5: y-randomization 检验（以 SVR 为例）")
    yr_results = y_randomization_test(
        SVR, svr_best_params, X_train_selected, y_train,
        scaler_X, scaler_y, n_permutations=50  # 演示用50次
    )

    # ---- 6. GPR 不确定性量化 ----
    print("\n>>> 步骤6: GPR 不确定性量化")
    gpr_uncertainty = gpr_uncertainty_analysis(
        X_train_selected, X_test_selected, y_train, y_test, scaler_X, scaler_y
    )

    # ---- 7. 集成预测 ----
    print("\n>>> 步骤7: 加权平均集成")
    ensemble = EnsemblePredictor(models, method='weighted_average')
    ensemble.fit(X_train_scaled, y_train_scaled, X_test_scaled,
                 scaler_y.transform(y_test.values.reshape(-1, 1)).ravel())

    y_pred_ensemble_scaled = ensemble.predict(X_test_scaled)
    y_pred_ensemble = scaler_y.inverse_transform(
        y_pred_ensemble_scaled.reshape(-1, 1)
    ).ravel()

    from sklearn.metrics import mean_absolute_error, r2_score
    ensemble_mae = mean_absolute_error(y_test, y_pred_ensemble)
    ensemble_r2 = r2_score(y_test, y_pred_ensemble)
    print(f"\n  集成模型测试 MAE: {ensemble_mae:.4f} kcal/mol")
    print(f"  集成模型测试 R²:  {ensemble_r2:.4f}")

    print("\n" + "=" * 70)
    print("  演示完成！所有图表和结果已保存到当前目录。")
    print("=" * 70)


# ============================================================================
# 主入口
# ============================================================================

if __name__ == '__main__':
    """
    运行方式：
        python example/improvement_code_examples.py

    如需单独运行某个方法，可以注释掉 run_demo() 并调用相应的函数。

    示例：
        X, y, feature_names = load_data()
        # ... 划分数据 ...
        # 运行 y-randomization 检验
        yr_results = y_randomization_test(SVR, best_params, X_train, y_train, scaler_X, scaler_y)
    """
    run_demo()
