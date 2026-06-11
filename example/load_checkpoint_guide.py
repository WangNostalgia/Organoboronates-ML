# -*- coding: utf-8 -*-
"""
===============================================================================
  SHAP-RFECV Checkpoint 使用指南
===============================================================================

训练完成后，每个模型在 models/<ModelName>/ 目录下保存了两种 .joblib 文件：

  1. *_iteration_N_<timestamp>.joblib  — 每轮迭代的模型快照（含该轮的特征集、所有指标）
  2. *_final_<timestamp>.joblib        — 最终模型（自动选择或手动指定的最优特征集）

本文件展示如何加载、检查和手动选择任意 checkpoint。
所有代码均独立运行，不修改项目源代码。
===============================================================================
"""

import joblib
import pandas as pd
import numpy as np
import os
import glob


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  示例 1：查看某个模型目录下有哪些 checkpoint                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def list_checkpoints(model_dir):
    """
    列出某个模型目录下所有可用的 checkpoint 文件。

    Usage:
        list_checkpoints('models/SVR')
    """
    # 迭代 checkpoint
    iter_files = sorted(glob.glob(os.path.join(model_dir, '*_iteration_*.joblib')))
    # 最终模型
    final_files = sorted(glob.glob(os.path.join(model_dir, '*_final_*.joblib')))

    print(f"\n{'='*60}")
    print(f"  Checkpoints in: {model_dir}")
    print(f"{'='*60}")

    if final_files:
        print(f"\n  ★ Final model (auto-selected or forced):")
        for f in final_files:
            info = joblib.load(f)
            print(f"    {os.path.basename(f)}")
            print(f"    Features ({len(info['features'])}): {info['features']}")
            print(f"    Metrics: MAE_mean={info['metrics'].get('mae_mean','?'):.4f}, "
                  f"R²_test={info['metrics'].get('r2_test','?'):.4f}, "
                  f"R²_LOO={info['metrics'].get('r2_loo','?'):.4f}")

    if iter_files:
        print(f"\n  Per-iteration checkpoints ({len(iter_files)} total):")
        for f in iter_files:
            info = joblib.load(f)
            n_feat = len(info['features'])
            removed = info.get('removed_features', [])
            last_removed = removed[-1] if removed else 'Initial'
            print(f"    {os.path.basename(f):<60s} "
                  f"features={n_feat:2d}  last_removed='{last_removed}'")
    else:
        print("  (No iteration checkpoints found)")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  示例 2：加载最终模型并预测                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def load_final_model(model_dir):
    """
    加载自动选择（或强制指定）的最优模型，用于预测新数据。

    Usage:
        model, scaler_X, scaler_y, features = load_final_model('models/SVR')

        # 预测新数据
        X_new = pd.read_csv('new_data.csv')
        X_new = X_new[features]  # 只用最终选定的特征
        X_scaled = scaler_X.transform(X_new)
        y_pred_scaled = model.predict(X_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    """
    final_files = sorted(glob.glob(os.path.join(model_dir, '*_final_*.joblib')))
    if not final_files:
        raise FileNotFoundError(f"No final model found in {model_dir}")

    # 取最新的（如果有多个，按时间戳排序取最后一个）
    info = joblib.load(final_files[-1])

    model = info['model']
    scaler_X = info['scaler_X']
    scaler_y = info['scaler_y']
    features = info['features']

    print(f"\n  Loaded final model from: {os.path.basename(final_files[-1])}")
    print(f"  Features ({len(features)}): {features}")
    print(f"  Selection mode: {'forced' if info.get('force_n_features') else 'auto'}")
    print(f"  Optimal N features: {info.get('optimal_n_features', '?')}")
    for k, v in info['metrics'].items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    # 打印 SHAP-RFECV 路径（如果存在）
    path = info.get('shap_rfecv_path_summary')
    if path:
        print(f"\n  SHAP-RFECV Path:")
        print(f"  {'Feat':<5} {'RKfold MAE':<14} {'RKfold R²':<12} "
              f"{'LOOCV R²':<10} {'LOOCV MAE':<10}")
        print(f"  {'-'*54}")
        for e in path:
            print(f"  {e['n_features']:<5} {e['rkf_mae_mean']:<14.4f} "
                  f"{e['rkf_r2_mean']:<12.4f} {e['loo_r2']:<10.4f} "
                  f"{e['loo_mae']:<10.4f}")

    return model, scaler_X, scaler_y, features


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  示例 3：手动选择特定特征数的 checkpoint（覆盖自动选择）                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def load_by_feature_count(model_dir, target_n_features):
    """
    当你通过 SHAP-RFECV Path Summary 发现某个特征数（比如 5）的综合表现
    比自动选择（比如 7）更好时，用此函数加载对应特征数的 checkpoint。

    Usage:
        model, scaler_X, scaler_y, features = load_by_feature_count('models/SVR', 5)
    """
    import re

    iter_files = sorted(glob.glob(os.path.join(model_dir, '*_iteration_*.joblib')))

    # 遍历所有迭代 checkpoint，找到特征数匹配的那个
    candidates = []
    for f in iter_files:
        info = joblib.load(f)
        n_feat = len(info['features'])
        if n_feat == target_n_features:
            candidates.append((f, info))

    if not candidates:
        # 如果恰好没有这个特征数的 checkpoint，找最接近的
        print(f"  ⚠ No checkpoint with exactly {target_n_features} features.")
        by_distance = []
        for f in iter_files:
            info = joblib.load(f)
            by_distance.append((abs(len(info['features']) - target_n_features), f, info))
        by_distance.sort()
        closest_dist, closest_f, closest_info = by_distance[0]
        print(f"  Using closest: {len(closest_info['features'])} features "
              f"(file: {os.path.basename(closest_f)})")
        candidates = [(closest_f, closest_info)]

    # 如果多个 checkpoint 特征数相同（不同迭代被移除的特征不同），
    # 取 MAE 最低的那个
    best_f, best_info = min(candidates,
                            key=lambda x: x[1]['metrics'].get('mae_mean', float('inf')))

    model = best_info['model']
    scaler_X = best_info['scaler_X']
    scaler_y = best_info['scaler_y']
    features = best_info['features']

    print(f"\n  ★ Manually selected: {len(features)} features")
    print(f"  ★ Source: {os.path.basename(best_f)}")
    print(f"  ★ Features: {features}")
    m = best_info['metrics']
    print(f"  ★ Metrics: MAE={m.get('mae_mean','?'):.4f}, "
          f"R²_test={m.get('r2_test','?'):.4f}, "
          f"MAE_test={m.get('mae_test','?'):.4f}")
    print(f"\n  (Compare this with the auto-selected final model to confirm")
    print(f"   your manual choice is indeed better for your criteria.)")

    return model, scaler_X, scaler_y, features


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  示例 4：对比两个 checkpoint（自动选择 vs 手动选择）                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def compare_checkpoints(model_dir, n_features_a, n_features_b):
    """
    并排对比两个不同特征数 checkpoint 的全部指标，帮助你做出手动选择决策。

    Usage:
        compare_checkpoints('models/SVR', 7, 5)
    """
    print(f"\n{'='*80}")
    print(f"  Manual Comparison: {n_features_a} features vs {n_features_b} features")
    print(f"{'='*80}")

    models_info = {}
    iter_files = sorted(glob.glob(os.path.join(model_dir, '*_iteration_*.joblib')))

    for target_n in [n_features_a, n_features_b]:
        best_f, best_info = None, None
        best_mae = float('inf')
        for f in iter_files:
            info = joblib.load(f)
            if len(info['features']) == target_n:
                mae = info['metrics'].get('mae_mean', float('inf'))
                if mae < best_mae:
                    best_mae = mae
                    best_f, best_info = f, info
        models_info[target_n] = (best_f, best_info)

    # 打印对比表
    metric_names = ['mae_mean', 'r2_test', 'mae_test', 'rkf_mae_mean',
                    'rkf_mae_std', 'rkf_r2_mean', 'loo_r2', 'loo_mae']
    metric_labels = ['100-spl MAE', 'Test R²', 'Test MAE',
                     'RKfold MAE', 'RKfold MAE std', 'RKfold R²',
                     'LOOCV R²', 'LOOCV MAE']

    print(f"\n  {'Metric':<20} {str(n_features_a)+' features':>20} {str(n_features_b)+' features':>20} {'Better':>8}")
    print(f"  {'-'*70}")
    for mname, mlabel in zip(metric_names, metric_labels):
        val_a = models_info[n_features_a][1]['metrics'].get(mname)
        val_b = models_info[n_features_b][1]['metrics'].get(mname)
        if val_a is None or val_b is None:
            continue
        # MAE 类指标：越低越好；R² 类：越高越好
        if 'mae' in mname.lower() or 'std' in mname.lower():
            better = '<-' if val_a <= val_b else '->'
        else:
            better = '<-' if val_a >= val_b else '->'
        print(f"  {mlabel:<20} {val_a:>20.4f} {val_b:>20.4f} {better:>8}")

    # 打印特征列表
    print(f"\n  Features ({n_features_a}): {models_info[n_features_a][1]['features']}")
    print(f"  Features ({n_features_b}): {models_info[n_features_b][1]['features']}")
    # 差异
    set_a = set(models_info[n_features_a][1]['features'])
    set_b = set(models_info[n_features_b][1]['features'])
    print(f"  Only in {n_features_a}: {set_a - set_b}")
    print(f"  Only in {n_features_b}: {set_b - set_a}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  示例 5：用选定模型预测新数据（完整流程）                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def predict_with_checkpoint(model_dir, csv_path, n_features=None):
    """
    用指定特征数的 checkpoint 对新数据进行预测。

    Usage:
        # 用自动选择的最终模型预测
        predict_with_checkpoint('models/SVR', 'new_data.csv')

        # 用 5 特征的 checkpoint 预测
        predict_with_checkpoint('models/SVR', 'new_data.csv', n_features=5)
    """
    # 1. 加载模型
    if n_features is not None:
        model, scaler_X, scaler_y, features = load_by_feature_count(model_dir, n_features)
    else:
        model, scaler_X, scaler_y, features = load_final_model(model_dir)

    # 2. 加载新数据
    data = pd.read_csv(csv_path)
    print(f"\n  New data: {len(data)} samples")

    # 3. 检查特征列是否存在
    missing = set(features) - set(data.columns)
    if missing:
        print(f"  ⚠ Missing columns in new data: {missing}")
        print(f"  Cannot proceed without these features.")
        return None

    # 4. 预测
    X_new = data[features]
    X_scaled = scaler_X.transform(X_new)
    y_pred_scaled = model.predict(X_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    # 5. 附加预测结果到原数据
    data['predicted_activation_energy'] = y_pred

    # 6. 保存
    out_path = csv_path.replace('.csv', f'_predicted_{len(features)}feat.csv')
    data.to_csv(out_path, index=False)
    print(f"  Predictions saved to: {out_path}")
    print(f"  Predicted range: {y_pred.min():.2f} ~ {y_pred.max():.2f} kcal/mol")

    return data


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  交互式使用示例（取消注释即可运行）                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

if __name__ == '__main__':
    # ── 示例：查看 SVR 的所有 checkpoint ──
    list_checkpoints('models/SVR')

    # ── 示例：加载自动选择的最终模型 ──
    # model, sX, sY, feats = load_final_model('models/SVR')

    # ── 示例：手动选择 5 特征的 checkpoint ──
    # model, sX, sY, feats = load_by_feature_count('models/SVR', 5)

    # ── 示例：并排对比 7 特征 vs 5 特征 ──
    # compare_checkpoints('models/SVR', 7, 5)

    # ── 示例：用选定模型预测新数据 ──
    # predict_with_checkpoint('models/SVR', 'new_data.csv', n_features=5)
    pass
