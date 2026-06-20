# All fonts use 'DejaVu Sans' (bundled with matplotlib) for cross-platform compatibility.
# 'Arial' and 'Times New Roman' are not installed by default on Linux.
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def plot_scatter(y_train, y_pred_train, y_test, y_pred_test, model_name, mae_mean, output_dir, output_name,
               X_train=None, X_test=None, r2_loo=None, rkf_mae=None, rkf_r2=None,
               precomputed_metrics=None):
    """
    Plot actual vs predicted scatter plot and export outlier data.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.clf()
    plt.figure(figsize=(8, 6))
    plt.subplot(111)
    plt.gca().set_aspect('equal', adjustable='box')

    train_diff = np.abs(y_train - y_pred_train)
    test_diff = np.abs(y_test - y_pred_test)

    train_normal = train_diff < 5.0
    train_outlier = train_diff >= 5.0
    test_normal = test_diff < 5.0
    test_outlier = test_diff >= 5.0

    plt.scatter(y_train[train_normal], y_pred_train[train_normal], color='blue', label='Development')
    plt.scatter(y_test[test_normal], y_pred_test[test_normal], color='green', label='Final Test')

    outlier_actual = np.concatenate([y_train[train_outlier], y_test[test_outlier]])
    outlier_pred = np.concatenate([y_pred_train[train_outlier], y_pred_test[test_outlier]])
    if len(outlier_actual) > 0:
        plt.scatter(outlier_actual, outlier_pred, color='red', label='Outliers')
        outliers_data = []

        if np.any(train_outlier):
            train_outliers = {
                'Set': ['Development'] * sum(train_outlier),
                'Actual': y_train[train_outlier],
                'Predicted': y_pred_train[train_outlier],
                'Difference': train_diff[train_outlier]
            }
            if X_train is not None:
                for col in X_train.columns:
                    train_outliers[col] = X_train.loc[y_train[train_outlier].index, col].values
            outliers_data.append(pd.DataFrame(train_outliers))

        if np.any(test_outlier):
            test_outliers = {
                'Set': ['Final Test'] * sum(test_outlier),
                'Actual': y_test[test_outlier],
                'Predicted': y_pred_test[test_outlier],
                'Difference': test_diff[test_outlier]
            }
            if X_test is not None:
                for col in X_test.columns:
                    test_outliers[col] = X_test.loc[y_test[test_outlier].index, col].values
            outliers_data.append(pd.DataFrame(test_outliers))

        if outliers_data:
            outliers_df = pd.concat(outliers_data, axis=0)
            outliers_df = outliers_df.sort_values('Difference', ascending=False)
            outliers_csv_path = output_dir + output_name.replace('.png', '_outliers.csv')
            outliers_df.to_csv(outliers_csv_path, index=True)
            print(f"\nOutlier Statistics:")
            print(f"Total outliers: {len(outliers_df)}")
            print(f"Development set outliers: {sum(train_outlier)}")
            print(f"Final test set outliers: {sum(test_outlier)}")
            print(f"Outlier data saved to: {outliers_csv_path}\n")

    plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)],
             color='red', label='Perfect Prediction')

    metrics = (
        precomputed_metrics
        if precomputed_metrics is not None
        else calculate_metrics(y_train, y_pred_train, y_test, y_pred_test)
    )

    if mae_mean is not None:
        add_plot_labels(metrics, mae_mean, model_name, r2_loo, rkf_mae=rkf_mae, rkf_r2=rkf_r2)

    plt.tight_layout()
    plt.savefig(output_dir + output_name, dpi=300, format='png', bbox_inches='tight')
    plt.close()


def _pearson_r(y_true, y_pred):
    """Safe Pearson correlation coefficient."""
    if len(y_true) > 1:
        return np.corrcoef(y_true, y_pred)[0, 1]
    return 0.0


def calculate_metrics(y_train, y_pred_train, y_test, y_pred_test):
    """Compute model evaluation metrics with true Pearson r."""
    return {
        'r_train': _pearson_r(y_train, y_pred_train),
        'r_test': _pearson_r(y_test, y_pred_test),
        'r2_train': r2_score(y_train, y_pred_train),
        'rmse_test': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'r2_test': r2_score(y_test, y_pred_test),
        'mae_test': mean_absolute_error(y_test, y_pred_test)
    }


def add_plot_labels(metrics, mae_mean, model_name, r2_loo, rkf_mae=None, rkf_r2=None):
    """Add compact metric labels to the scatter plot."""
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Actual vs Predicted Values for {model_name}')
    plt.legend(loc='upper left')

    ax = plt.gca()
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    text_x = xmax + (xmax - xmin) * 0.05
    y_range = ymax - ymin
    y_spacing = y_range * 0.1

    metrics_text = [
        f"Development Pearson R: {metrics['r_train']:.4f}",
        f"Final Test Pearson R: {metrics['r_test']:.4f}",
        f"Final Test RMSE: {metrics['rmse_test']:.4f}",
        f"Final Test MAE: {metrics['mae_test']:.4f}",
        f"100-split MAE: {mae_mean:.4f}",
    ]
    if rkf_mae is not None:
        metrics_text.append(f"5×5 RKFold MAE: {rkf_mae:.4f}")
    if rkf_r2 is not None:
        metrics_text.append(f"5×5 RKFold R²: {rkf_r2:.4f}")
    if r2_loo is not None:
        metrics_text.append(f"LOOCV R²: {r2_loo:.4f}")

    for i, text in enumerate(metrics_text):
        y_pos = ymin + i * y_spacing + y_range * 0.1
        plt.text(text_x, y_pos, text, fontsize=12, ha='left', va='top', fontname='DejaVu Sans')

    plt.subplots_adjust(right=0.85)


def add_plot_labels_standard(metrics, mae_mean, model_name, r2_loo, fontsize=20, fontname='DejaVu Sans', rkf_mae=None, rkf_r2=None):
    """Add compact publication labels."""
    labels = [
        f"Development Pearson R: {metrics['r_train']:.4f}",
        f"Final Test Pearson R: {metrics['r_test']:.4f}",
        f"Final Test RMSE: {metrics['rmse_test']:.4f}",
        f"Final Test MAE: {metrics['mae_test']:.4f}",
        f"100-split MAE: {mae_mean:.4f}",
    ]
    if r2_loo is not None:
        labels.append(f"LOOCV R²: {r2_loo:.4f}")
    if rkf_mae is not None:
        labels.append(f"5×5 RKFold MAE: {rkf_mae:.4f}")
    if rkf_r2 is not None:
        labels.append(f"5×5 RKFold R²: {rkf_r2:.4f}")

    for row, label in enumerate(labels):
        plt.text(
            0.95,
            0.05 + row * 0.07,
            label,
            fontsize=fontsize - 5,
            ha='right',
            va='bottom',
            fontname=fontname,
            transform=plt.gca().transAxes,
        )


def plot_scatter_standard(y_train, y_pred_train, y_test, y_pred_test, model_name, mae_mean, output_dir, output_name,
               X_train=None, X_test=None, r2_loo=None, fontsize=20, fontname='DejaVu Sans'):
    """Plot actual vs predicted scatter.
    """
    fontsize = 20
    fontname = 'Times New Roman'
    plt.rcParams['font.family'] = fontname
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.clf()
    plt.figure(figsize=(8, 6))
    plt.subplot(111)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.scatter(y_train, y_pred_train, color='blue', label='Development')
    plt.scatter(y_test, y_pred_test, color='green', label='Final Test')
    plt.xlabel('True Values / kcal/mol', fontsize=fontsize, fontname=fontname)
    plt.ylabel('Predicted Values / kcal/mol', fontsize=fontsize, fontname=fontname)
    plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)],
             color='red', label='Perfect Prediction')
    plt.legend(loc='upper left', fontsize=fontsize - 6)

    plt.xticks(fontsize=fontsize - 3, fontname=fontname)
    plt.yticks(fontsize=fontsize - 3, fontname=fontname)

    metrics = calculate_metrics(y_train, y_pred_train, y_test, y_pred_test)

    if mae_mean is not None:
        add_plot_labels_standard(metrics, mae_mean, model_name, r2_loo, fontsize, fontname)

    plt.tight_layout()
    plt.savefig(output_dir + output_name, dpi=300, format='png', bbox_inches='tight')
    plt.close()
