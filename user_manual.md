# User Manual: Following the ML Pipeline

This manual provides step-by-step instructions for executing the machine learning pipeline described in [Pipeline.md](Pipeline.md) and [Pipeline.txt](Pipeline.txt). The pipeline predicts **activation energy (kcal/mol)** of organoboronate reactions.

For a conceptual overview of the workflow, read [Pipeline.md](Pipeline.md) first.

---

## Pipeline-to-Code Mapping

Below, each pipeline stage is mapped to the exact code, commands, and steps required.

---

### Stage 1: Data Preparation (Data Preprocessing & Split)

**What the pipeline says:**
- Clean data: drop all-NaN columns, filter numeric features
- Split data into Training Set / Test Set (80/20)

**How to execute:**

Data preparation happens in `main.py` (lines 76-81):

```python
data = pd.read_csv('example/B_dataset.csv')
data = data.dropna(axis=1, how='all')   # Drop columns that are entirely NaN
features = data.select_dtypes(include=[np.number]).columns  # Select numeric columns
X = data[features].drop('activation_energy', axis=1)  # Feature matrix
y = data['activation_energy']                           # Target variable
```

The 80/20 split with `random_state=42` is performed inside `src/train_and_evaluate.py`:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

**Input CSV must contain:**
- Non-numeric identifier columns (`sub_H`, `sub_B`) — automatically excluded
- Numeric physicochemical property columns (e.g., `pka_H`, `dipole_H`, `homo_H`, `lumo_H`, `delta_G_B`, `NICS1`, etc.)
- The target column: **`activation_energy`** (kcal/mol)

**To start:** place your CSV at `example/B_dataset.csv` (or edit the path in `main.py` line 76).

---

### Stage 2: Model Training & Hyperparameter Optimization (LOOCV + Bayesian Opt.)

**What the pipeline says:**
- Train SVR, RandomForest, and KNR models
- Use Leave-One-Out Cross Validation (LOOCV) inside Optuna's objective function
- Use Bayesian Optimization (TPESampler) to tune hyperparameters
- Train initial model with best hyperparameters

**How to execute:**

Fully implemented in `src/train_and_evaluate.py`. The `objective()` function performs LOOCV on the training split for each Optuna trial. Optuna uses `TPESampler` with `n_startup_trials=5`, all random seeds fixed to 42.

**To run this for all active models:**
```bash
python main.py --n_trials 100 --mae_threshold 2.0 --min_features 5
```

**To run for a single model in a script/notebook:**
```python
from src.train_and_evaluate import train_and_evaluate
from sklearn.svm import SVR
import pandas as pd

data = pd.read_csv('example/B_dataset.csv').dropna(axis=1, how='all')
features = data.select_dtypes(include=['number']).columns
X = data[features].drop('activation_energy', axis=1)
y = data['activation_energy']

mae_loo, mae_mean, best_params = train_and_evaluate(
    SVR, X, y, random_state=42, n_trials=100, n_jobs=-1
)
# Prints: best hyperparameters, train/test R², RMSE, MAE
# Automatically evaluates on 100 random splits
```

Optuna studies are persisted as `optuna_optimization_{ModelClass}.db` SQLite files for resuming interrupted runs.

---

### Stage 3: Model Evaluation 1 (Test Set Assessment + 100-split stability)

**What the pipeline says:**
- Evaluate initial model on test set; check for overfitting
- Run 100 random train/test splits to assess stability; compute average MAE

**How to execute:**

Done automatically inside `train_and_evaluate()` (lines 277-333). After Optuna optimization:

```
random_state = 42
Best hyperparameters: {'C': 63.91, 'epsilon': 0.017, 'gamma': 5.27}
Train R²: 0.9234         ← compare these two
Test R²:  0.8745         ← to check overfitting
Test RMSE: 3.25 kcal/mol
Test MAE:  2.13 kcal/mol
Average MAE on 100 times random test sets: 2.13
Best random state: 17 with best MAE: 1.45
```

If Train R² significantly exceeds Test R², overfitting is present.

---

### Stage 4: Iterative Feature Selection (SHAP + Correlation + Filtering)

**What the pipeline says:**
- SHAP analysis: per-feature importance percentages
- Correlation analysis: identify redundant pairs (|r| > 0.8)
- Feature selection: remove least important in high-correlation pairs, then low-importance features (<20%)
- Iterative filtering: retrain after removal, repeat until stopping condition

**How to execute:**

This is the core loop in `src/iterative_optimization.py`. Each iteration:
1. Re-trains the model with current features (calls Stage 2)
2. Runs `feature_importance_analysis()` — automatically selects the right SHAP explainer
3. Runs `feature_correlation_analysis()` — Pearson correlation matrix
4. Runs `feature_selection()` — two-phase selection (high-correlation → low-importance)
5. Removes selected feature(s) and loops back

```bash
python main.py --n_trials 100 --mae_threshold 2.0 --min_features 5
```

Console/log output per iteration:
```
=== Feature Importance Ranking (Iteration 1) ===
  [1] delta_G_B:         35.21%
  [2] lumo_B:            28.14%
  [3] NICS1:             15.33%
  ...
Removing feature: dipole_H
Feature importance: 4.21%
Correlation with other features:
  dipole_H - electronegativity_H: 0.85
```

Feature importance < 20% and correlation > 0.8 thresholds are hardcoded in `src/feature_selection.py`.

**Stopping conditions:**
- `len(features) <= min_features` (default: 5)
- No high-correlation pairs AND no features with importance < 20%

**Exhaustive combinatorial search** (alternative, slower):
```bash
# Edit example/model_feature_filter.py first:
#   - set data path, max_features, uncomment desired models
python example/model_feature_filter.py
```

---

### Stage 5: Final Build & Rigorous Testing

**What the pipeline says:**
- Build final model with selected features only
- LOO-CV validation on final feature set
- Performance history tracking
- Generate scatter plots and outlier analysis

**How to execute:**

Done automatically at the end of the iterative loop in `iterative_optimization.py`:

1. LOO-CV on the final feature set → R²_LOO
2. Final model trained with all remaining features
3. Performance history saved as CSV + PNG
4. Final scatter plot (actual vs. predicted, with outlier detection at deviation ≥ 5.0 kcal/mol)
5. Model saved as `models/<ModelName>/<ModelName>_final_<timestamp>.joblib`

**Final model `.joblib` contents:**
```python
{
    'model': best_model,           # Trained sklearn model
    'scaler_X': MinMaxScaler,      # Feature scaler
    'scaler_y': MinMaxScaler,      # Target scaler (0, 100)
    'features': [...],             # Final feature names
    'hyperparameters': {...},      # Best hyperparameters
    'metrics': {
        'mae_mean': ...,           # Mean MAE from Optuna
        'r2_test': ...,            # Test set R²
        'mae_test': ...,           # Test set MAE
        'r2_loo': ...              # LOO cross-validation R²
    },
    'removed_features': [...]      # Features removed in order
}
```

**To load and use a saved model:**
```python
import joblib
model_info = joblib.load('models/SVR/SVR_final_20260412_133724.joblib')
model = model_info['model']
features = model_info['features']
# Predict: model.predict(scaler_X.transform(X[features]))
```

**To run standalone 100-split evaluation on an existing model**, use `example/example_pic.ipynb`:
```python
model_info = joblib.load('models/SVR/SVR_final_20260412_133724.joblib')
# Use the plot_r2_on_100_random_samples function from the notebook
```

---

### Stage 6: External Validation & Ensemble Prediction

**What the pipeline says:**
- Test on completely independent, external data
- Generate full combinatorial sub_H × sub_B validation space
- Ensemble prediction across 100 random-split models
- Weighted prediction (by inverse MSE) for final output

**How to execute:**

This stage is **notebook-based** (not in the `main.py` CLI).

#### Step 6a: Generate validation dataset

```python
from src.validation_process import validation_data_produce
import pandas as pd

data = pd.read_csv('your_data.csv')
H_feature_cols = ['pka_H', 'dipole_H', 'homo_H', 'lumo_H', ...]
B_feature_cols = ['pka_B', 'delta_G_B', 'delta_G_B_TS', 'dipole_B', ...]

validation_data = validation_data_produce(data, H_feature_cols, B_feature_cols)
validation_data.to_csv('validation_data.csv', index=False)
```

#### Step 6b: Run prediction notebook

```bash
jupyter notebook example/prediction_round2.ipynb
```

The notebook provides three prediction strategies:

| Cell | Strategy | Description |
|---|---|---|
| 1-2 | **Weighted average** | 100 models, weighted by `1/MSE`. Output: `final_with_y_pred_weighted_mean.csv` |
| 4 | **Simple average** | 100 models, arithmetic mean. Output: `final_with_y_pred_mean.csv` |
| 5 | **Single model** | Uses a single `random_state=10` split. Output: `target_with_y_pred_mean.csv` |

**Before running the notebook:**
1. Place your training data CSV in the working directory
2. Generate `validation_data.csv` using `validation_process.py`
3. Edit the model hyperparameters in each cell to match your best model
4. Edit the feature column list to match your selected features
5. Edit the target column name (the notebook defaults to `yield`)

---

## Complete Execution Checklist

| Stage | Action | Output |
|---|---|---|
| **1. Data Prep** | Place `example/B_dataset.csv` in the repo | — |
| **2. Initial Training** | `python main.py --n_trials 100 --mae_threshold 2.0 --min_features 5` | `models/*/`, Optuna DBs, logs |
| **3. Evaluation 1** | Review console output: compare Train R² vs Test R², check 100-split MAE | Overfitting assessment |
| **4. Feature Selection** | Review `models/optimization_*.log` for per-iteration feature ranking | Feature importance ranking, removed features list |
| **4. Exhaustive Filtering** | Edit & run `example/model_feature_filter.py` (optional) | Optimal feature subset |
| **5. Final Build** | `models/<Model>/<Model>_final_*.joblib` is the final model | `.joblib` file with all artifacts |
| **5. Visualizations** | Open `models/<Model>/final_scatter_*.png` and `performance_history_*.png` | Scatter plot, performance traces |
| **6. Validation Space** | Run `validation_process.py` | `validation_data.csv` |
| **6. External Valid.** | Run `example/prediction_round2.ipynb` | Prediction CSV, evaluation scatter plot |

---

## Runtime Estimates

| Stage | Typical Runtime |
|---|---|
| Data preparation | Seconds |
| `main.py --n_trials 100` (3 models) | Several days on a personal computer |
| `main.py --n_trials 20` (quick test) | A few hours |
| `model_feature_filter.py` with many combinations | Days |
| Notebook evaluation/visualization | Minutes to hours |
| External validation prediction | Minutes |
