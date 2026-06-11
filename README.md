# Machine Learning Tool for Organoboronate Reactivity Prediction

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)]() [![License](https://img.shields.io/badge/License-MIT-green)]() [![DOI](https://img.shields.io/badge/DOI-10.1038%2Fs41467--025--60674--9-blue)](https://doi.org/10.1038/s41467-025-60674-9)

[English](README.md) | [中文](README_CN.md)

This project is a supplementary material for the machine learning component of our research paper "[**Organometallic-type reactivity of stable organoboronates for selective (hetero)arene C−H/C-halogen borylation and beyond**](https://doi.org/10.1038/s41467-025-60674-9)". It contains machine learning models and analysis tools for predicting the activation energies of organoboronate reactions.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation Guide](#installation-guide)
3. [Workflow Overview](#workflow)
4. [Detailed Usage Guide](#detailed-usage-guide)
   - [Data Preparation](#data-preparation)
   - [Model Training and Feature Selection](#model-training-and-feature-selection)
   - [Further Feature Filtering](#further-feature-filtering)
   - [Model Evaluation and Visualization](#model-evaluation-and-visualization)
   - [External Validation and Prediction](#external-validation-and-prediction)
5. [Expected Runtime](#expected-runtime)
6. [Reproduction Instructions](#reproduction-instructions)
7. [License](#license)

## System Requirements

### Software Dependencies and Operating Systems

Main dependencies:

- Python >= 3.9
- pandas >= 2.0
- numpy >= 1.20
- scikit-learn >= 1.0
- matplotlib >= 3.5
- seaborn >= 0.12
- xgboost >= 1.5
- lightgbm >= 3.3
- optuna >= 3.0
- shap >= 0.40
- joblib >= 1.1

For a complete list of dependencies, see the [requirements.txt](requirements.txt) file.

### Tested Operating Systems

- Linux: Arch Linux
- Windows: Windows 10, Windows 11

### Hardware Requirements

- CPU: Any modern multi-core processor (4+ cores recommended)
- RAM: Minimum 4GB, 8GB+ recommended
- Storage: At least 500MB free space
- GPU: Not required, but if a CUDA-compatible GPU is installed, some models (like XGBoost) may be accelerated

## Installation Guide

### Installation Steps

1. Clone or download this repository:

```bash
git clone https://github.com/Daojing-Li/Stable-Organoboronates-ML-Supp
cd Stable-Organoboronates-ML-Supp
```

2. Set up the environment and install dependencies:

#### Option 1: Using uv (Recommended - Much Faster)

uv is a high-performance Python package manager written in Rust that can significantly speed up dependency installation.

**Typical install time: ~30 seconds** on a standard desktop computer

```bash
# Install uv if you don't have it
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Single command to create environment and install all dependencies
uv sync
```

This single command automatically:
- Creates a virtual environment at `.venv`
- Installs all dependencies from requirements.txt
- Optimizes installation with parallel downloads and caching

To activate the virtual environment:
```bash
# On Linux/macOS
source .venv/bin/activate

# On Windows
.venv\Scripts\activate
```

#### Option 2: Using standard tools

**Typical install time: 3-5 minutes** on a standard desktop computer

```bash
# Create virtual environment
python -m venv .venv

# Activate the environment
# On Linux/macOS
source .venv/bin/activate
# On Windows
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Workflow

This project provides a complete workflow for building and evaluating machine learning models for organoboronate reactivity prediction:

### 1. Data Preparation

Input data should be in CSV format, containing physicochemical properties and target activation energy values. See the `example/B_dataset.csv` file for an example of the data format.

### 2. Model Selection and Feature Optimization

Use the main script `main.py` for model training, hyperparameter optimization, and iterative feature selection:

```bash
python main.py --n_trials 100 --mae_threshold 2.0 --min_features 5
```

This step will:
- Train multiple machine learning models (SVR, Random Forest, K-Neighbors Regressor, etc.)
- Perform hyperparameter optimization via Optuna with Leave-One-Out Cross-Validation (LOO-CV)
- Evaluate models with a **dual cross-validation system**: 5×5 RepeatedKFold as primary metric (lower variance, better stability assessment) and LOOCV as auxiliary reference
- Conduct iterative feature selection using a **hybrid strategy**: original SHAP + correlation method when features ≥ 10, switching to SHAP-RFECV for refined selection when features < 10
- Execute **y-randomization test** on the final best model to verify that predictions are based on genuine structure-activity relationships rather than chance correlations
- Save the best models and all performance metrics (including RepeatedKFold and LOOCV results)

### 3. Further Feature Filtering

The built-in **SHAP-RFECV** pipeline (Stages 4-5) already performs automatic feature selection with multi-fold CV consensus SHAP and 1-SE rule optimization — no additional scripts are needed.

### 4. Model Evaluation and Visualization

Use `example/example_pic.ipynb` to generate comprehensive model evaluation and visualization results:

- Scatter plots comparing actual vs. predicted values
- Leave-one-out cross-validation (LOO)
- Model stability assessment across 100 random splits
- Feature importance analysis

### 5. External Validation and New Data Prediction

Use `example/prediction_round2.ipynb` to predict and externally validate on new data:

- Load the trained best model
- Preprocess new data
- Generate prediction results
- Evaluate prediction performance (if actual values are available for comparison)

## Detailed Usage Guide

### Data Preparation

#### Data Format Requirements

Input data should be in CSV format, containing the following columns:

1. Compound identifiers (e.g., `sub_H`, `sub_B`)
2. Various physicochemical properties, including but not limited to:
   - `pka_H`: pKa value
   - `dipole_H`: Dipole moment
   - `homo_H`: Highest Occupied Molecular Orbital energy
   - `lumo_H`: Lowest Unoccupied Molecular Orbital energy
   - `delta_G_B`: Gibbs free energy of boron reagent
   - `NICS1`: Nucleus Independent Chemical Shift
   - Other physicochemical properties
3. Target variable `activation_energy` (in kcal/mol)

### Model Training and Feature Selection

#### Using the main.py Script

`main.py` is the main entry point of the project, used for training multiple regression models, optimizing hyperparameters, and performing iterative feature selection:

```bash
python main.py --n_trials 100 --mae_threshold 2.0 --min_features 5
```

#### Parameter Explanation

- `--n_trials`: Number of trials for Optuna optimization (default 100)
  - Higher values may lead to better model performance but require longer training time
  - Recommended range: 50-200

- `--mae_threshold`: MAE threshold for filtering good models (default 2.0, unit: kcal/mol)
  - Models with MAE below this threshold are considered "good" — their evaluation plots are displayed
  - Recommended range: 1.0-5.0, depending on your data and expected accuracy

- `--min_features`: Minimum number of features to retain (default 5)
  - The iterative feature removal stops when this many features remain
  - Recommended range: 5-10

- `--n_jobs`: Number of CPU cores to use (default -1, meaning all cores)
  - Can speed up the training process, especially with tree-based models
  - In resource-constrained environments, you can set this to a smaller value

- `--keep_versions`: Number of versions to keep for each model (default 2)
  - Controls how many best model versions to save for each model type
  - Higher values will use more disk space

#### Training Process

1. The script will automatically start training multiple models, including SVR, Random Forest, K-Neighbors Regressor, etc.
2. For each model, Optuna performs Bayesian hyperparameter optimization with LOOCV on the training set. **Ridge** uses RidgeCV for automatic alpha selection (skipping Optuna entirely); **Lasso** uses LassoCV for alpha selection while Optuna tunes the tolerance parameter.
3. **SVR** uses a narrowed epsilon search range (0.001–0.5) for better fitting precision on activation energy prediction.
4. The script iteratively removes low-importance or high-correlation features and retrains until reaching `min_features` or no more features qualify for removal. When the number of features drops below 10, the feature selection method switches from fixed-threshold SHAP+correlation to **SHAP-RFECV** for data-driven feature elimination.
5. After the best model is obtained, a **y-randomization test** (100 permutations, 5×5 RepeatedKFold evaluation) automatically verifies the model's predictive validity.
6. Training process logs will be output to the console and saved in the `models` directory.

#### Evaluation System

The project uses a **dual cross-validation** framework:

| Layer | Method | Role | Output |
|---|---|---|---|
| **Primary** | 5×5 RepeatedKFold | Model comparison, feature selection, hyperparameter tuning | MAE ± std, R² ± std |
| **Auxiliary** | LOOCV | Robustness reference, literature comparison | R², MAE |

All model ranking and feature selection decisions are based on **5×5 RepeatedKFold** results. LOOCV values are provided as supplementary reference only.

#### Training Results

After training is complete, the best models and related results will be saved in the `models` directory, with a subdirectory for each model type:

```
models/
├── SVR/
│   ├── SVR_final_20260412133724.joblib       # Saved final model (includes scalers, features, hyperparameters, all metrics)
│   ├── SVR_final_20260412133724_metrics.txt   # Final model performance metrics
│   ├── SVR_iteration_1_20260412133241.joblib  # Per-iteration checkpoints
│   ├── final_scatter_20260412133724.png       # Actual vs predicted scatter plot
│   ├── performance_history_20260412133724.csv # Iteration-by-iteration metrics
│   └── ...
├── RandomForest/
│   └── ...
└── KNR/
    └── ...
```

### Further Feature Filtering

If the number of features after initial filtering is still high, you can use the `example/model_feature_filter.py` script to further reduce the number of features:

#### Using model_feature_filter.py

1. Open the script and modify the following parameters:
   - Data file path
   - Target maximum number of features (`max_features` parameter)
   - Model types (uncomment or comment the corresponding models)

2. Run the script:

```bash
python example/model_feature_filter.py
```

3. The script will output the optimal feature set after filtering and the corresponding model performance.

#### Feature Filtering Principle

This script uses the Recursive Feature Elimination with Cross-Validation (RFECV) method, filtering features through the following steps:

1. Initialize the model and set hyperparameters
2. Train the initial model and evaluate performance
3. Recursively remove the least important features and re-evaluate performance
4. Find the smallest feature subset that provides the best performance

### Model Evaluation and Visualization

Use the `example/example_pic.ipynb` Jupyter notebook for detailed model evaluation and visualization:

The notebook contains the following main functions:
- Load the trained best model
- Generate scatter plots comparing actual vs. predicted values
- Perform Leave-One-Out (LOO) cross-validation to assess model robustness
- Evaluate model stability across 100 random data splits
- Analyze feature importance

### External Validation and Prediction

Use `example/prediction_round2.ipynb` for external validation and prediction on new data:

The notebook contains the following main functions:
- Load the trained best model
- Load new validation datasets
- Preprocess data to fit the model input
- Generate prediction results
- Evaluate prediction performance (if actual values are available for comparison)
- Visualize prediction results

#### Predicting New Data

To predict completely new data, you need to:

1. Prepare a new data file containing the same features
2. Modify the data loading section in the notebook to point to your new data
3. Run the prediction part of the notebook
4. Analyze and export the prediction results

## Expected Runtime

The runtime for the different steps of the workflow can vary significantly depending on your hardware. Below are typical runtime estimates on a standard desktop computer:

1. **Data Preparation**: Negligible (seconds)

2. **Model Training and Iterative Feature Selection** (Step 2): 
   - **Several days** on a personal computer for complete model training with the recommended parameters (3 models × 100 trials each, with iterative feature removal)
   - For testing purposes, you can reduce `n_trials` to 20, which will complete in a few hours

3. **Further Feature Filtering** (Step 3):
   - Runtime depends heavily on the `max_features` parameter and the total number of features
   - Exhaustive search over many feature combinations can take **days** on a personal computer
   - For testing purposes, setting `max_features=3-5` will significantly reduce runtime

4. **Model Evaluation and Visualization**: Minutes to hours depending on the complexity and number of models

5. **External Validation and Prediction**: Minutes

## Reproduction Instructions

To reproduce the exact results from our paper, follow these steps:

1. Complete the installation as described in the [Installation Guide](#installation-guide)

2. Run the initial model training with the recommended parameters:
   ```bash
   python main.py --n_trials 100 --mae_threshold 2.0 --min_features 5
   ```

3. The built-in SHAP-RFECV pipeline automatically determines the optimal feature subset — no additional filtering scripts are needed.

4. Generate evaluation visualizations using the notebooks in the `example` directory

5. For external validation, use the `example/prediction_round2.ipynb` notebook

These steps will reproduce the results presented in our paper. For quick testing or exploration, you can reduce computational time by:
- Using fewer trials in Step 2 (e.g., `--n_trials 20`)
- Setting a smaller `max_features` value (e.g., 3-5) in Step 3

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details. 