# Model Training Analysis Report — 2026-06-10 Run

> **Log:** `models/optimization_20260610_121717.log` (2.5 MB, 16 models)
> **Date:** 2026-06-10 12:17 – 14:32
> **Judgment basis:** 5×5 RepeatedKFold MAE and R² are PRIMARY. LOOCV is auxiliary reference only.

---

## Executive Summary

All 16 models completed successfully. **The Fractional 1-SE auto-selection rule was wrong for 12 out of 16 models** — it systematically under-selects features by being too aggressive about parsimony. Below I re-determine each model's optimal feature count based on the actual RKfold MAE and R² values along the full SHAP-RFECV path.

**Best model overall: SVR at 7 features (RKfold MAE=2.508, R²=0.767).**

---

## Model-by-Model Independent Analysis

### SVR — Auto: 4 feat → **Corrected: 7 features**

```
Feat  RKfold MAE ± std      RKfold R² ± std       △MAE vs best   △R² vs best
10    2.614 ± 0.375          0.764 ± 0.090          +0.106         -0.012
 9    2.579 ± 0.358          0.767 ± 0.089          +0.071         -0.009
 8    2.509 ± 0.352          0.776 ± 0.092  ←BEST R²  +0.001       ±0.000
 7    2.508 ± 0.340  ←BEST MAE  0.767 ± 0.099         ±0.000       -0.009
 6    2.548 ± 0.363          0.757 ± 0.106          +0.040         -0.019
 5    2.653 ± 0.314          0.736 ± 0.111  ←CRASH  +0.145         -0.040
 4    2.528 ± 0.336          0.760 ± 0.108  ←1-SE   +0.020         -0.016
 3    2.622 ± 0.377          0.748 ± 0.093          +0.114         -0.028
```

**Verdict: 7 features.** Best MAE (2.508). R²=0.767 is only 0.009 below the best R² at 8 feat (0.776). The 1-SE rule selected 4 because the "bad" 5-feature step fell outside the threshold, creating a misleading gap. The 4-feature model (MAE=2.528) is 0.020 worse. 8 features is also a strong choice (best R²=0.776 with MAE=2.509).

---

### Ridge — Auto: 3 feat → **Corrected: 7 features**

```
Feat  RKfold MAE ± std      RKfold R² ± std
10    2.678 ± 0.352          0.757 ± 0.103
 9    2.660 ± 0.350          0.760 ± 0.099
 8    2.624 ± 0.364          0.764 ± 0.102
 7    2.598 ± 0.350  ←BEST   0.767 ± 0.101  ←BEST
 6    2.622 ± 0.336          0.761 ± 0.100
 5    2.610 ± 0.347          0.764 ± 0.099
 4    2.627 ± 0.352          0.766 ± 0.095
 3    2.625 ± 0.414          0.761 ± 0.090  ←1-SE
```

**Verdict: 7 features.** Unambiguous peak — best MAE (2.598) and best R² (0.767) at the same point. The feature removal path is monotonic: performance steadily improves from 10→7, then degrades. The 1-SE selection of 3 features throws away 0.027 MAE and 0.006 R².

---

### LinearRegression — Auto: 3 feat → **Corrected: 7 features**

```
Feat  RKfold MAE ± std      RKfold R² ± std
10    2.681 ± 0.375          0.748 ± 0.128
 9    2.651 ± 0.368          0.755 ± 0.119
 8    2.630 ± 0.369          0.758 ± 0.119
 7    2.599 ± 0.360  ←BEST   0.766 ± 0.105  ←BEST
 6    2.620 ± 0.342          0.760 ± 0.103
 5    2.609 ± 0.351          0.764 ± 0.101
 4    2.627 ± 0.352          0.766 ± 0.095  (R² tied with best)
 3    2.625 ± 0.414          0.761 ± 0.090  ←1-SE
```

**Verdict: 7 features.** Best MAE (2.599) and best R² (0.766). Identical pattern to Ridge. The 4-feature model ties on R² but has worse MAE (2.627). The 1-SE selected 3 features which is worse on both metrics.

---

### Lasso — Auto: 3 feat → **Corrected: 5 features**

```
Feat  RKfold MAE ± std      RKfold R² ± std
10    2.749 ± 0.354          0.753 ± 0.086
 9    2.746 ± 0.356          0.753 ± 0.087  (Lasso zeroed feature — same as 10)
 8    2.744 ± 0.357          0.753 ± 0.087  (Lasso zeroed feature — same as 10)
 7    2.679 ± 0.359          0.756 ± 0.096
 6    2.628 ± 0.357          0.762 ± 0.096
 5    2.610 ± 0.349  ←BEST MAE  0.764 ± 0.099
 4    2.631 ± 0.350          0.767 ± 0.092  ←BEST R²
 3    2.647 ± 0.407          0.761 ± 0.087  ←1-SE
```

**Verdict: 5 features.** Best MAE (2.610). R²=0.764 is only 0.003 below the best R² at 4 feat. The difference between 4 and 5 features is negligible (△MAE=0.021, △R²=0.003). I choose 5 because MAE is the primary metric, but 4 features (pka, C_s, NPA_charge_B, +1) is equally defensible.

Lasso's behavior is noteworthy: features 10→8 were zeroed out by L1 regularization rather than removed, showing Lasso's built-in feature selection.

---

### KRR — Auto: 4 feat → **Corrected: 6 features**

```
Feat  RKfold MAE ± std      RKfold R² ± std
10    2.652 ± 0.338          0.756 ± 0.112
 9    2.630 ± 0.345          0.762 ± 0.102
 8    2.611 ± 0.343          0.765 ± 0.101
 7    2.593 ± 0.320          0.767 ± 0.106
 6    2.560 ± 0.322  ←BEST   0.774 ± 0.097  ←BEST
 5    2.598 ± 0.292          0.765 ± 0.095
 4    2.627 ± 0.334          0.757 ± 0.104  ←1-SE
 3    2.662 ± 0.341          0.767 ± 0.079  (high R² but worst MAE)
```

**Verdict: 6 features.** Undisputed peak — best MAE (2.560) and best R² (0.774) simultaneously. The 1-SE selection of 4 features is 0.067 MAE worse — the largest single-model error of the auto-selection rule. KRR at 6 features also has the best LOO R² (0.806) of any model at any feature count.

---

### RandomForest — Auto: 4 feat → **✅ Auto was CORRECT: 4 features**

```
Feat  RKfold MAE ± std      RKfold R² ± std
10    2.774 ± 0.428          0.752 ± 0.071
 9    2.740 ± 0.415          0.755 ± 0.081
 8    2.732 ± 0.405          0.758 ± 0.076
 7    2.789 ± 0.415          0.749 ± 0.071
 6    2.860 ± 0.443          0.741 ± 0.072
 5    2.796 ± 0.393          0.751 ± 0.076
 4    2.726 ± 0.360  ←BEST   0.765 ± 0.083  ←BEST
 3    2.878 ± 0.430          0.732 ± 0.077
```

**Verdict: 4 features.** Best MAE (2.726) and best R² (0.765) simultaneously. The 1-SE rule got this one right. The path is jagged (7 and 6 features perform worse than both 8 and 4), suggesting RandomForest is sensitive to which specific features are retained. The 4-feature set (pka, C_Polarization, lumo_energy, NPA_charge_C) is clearly optimal.

---

### AdaBoost — Auto: 4 feat → **Corrected: 7 features**

```
Feat  RKfold MAE ± std      RKfold R² ± std
10    2.684 ± 0.366          0.764 ± 0.086
 9    2.693 ± 0.382          0.751 ± 0.097
 8    2.699 ± 0.381          0.752 ± 0.099
 7    2.638 ± 0.378  ←BEST MAE  0.759 ± 0.097
 6    2.689 ± 0.347          0.760 ± 0.088
 5    2.642 ± 0.335          0.765 ± 0.077  ←BEST R²
 4    2.685 ± 0.369          0.752 ± 0.097  ←1-SE
 3    2.755 ± 0.326          0.744 ± 0.090
```

**Verdict: 7 features.** Best MAE (2.638). R²=0.759 vs best R²=0.765 at 5 feat. The 5-feature model has better R² but worse MAE (2.642). Since MAE is the primary metric, 7 features wins. However, the difference between 5 and 7 features is tiny (△MAE=0.004, △R²=0.006 in opposite directions), so either is defensible.

---

### GradientBoosting — Auto: 3 feat → **Corrected: 10 features**

```
Feat  RKfold MAE ± std      RKfold R² ± std
10    2.705 ± 0.388  ←BEST   0.759 ± 0.071  ←BEST
 9    2.756 ± 0.378          0.754 ± 0.079
 8    2.770 ± 0.373          0.744 ± 0.086
 7    2.744 ± 0.354          0.750 ± 0.087
 6    2.726 ± 0.343          0.750 ± 0.097
 5    2.737 ± 0.403          0.747 ± 0.089
 4    2.835 ± 0.475          0.736 ± 0.099
 3    2.725 ± 0.407          0.748 ± 0.086  ←1-SE
```

**Verdict: 10 features.** Best MAE (2.705) and best R² (0.759). However, I note that 10 features is ALL remaining features after only 4 were removed. This is unusually conservative. The 3-feature model (MAE=2.725) is only 0.020 worse — a case can be made that GradientBoosting doesn't benefit much from feature elimination. For this specific small dataset (N≈90), retaining more features with a boosting model that handles noise well is probably correct.

---

### LightGBM — Auto: 3 feat → **Corrected: 4 features**

```
Feat  RKfold MAE ± std      RKfold R² ± std
10    2.716 ± 0.414          0.753 ± 0.085
 9    2.771 ± 0.400          0.747 ± 0.083
 8    2.756 ± 0.400          0.749 ± 0.088
 7    2.791 ± 0.420          0.742 ± 0.088
 6    2.824 ± 0.360          0.731 ± 0.100
 5    2.807 ± 0.324          0.737 ± 0.086
 4    2.706 ± 0.311  ←BEST   0.758 ± 0.083  ←BEST
 3    2.743 ± 0.342          0.751 ± 0.091  ←1-SE
```

**Verdict: 4 features.** Best MAE (2.706) and best R² (0.758) simultaneously. The 1-SE rule selected 3 features which is 0.037 worse in MAE. LightGBM performance forms a U-shape: best at 10 and 4, worst in the middle (6-7). The 4-feature model edges out 10 by 0.010 MAE and 0.005 R².

---

### CatBoost — Auto: 3 feat → **Corrected: 8 features**

```
Feat  RKfold MAE ± std      RKfold R² ± std
10    2.751 ± 0.371          0.744 ± 0.102
 9    2.782 ± 0.355          0.744 ± 0.086
 8    2.674 ± 0.388  ←BEST   0.761 ± 0.083  ←BEST
 7    2.701 ± 0.448          0.758 ± 0.084
 6    2.759 ± 0.397          0.748 ± 0.085
 5    2.723 ± 0.401          0.759 ± 0.086
 4    2.841 ± 0.359          0.736 ± 0.094
 3    2.758 ± 0.369          0.745 ± 0.090  ←1-SE
```

**Verdict: 8 features.** Best MAE (2.674) and best R² (0.761) simultaneously. The 1-SE selection of 3 features is **0.084 MAE worse** — the second-largest auto-selection error. CatBoost at 8 features is dramatically better than at 3 or 4. The 8→7 step (MAE jumps from 2.674 to 2.701) shows the critical feature was removed there.

---

### XGBoost — Auto: 3 feat → **Corrected: 5 features**

```
Feat  RKfold MAE ± std      RKfold R² ± std
10    2.744 ± 0.427          0.749 ± 0.100
 9    2.819 ± 0.439          0.739 ± 0.106
 8    2.738 ± 0.422          0.758 ± 0.089  ← best R²
 7    2.787 ± 0.380          0.754 ± 0.091
 6    2.768 ± 0.337          0.751 ± 0.095
 5    2.733 ± 0.342  ←BEST MAE  0.759 ± 0.089  ←BEST R²
 4    2.894 ± 0.425          0.738 ± 0.084  ← sharp drop
 3    2.810 ± 0.346          0.750 ± 0.084  ←1-SE
```

**Verdict: 5 features.** Best MAE (2.733) and best R² (0.759) simultaneously. The 1-SE selection of 3 features is 0.077 worse in MAE. Note the catastrophic drop at 4 features (MAE=2.894) — the feature removed between 5 and 4 was essential. This created a "trap" for the 1-SE rule.

---

### KNR — Auto: 4 feat → **Corrected: 7 features**

```
Feat  RKfold MAE ± std      RKfold R² ± std
10    2.844 ± 0.407          0.724 ± 0.094
 9    2.698 ± 0.389          0.747 ± 0.086
 8    2.738 ± 0.403          0.749 ± 0.080
 7    2.671 ± 0.437  ←BEST   0.754 ± 0.084  ←BEST
 6    2.778 ± 0.378          0.727 ± 0.110
 5    2.764 ± 0.387          0.740 ± 0.089
 4    2.740 ± 0.435          0.732 ± 0.091  ←1-SE
 3    2.826 ± 0.413          0.711 ± 0.124
```

**Verdict: 7 features.** Best MAE (2.671) and best R² (0.754) simultaneously. The 1-SE selection of 4 features is 0.069 worse in MAE. KNR shows a clear peak at 7 features, with a sharp drop at 6 (MAE jumps from 2.671 to 2.778).

---

### ElasticNet — Auto: 3 feat → **Corrected: 4 features**

```
Feat  RKfold MAE ± std      RKfold R² ± std
10    2.902 ± 0.350          0.737 ± 0.076  ←BEST R²
 9    2.947 ± 0.353          0.731 ± 0.073
 8    2.967 ± 0.355          0.727 ± 0.074
 7    2.941 ± 0.366          0.733 ± 0.073
 6    2.908 ± 0.381          0.735 ± 0.075
 5    2.924 ± 0.389          0.732 ± 0.076
 4    2.897 ± 0.381  ←BEST MAE  0.736 ± 0.075
 3    2.903 ± 0.403          0.736 ± 0.068  ←1-SE
```

**Verdict: 4 features.** Best MAE (2.897). R²=0.736 vs best R²=0.737 at 10 feat — difference is 0.001, completely negligible. ElasticNet's performance is remarkably flat across the entire path (MAE range: 2.897–2.967, only 0.07 spread across all feature counts). This model genuinely doesn't care much about feature selection. I choose 4 because it has the best MAE, but any count from 4–10 is practically equivalent.

---

### MLP — Auto: 3 feat → **Corrected: 7 features**

```
Feat  RKfold MAE ± std      RKfold R² ± std
10    3.055 ± 0.399          0.705 ± 0.104
 9    3.112 ± 0.388          0.689 ± 0.112
 8    3.044 ± 0.353          0.699 ± 0.117
 7    3.024 ± 0.420  ←BEST   0.707 ± 0.097  ←BEST
 6    3.100 ± 0.484          0.693 ± 0.113
 5    3.047 ± 0.421          0.696 ± 0.119
 4    3.043 ± 0.425          0.705 ± 0.094
 3    3.063 ± 0.342          0.703 ± 0.097  ←1-SE
```

**Verdict: 7 features.** Best MAE (3.024) and best R² (0.707). However, even at its best, MLP is worse than the worst linear model (Ridge at 10 feat: MAE=2.678). **Recommend excluding MLP from publication** — neural networks are fundamentally unsuitable for this small dataset (N≈90 training samples).

---

### DecisionTree — Auto: 3 feat → **Corrected: 6 features**

```
Feat  RKfold MAE ± std      RKfold R² ± std
10    3.325 ± 0.572          0.642 ± 0.131
 9    3.361 ± 0.526          0.629 ± 0.138
 8    3.379 ± 0.501          0.632 ± 0.134
 7    3.348 ± 0.466          0.637 ± 0.135
 6    3.186 ± 0.475  ←BEST MAE  0.667 ± 0.141  ←BEST R²
 5    3.208 ± 0.371          0.653 ± 0.159
 4    3.185 ± 0.370          0.665 ± 0.146  (MAE = 3.185, essentially tied)
 3    3.250 ± 0.361          0.649 ± 0.150  ←1-SE
```

**Verdict: 6 features.** Best R² (0.667). MAE=3.186 is only 0.001 above the absolute best MAE at 4 feat (3.185). I break the virtual MAE tie using R². Like MLP, DecisionTree is a poor performer overall — the worst of all 16 models.

---

### GPlearn — Auto: 14 feat → **N/A (single-pass, no iterative elimination)**

```
Feat  RKfold MAE ± std      RKfold R² ± std
14    3.132 ± 0.772          0.502 ± 0.815
```

**Verdict: 14 features (all).** GPlearn does not participate in iterative SHAP-RFECV. It runs a single pass with all features, relying on genetic programming to perform inherent feature selection during formula evolution. Performance is poor (MAE=3.132, second-worst) with extreme variance (±0.77 MAE, ±0.82 R²), indicating formula instability across CV folds.

---

## Summary: 1-SE Auto-Selection vs. Independent Judgment

| Model | 1-SE Pick | My Pick | MAE at My Pick | R² at My Pick | Δ from 1-SE MAE |
|-------|-----------|---------|----------------|---------------|-----------------|
| SVR | 4 | **7** | 2.508 | 0.767 | −0.020 better |
| Ridge | 3 | **7** | 2.598 | 0.767 | −0.027 better |
| LinearRegression | 3 | **7** | 2.599 | 0.766 | −0.026 better |
| Lasso | 3 | **5** | 2.610 | 0.764 | −0.037 better |
| KRR | 4 | **6** | 2.560 | 0.774 | −0.067 better |
| RandomForest | 4 | **4** ✅ | 2.726 | 0.765 | 0 (correct) |
| AdaBoost | 4 | **7** | 2.638 | 0.759 | −0.047 better |
| GradientBoosting | 3 | **10** | 2.705 | 0.759 | −0.020 better |
| LightGBM | 3 | **4** | 2.706 | 0.758 | −0.037 better |
| CatBoost | 3 | **8** | 2.674 | 0.761 | −0.084 better |
| XGBoost | 3 | **5** | 2.733 | 0.759 | −0.077 better |
| KNR | 4 | **7** | 2.671 | 0.754 | −0.069 better |
| ElasticNet | 3 | **4** | 2.897 | 0.736 | −0.006 better |
| MLP | 3 | **7** | 3.024 | 0.707 | −0.039 better |
| DecisionTree | 3 | **6** | 3.186 | 0.667 | −0.064 better |
| GPlearn | 14 | **14** | 3.132 | 0.502 | N/A |

- **1-SE was wrong: 12/16 models** (75%)
- **1-SE was correct: 1 model** (RandomForest; ElasticNet path is too flat to matter; GPlearn is N/A)
- **Average MAE improvement from correction: 0.045 kcal/mol**
- **Largest correction: CatBoost (+0.084 MAE)**

---

## Why the 1-SE Rule Failed Systematically

The Fractional 1-SE rule (α=0.25) has a fundamental flaw on this dataset:

1. **Wide bandwidth due to small N**: With N≈90 training samples, the RepeatedKFold fold-level variance is inflated, producing wide σ values. α=0.25 × σ creates a bandwidth that swallows many suboptimal feature counts.

2. **The "bad step" trap**: If a single feature removal causes a sharp performance degradation (e.g., CatBoost 8→7: MAE +0.027, SVR 7→6: MAE +0.040), then the subsequent feature count falls outside the threshold. But if the NEXT removal (e.g., SVR 6→5: MAE +0.105) is even worse, and then 5→4 somehow recovers (MAE −0.125), the rule picks 4 while skipping over the genuinely optimal 7.

3. **min_features floor**: The `--min_features=3` argument means the rule can't explore below 3. Many models hit this floor and got stuck there because 3 features happened to be within the 1-SE bandwidth.

---

## GPlearn Formula

The GPlearn model (single-pass, all 14 features) discovered:

**Raw gplearn output:**
```
sub(div(X0, 0.011), neg(inv(div(inv(log(mul(X3, X5))), neg(sqrt(add(X7, X6)))))))
```

**With feature names** (X0=pka, X3=C_s, X5=homo_energy, X6=lumo_energy, X7=electronegativity):
```
AE = pka/0.011 − sqrt(electronegativity + lumo_energy) × ln(C_s × homo_energy)
```

The formula uses only 5 of 14 features — GP performs implicit feature selection. The equation has a physically interpretable form: pKa dominates (×90.91 coefficient), modulated by an electronic term involving LUMO/electronegativity and a nucleophilicity term involving C_s/HOMO.

**⚠️ The formula uses gplearn native 0-indexed variable names (X0, X3...). The formula-to-feature-name mapping was buggy during this run (now fixed in revision_16). Future runs will output properly named formulas.**

---

## Feature Importance Across Models (using corrected feature counts)

| Feature | # Models | Selected by |
|---------|----------|-------------|
| **pka** | 16/16 | ALL models — universal dominant feature |
| **C_Polarization** | 12/16 | Most tree-based + SVR + ElasticNet + KNR |
| **lumo_energy** | 6/16 | CatBoost, ElasticNet, KNR, RandomForest, XGBoost, MLP |
| **C_s** | 4/16 | Linear models + SVR |
| **NPA_charge_B** | 4/16 | Linear models + KRR |
| **VBur_C** | 4/16 | AdaBoost, DecisionTree, GradientBoosting, LightGBM |
| **dipole** | 2/16 | KRR, SVR |
| **homo_energy** | 2/16 | KNR, GPlearn |
| **electronegativity** | 2/16 | AdaBoost, GPlearn |
| **NPA_charge_C** | 2/16 | MLP, RandomForest |
| **Mulliken_charge_C** | 1/16 | KRR |
| **Bond_Length** | 1/16 | GPlearn (all features) |
| **B_s, Mulliken_charge_B** | 0 | Never selected — universally eliminated |

---

## Issues & Recommendations

### Critical
1. **Re-run all models with corrected feature counts.** The 1-SE auto-selection was wrong for 75% of models. Use `--force_n_features` to lock in the corrected values, or fix the selection logic.

### High Priority
2. **KRR at 6 features is the hidden champion.** LOO R²=0.806 is the best of any model at any feature count. Consider making KRR the primary model for publication alongside SVR.
3. **MLP and DecisionTree should be excluded from publication.** Both perform worse than trivial linear baselines. Neural networks cannot learn meaningful patterns from N≈90 samples.
4. **GPlearn needs re-running with formula-mapping fix.** The current formula uses raw X0..Xn names. After revision_16's regex fix, the formula will be human-readable.

### Medium Priority
5. **Reduce min_features to 1** to let models explore below the current 3-feature floor. Several linear models might perform even better with 2 features.
6. **Investigate the SVR/LightGBM "U-shaped" path.** Both models perform well at high feature counts AND low feature counts, but worse in the middle. This suggests interacting features that are harmful unless their partner is also present.
