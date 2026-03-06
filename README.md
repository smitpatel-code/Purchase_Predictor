# Purchase_Predictor
# 🛒 Smart Shop — E-Commerce Purchase Intent Prediction

> A machine learning project that predicts whether an online shopper will complete a purchase, using Decision Tree classification with systematic hyperparameter tuning and pruning strategies.

---

## 📌 Overview

Online retailers lose billions annually to cart abandonment. **Smart Shop** tackles this problem by analyzing user session behavior and predicting purchase intent in real time. Using the [Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset), this project walks through the complete ML pipeline — from raw data to an optimized, pruned classifier.

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | `shop_smart_ecommerce.csv` |
| Records | 12,330 sessions |
| Features | 18 (numerical + categorical) |
| Target | `Revenue` (True/False — did user purchase?) |
| Missing Values | None |

### Feature Summary

| Category | Features |
|---|---|
| Page Activity | `Administrative`, `Informational`, `ProductRelated` (counts & durations) |
| Engagement Metrics | `BounceRates`, `ExitRates`, `PageValues` |
| Temporal | `Month`, `SpecialDay`, `Weekend` |
| Technical | `OperatingSystems`, `Browser`, `Region`, `TrafficType` |
| User Type | `VisitorType` (New / Returning / Other) |

---

## 🔧 Methodology

### 1. Data Preprocessing
- Loaded and inspected the dataset — confirmed **zero null values**
- Applied **Label Encoding** for boolean columns (`Weekend`, `Revenue`)
- Applied **One-Hot Encoding** (drop-first) for categorical columns (`Month`, `VisitorType`)

### 2. Train/Test Split
- Split ratio: **67% train / 33% test**
- `random_state=42` for reproducibility

### 3. Baseline Model
- Trained a default `DecisionTreeClassifier` with no depth constraints
- Baseline accuracy: **~85.3%** (overfitting likely due to unbounded depth)

### 4. Pre-Pruning (Hyperparameter Tuning)
Systematically swept key hyperparameters to find the optimal configuration:

**`max_depth` sweep:**
| Depth | Accuracy |
|---|---|
| 2 | 88.50% |
| 3 | 88.70% |
| **4** | **89.16%** ✅ |
| 5 | 89.09% |
| 6–20 | Declining |

**`min_samples_split` sweep** (fixed `max_depth=4`):
- Accuracy remained stable at **89.16%** across all tested values — minimal sensitivity

**`min_samples_leaf` sweep** (fixed `max_depth=4`, `min_samples_split=2`):
- Accuracy remained stable at **89.16%** — tree structure was robust to leaf size changes

### 5. Post-Pruning (Cost Complexity Pruning)
- Used `cost_complexity_pruning_path` to extract the full sequence of `ccp_alpha` values
- Trained a separate tree for each alpha; evaluated on test set
- **Best alpha:** `0.000471`
- **Post-pruning accuracy: 89.31%** 🏆

---

## 📈 Results

| Model | Accuracy |
|---|---|
| Baseline (no constraints) | 85.28% |
| Pre-pruned (`max_depth=4`) | 89.16% |
| **Post-pruned (CCP)** | **89.31%** ✅ |

Post-pruning via Cost Complexity Pruning yielded the best generalization, reducing tree complexity while slightly improving accuracy.

---
