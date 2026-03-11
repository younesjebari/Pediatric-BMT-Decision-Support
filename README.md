# 🏥 Pediatric-BMT-Decision-Support

A Machine Learning application designed to assist physicians in predicting the success rate of pediatric bone marrow transplants (BMT) using explainable AI.

## 📈 Final Model Performance (Random Forest)

The model has been validated on an independent test set (data never seen during training) to ensure realistic results in a clinical setting.

| Metric | Result | Medical Significance |
| :--- | :--- | :--- |
| **Recall (Death cases)** | **81.0%** | Ability to correctly identify high-risk patients (critical cases). |
| **Precision** | **70.8%** | Reliability of the alert to limit false alarms for medical staff. |
| **ROC-AUC Score** | **71.0%** | Overall ability of the model to distinguish between survival classes. |

---

## 🛠️ Methodology & Technical Rigor

### 🛡️ Prevention of "Data Leakage"
Initially, the model showed a 100% accuracy score, indicating a mathematical "leak." We corrected this instability by:
* **Dropping the `survival_time` column**: This information is not known at the time a doctor must make a decision upon admission.
* **Train/Test Split**: The model is trained on 80% of the data and evaluated on the remaining 20% (unseen data) to simulate real-world patient arrivals.

### ⚖️ Class Balancing (SMOTE)
Since the dataset was imbalanced (more survivors than deaths), we applied **SMOTE** (*Synthetic Minority Over-sampling Technique*) to the training set to improve the model's ability to detect critical cases.

---

## ⚡ Memory Optimization

The `optimize_memory(df)` function in `src/data_processing.py` reduces RAM usage by converting data types without losing precision:

| Original Type | Optimized Type | Reduction |
| :--- | :--- | :--- |
| float64 | float32 | ~50% |
| int64 | int32 | ~50% |

### Measured Results on BMT Dataset:
* **Before Optimization:** 0.12 MB
* **After Optimization:** 0.06 MB
* **Total Gain:** **~50% reduction**

> *Reproducible proof: run `notebooks/eda.ipynb` section 6.*

---

## 🚀 Installation and Usage

### 1. Environment Setup
```powershell
# Create virtual environment
python -m venv .venv

# Activate environment
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt