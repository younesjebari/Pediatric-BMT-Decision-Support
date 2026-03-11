# Pediatric-BMT-Decision-Support
A machine learning application designed to assist physicians in predicting the success rate of pediatric bone marrow transplants using explainable AI (SHAP).
## ⚡ Optimisation Mémoire

La fonction `optimize_memory(df)` dans `src/data_processing.py`
réduit l'usage RAM en convertissant les types de données :

| Type original | Type optimisé | Réduction |
|---------------|---------------|-----------|
| float64       | float32       | ~50%      |
| int64         | int32         | ~50%      |

### Résultats mesurés sur le dataset BMT :

| | Mémoire |
|-|---------|
| Avant optimisation | 0.12 MB |
| Après optimisation | 0.06 MB |
| **Réduction totale** | **~50%** |

> Preuve reproductible : lancer `notebooks/eda.ipynb` section 6.
# 🏥 Support à la décision pour la greffe pédiatrique (BMT)

Ce projet implémente une solution de Machine Learning pour prédire la survie des patients pédiatriques après une greffe de moelle osseuse.

## 📈 Performance du Modèle Final (Random Forest)
Le modèle sélectionné offre les performances suivantes sur les cas critiques :
- **Rappel (Recall) : 81.0%** (Capacité à détecter les patients à haut risque)
- **Précision : 70.8%**
- **Score ROC-AUC : 71.0%**

## 🛠️ Structure Technologique
- **Traitement :** Nettoyage automatisé des données (imputation par médiane/mode).
- **Équilibrage :** Technique SMOTE pour renforcer l'apprentissage sur les cas de décès.
- **Modélisation :** Pipeline robuste utilisant `RandomForestClassifier`.

## 🚀 Utilisation
Pour ré-entraîner le modèle avec de nouvelles données :
```powershell
.\.venv\Scripts\python.exe src/train_model.py


