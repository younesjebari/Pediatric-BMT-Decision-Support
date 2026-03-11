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


