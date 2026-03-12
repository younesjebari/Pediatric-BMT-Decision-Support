# 1.Pediatric-BMT-Decision-Support
A machine learning application designed to assist physicians in predicting the success rate of pediatric bone marrow transplants using explainable AI (SHAP).
## 1.1-Optimisation Mémoire

La fonction `optimize_memory(df)` dans `src/data_processing.py`
réduit l'usage RAM en convertissant les types de données :

| Type original | Type optimisé | Réduction |
|---------------|---------------|-----------|
| float64       | float32       | ~50%      |
| int64         | int32         | ~50%      |

## 1.2-1Résultats mesurés sur le dataset BMT :

| | Mémoire |
|-|---------|
| Avant optimisation | 0.12 MB |
| Après optimisation | 0.06 MB |
| **Réduction totale** | **~50%** |

> Preuve reproductible : lancer `notebooks/eda.ipynb` section 6.
# 2.Support à la décision pour la greffe pédiatrique (BMT)

Ce projet implémente une solution de Machine Learning pour prédire la survie des patients pédiatriques après une greffe de moelle osseuse.

## 2.1-Performance du Modèle Final (Random Forest)
Le modèle sélectionné offre les performances suivantes sur les cas critiques :
- **Rappel (Recall) : 81.0%** (Capacité à détecter les patients à haut risque)
- **Précision : 70.8%**
- **Score ROC-AUC : 71.0%**
## 2.2-Analyse et Explicabilité Médicale
En tant que responsable de l'explicabilité, j'ai intégré la librairie **SHAP (SHapley Additive exPlanations)** pour garantir que chaque prédiction du modèle puisse être interprétée et validée par un clinicien.
## 2.3-Interprétation Globale (Summary Plot)
L'analyse SHAP sur l'ensemble du dataset a permis d'identifier les biomarqueurs et facteurs cliniques les plus influents pour la survie post-greffe :
* **Dose de CD34+ (×10⁶/kg) :** Le facteur le plus déterminant. Une dose élevée est systématiquement associée à une augmentation des chances de succès de la prise de greffe.
* **Âge du Donneur :** On observe qu'un donneur plus jeune tend à améliorer les scores de survie à long terme.
* **Compatibilité HLA :** Les disparités (mismatch) impactent négativement la prédiction, alertant le médecin sur des risques potentiels de complications.
## 2.4-Transparence Clinique
Chaque prédiction générée par l'interface Streamlit est accompagnée d'une visualisation explicative. Cela permet au médecin :
* De comprendre quels facteurs spécifiques ont poussé le modèle vers une prédiction de "Succès" ou d'"Échec".
* De comparer la logique de l'IA avec son expertise médicale pour une décision finale plus sûre.

> **Livrable :** Le graphique d'explication globale est disponible dans l'interface finale et a été utilisé pour justifier la sélection du modèle, dont les décisions restaient les plus cohérentes avec la littérature médicale actuelle.
# 3-Structure Technologique
- **Traitement :** Nettoyage automatisé des données (imputation par médiane/mode).
- **Équilibrage :** Technique SMOTE pour renforcer l'apprentissage sur les cas de décès.
- **Modélisation :** Pipeline robuste utilisant `RandomForestClassifier`.

# 4-Utilisation
Pour ré-entraîner le modèle avec de nouvelles données :
```powershell
.\.venv\Scripts\python.exe src/train_model.py