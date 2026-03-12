# Pediatric-BMT-Decision-Support
A machine learning application designed to assist physicians in predicting the success rate of pediatric bone marrow transplants using explainable AI (SHAP).
## Optimisation Mémoire

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
#  support à la décision pour la greffe pédiatrique (BMT)

Ce projet implémente une solution de Machine Learning pour prédire la survie des patients pédiatriques après une greffe de moelle osseuse.

##  Performance du Modèle Final (Random Forest)
Le modèle sélectionné offre les performances suivantes sur les cas critiques :
- **Rappel (Recall) : 81.0%** (Capacité à détecter les patients à haut risque)
- **Précision : 70.8%**
- **Score ROC-AUC : 71.0%**

##  Analyse et Explicabilité Médicale 

En tant que responsable de l'explicabilité, j'ai intégré la librairie SHAP pour garantir que chaque prédiction du modèle puisse être interprétée et validée par un clinicien.

### 1. Interprétation Globale (Summary Plot)
L'analyse SHAP sur l'ensemble du dataset a permis d'identifier les biomarqueurs et facteurs cliniques les plus influents pour la survie post-greffe :
-**Dose de CD34+ (×10⁶/kg) :** Le facteur le plus déterminant. Une dose élevée est systématiquement associée à une augmentation des chances de succès de la prise de greffe.
-**Âge du Donneur :** On observe qu'un donneur plus jeune tend à améliorer les scores de survie à long terme.
-**Compatibilité HLA :** Les disparités (mismatch) impactent négativement la prédiction, alertant le médecin sur des risques potentiels de complications.

### 2. Transparence Clinique
Chaque prédiction générée par l'interface Streamlit est accompagnée d'une visualisation explicative. Cela permet au médecin :
-De comprendre quels facteurs spécifiques ont poussé le modèle vers une prédiction de "Succès" ou d'"Échec".
-De comparer la logique de l'IA avec son expertise médicale pour une décision finale plus sûre.

> **Livrable :** Le graphique d'explication globale est disponible dans l'interface finale et a été utilisé pour justifier la sélection du modèle Random Forest, dont les décisions restaient les plus cohérentes avec la littérature médicale actuelle.

##  Structure Technologique
- **Traitement :** Nettoyage automatisé des données (imputation par médiane/mode).
- **Équilibrage :** Technique SMOTE pour renforcer l'apprentissage sur les cas de décès.
- **Modélisation :**Pipeline robuste utilisant `RandomForestClassifier`.

## Utilisation de l'IA (Prompt Engineering)

<<<<<<< Updated upstream
Dans le cadre de ce projet d'analyse de données médicales, notre premier objectif technique était de mettre en place un flux de travail collaboratif solide et de maîtriser les commandes Git/GitHub (Push, Pull, gestion des branches).

Afin de disposer rapidement d'une base de code pertinente à nous partager et à fusionner, nous avons utilisé l'assistant IA **Gemini**. L'objectif n'était pas de faire écrire le projet par l'IA, mais de générer des squelettes de code pour nos différents modèles de Machine Learning (destinés à la prédiction de l'évolution des patients), nous permettant ainsi de nous concentrer sur la pratique de GitHub.

### Exemples de requêtes (Prompts) utilisées :
* Génère un script Python clair utilisant scikit-learn pour entraîner un modèle de Random Forest. Le code doit inclure la séparation des données (train/test split) et être prêt à être poussé sur un repository.
* Écris le code d'une régression logistique de base pour de la classification, avec l'affichage des métriques d'évaluation standard (accuracy, matrice de confusion).
* Crée une structure de base pour une application Streamlit simple permettant d'afficher un DataFrame Pandas.

### Méthodologie appliquée :
1. **Génération :** Création des algorithmes de base via Gemini.
2. **Répartition :** Chaque membre de l'équipe a pris en charge un modèle spécifique.
3. **Collaboration :** Utilisation de notre repository GitHub pour créer des branches, faire nos *commits*, et fusionner le tout via des *Pull Requests*.
4. **Adaptation :** Le code généré a ensuite été relu, débuggé humainement et adapté aux spécificités de notre jeu de données.
=======
Pour ré-entraîner le modèle avec de nouvelles données :
```powershell
.\.venv\Scripts\python.exe src/train_model.py
>>>>>>> Stashed changes
## Résultats Tests Statistiques

Variables numériques significatives :
- Rbodymass (p=0.0033)
- CD3dkgx10d8 (p=0.0016)
- CD34kgx10d6 (p=0.0070)

Variables catégorielles significatives :
- Disease (p=0.0185)
- Relapse (p=0.0001)
- extcGvHD (p=0.0000)