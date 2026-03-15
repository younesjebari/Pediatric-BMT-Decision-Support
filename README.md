# 🏥 Rapport de Projet : Pediatric-BMT-Decision-Support

Ce projet implémente une solution de **Machine Learning de précision** pour assister les cliniciens dans la prédiction des résultats de greffes de moelle osseuse chez les enfants. Il allie rigueur statistique, optimisation logicielle et explicabilité de l'IA.

---

## 📊 1. Analyse Exploratoire des Données (EDA)
L'étape initiale consistait à extraire la "vérité" biologique du dataset `bone-marrow.arff`.

* **Tests Statisiques :** Application des tests de **Mann-Whitney U** pour identifier les variables ayant un pouvoir prédictif réel ($p < 0.05$).
    * **Recipientage** ($p=0.0053$) et **CD34kgx10d6** ($p=0.0074$) ont été isolés comme les biomarqueurs les plus critiques.
* **Analyse de Corrélation (Spearman) :** Identification d'une multicolinéarité de **85%** entre `Rbodymass` et `Recipientage`.
    * **Décision :** Suppression de la masse corporelle pour éviter la redondance et simplifier le modèle sans perte d'information.

## ⚙️ 2. Prétraitement & Optimisation Mémoire (Data Processing)
Le traitement des données a été conçu pour être à la fois robuste et léger.

### 💾 Choix des Formats et Optimisation RAM
Pour garantir que l'application puisse tourner sur des systèmes avec peu de ressources, nous avons implémenté une fonction de **Downcasting de Type** :
* **De Float64 vers Float32 :** Réduction de 64 bits à 32 bits pour les variables continues (Âge, Doses).
* **De Int64 vers Int32 :** Réduction pour les variables discrètes (Codes de maladies).
* **Résultat :** Une réduction de l'usage RAM de **~50%** (de 0.12 MB à 0.06 MB), optimisant ainsi les temps de calcul et de chargement.

### 🛡️ Stratégie Anti-Fuite (Data Leakage)
* **Exclusion de `survival_time` :** Bien que très corrélée au succès ($r = -0.76$), cette variable est une fuite de données car elle n'est connue qu'après l'issue de l'opération. Sa suppression garantit une prédiction réaliste en conditions réelles.

## 🧠 3. Entraînement & Pipeline (Training)
* **Algorithme :** Utilisation d'une **Forêt Aléatoire (Random Forest)** pour sa capacité à gérer les interactions complexes et non-linéaires entre les biomarqueurs.
* **Gestion des Catégories :** Encodage automatique des variables textuelles (ex: types de maladies) en codes numériques pour permettre l'apprentissage machine.
* **Sauvegarde :** Exportation du modèle via `joblib` pour un déploiement instantané.

## 📈 4. Évaluation & Validation Clinique
Le modèle a été validé sur un ensemble de données indépendant avec des scores très élevés.

| Métrique | Score | Signification Clinique |
| :--- | :--- | :--- |
| **ROC-AUC** | **95.86%** | Capacité quasi-parfaite de séparation des cas. |
| **Rappel (Recall) Décès** | **92.9%** | Seulement 7% de risques non-détectés (Sécurité max). |
| **Précision Décès** | **96.3%** | Très peu de fausses alertes pour les médecins. |
| **Exactitude (Accuracy)** | **95.0%** | Fiabilité globale sur l'ensemble des prédictions. |

## 🧬 5. Explicabilité & IA Transparente (SHAP)
Pour briser l'effet "boîte noire", nous avons intégré la technologie **SHAP (SHapley Additive exPlanations)**.
* Chaque prédiction est accompagnée d'une explication visuelle montrant quels biomarqueurs (ex: Dose CD34+) ont poussé le modèle vers un diagnostic de succès ou de risque.

## 🧪 6. Tests Unitaires (QA)
Fiabilisation du code source via `pytest` :
* **test_optimize_memory_types :** Vérification du succès du downcasting (float32).
* **test_imputation_values :** Vérification que les valeurs manquantes sont remplacées par la médiane sans altérer la distribution.
* **test_categorical_encoding :** Validation de la conversion des textes en nombres.

## 🌐 7. Déploiement (Streamlit & Docker)
* **Interface Streamlit :** Un dashboard élégant divisé en deux sections : saisie des paramètres cliniques à gauche, diagnostic visuel et probabilités à droite.
* **Conteneurisation Docker :** Création d'une image `python:3.12-slim` pour garantir que l'application fonctionne de manière identique sur n'importe quel ordinateur, sans conflit de bibliothèque.

---

## 🛠️ Manuel d'Utilisation
### Localement
1. Activer l'environnement : `.\.venv\Scripts\Activate.ps1`
2. Lancer l'app : `streamlit run app/app.py`

### Via Docker
1. Build : `docker build -t bmt-app .`
2. Run : `docker run -p 8501:8501 bmt-app`