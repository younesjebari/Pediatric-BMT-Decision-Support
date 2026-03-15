# Pediatric-BMT-Decision-Support

Outil d'aide a la decision medicale base sur le Machine Learning pour predire les resultats des greffes de moelle osseuse pediatriques. Le projet combine un modele XGBoost avec l'explicabilite SHAP et une interface web Flask.

---

## Structure du Projet

```
.
├── app/
│   ├── app.py                  # Backend Flask (auth, prediction, SHAP, admin)
│   ├── static/
│   │   ├── css/style.css       # Theme medical, animations, responsive
│   │   └── js/main.js          # Interactions cote client
│   └── templates/
│       ├── base.html           # Template de base (navbar, footer)
│       ├── auth/
│       │   ├── login.html      # Page de connexion
│       │   └── register.html   # Page d'inscription
│       ├── admin/
│       │   └── dashboard.html  # Gestion des utilisateurs (admin)
│       ├── home.html           # Accueil (hero, features)
│       ├── predict.html        # Wizard de prediction (4 etapes)
│       ├── results.html        # Resultats + SHAP individuel
│       └── shap.html           # Analyse SHAP globale
├── src/
│   ├── data_processing.py      # Preprocessing (selection, imputation, outliers)
│   ├── train_model.py          # Benchmark 4 modeles + entrainement
│   └── evaluate_model.py       # Evaluation + analyse SHAP globale
├── data/
│   └── bone-marrow.arff        # Dataset (187 patients, 37 attributs)
├── models/
│   └── final_model.joblib      # Modele XGBoost entraine
├── tests/
│   ├── test_data_processing.py # Tests du preprocessing
│   ├── test_train_model.py     # Tests du pipeline d'entrainement
│   └── test_evaluate_model.py  # Tests de l'evaluation et SHAP
├── notebooks/
│   └── eda.ipynb               # Analyse exploratoire
├── .github/workflows/
│   └── tests.yml               # CI GitHub Actions
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Analyse Exploratoire des Donnees (EDA)

- **Tests statistiques :** Mann-Whitney U pour identifier les variables a pouvoir predictif reel (p < 0.05).
  - `Recipientage` (p=0.0053) et `CD34kgx10d6` (p=0.0074) sont les biomarqueurs les plus critiques.
- **Analyse de correlation (Spearman) :** Multicolinearite de 85% entre `Rbodymass` et `Recipientage`.
- **Anti-fuite :** Exclusion de `survival_time` (connue uniquement apres l'issue de la greffe).

## Preprocessing et Optimisation Memoire

- **Selection de 11 variables** parmi 37 attributs (p-value < 0.05).
- **Downcasting :** float64 vers float32, int64 vers int32 -- reduction de ~50% de la RAM.
- **Imputation :** Valeurs manquantes remplacees par la mediane.
- **Outliers :** Detection et traitement via IQR.

## Entrainement (Benchmark)

Quatre modeles compares avec validation croisee :

| Modele       | Accuracy | ROC-AUC | Rappel  |
|:-------------|:---------|:--------|:--------|
| RandomForest | 73.7%    | 71.8%   | 58.8%   |
| **XGBoost**  | **76.3%**| **70.0%**| **64.7%**|
| LightGBM     | 73.7%    | 71.1%   | 58.8%   |
| SVM          | 68.4%    | 67.8%   | 47.1%   |

Le modele XGBoost est retenu et sauvegarde dans `models/final_model.joblib`.

## Explicabilite (SHAP)

Chaque prediction est accompagnee d'une analyse SHAP individuelle montrant l'impact de chaque variable clinique sur le pronostic. Une analyse SHAP globale est aussi disponible sur la page dediee.

## Interface Web (Flask)

L'application est decoupee en 4 pages avec un systeme d'authentification :

1. **Connexion / Inscription** -- comptes utilisateurs avec hash des mots de passe (werkzeug).
2. **Accueil** -- presentation du projet, performances du modele, fonctionnalites.
3. **Prediction** -- wizard en 4 etapes (Patient, Donneur, Greffe, Recapitulatif) puis affichage des resultats avec jauge de survie et barres SHAP.
4. **Analyse SHAP** -- summary plot global et guide d'interpretation des 11 variables.

Un compte **admin** est cree automatiquement au demarrage (`admin` / `admin`). L'admin peut gerer les utilisateurs (promouvoir, supprimer) depuis le panneau d'administration.

## Tests (17 tests)

```bash
pytest tests/ -v
```

- `test_data_processing.py` : optimisation memoire, imputation.
- `test_train_model.py` : pipeline 11 features, cible binaire, SMOTE, modele fonctionnel.
- `test_evaluate_model.py` : accuracy et ROC-AUC > 60%, matrice de confusion, SHAP.

---

## Lancement

### Localement

```bash
pip install -r requirements.txt
python app/app.py
```

L'application est accessible sur `http://127.0.0.1:5000`.

### Via Docker

```bash
docker build -t bmt-app .
docker run -p 5000:5000 bmt-app
```

### Entrainement du modele

```bash
python3 src/data_processing.py
python3 src/train_model.py
python3 src/evaluate_model.py
```

---

## Technologies

- **ML :** XGBoost, scikit-learn, SHAP, imbalanced-learn (SMOTE)
- **Web :** Flask, Jinja2, HTML/CSS/JS
- **Data :** pandas, numpy, scipy
- **Tests :** pytest
- **CI/CD :** GitHub Actions
- **Conteneurisation :** Docker
