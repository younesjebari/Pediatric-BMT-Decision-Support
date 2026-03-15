## Contexte et Objectif du Projet

La greffe de moelle osseuse est une procédure médicale complexe. Avant l'opération, les médecins doivent prendre des décisions cruciales et ajuster des paramètres pré-opératoires précis comme les dosages de cellules CD34+ et CD3+ pour maximiser les chances de prise de la greffe tout en minimisant les risques d'échec ou de rechute. Le corps humain étant imprévisible, anticiper l'issue de cette procédure représente un défi clinique majeur.

**L'objectif de ce projet est de concevoir un Système d'Aide à la Décision basé sur l'Intelligence Artificielle pour accompagner les hématologues.** En s'appuyant sur des algorithmes de Machine Learning entraînés sur des données historiques de patients, la plateforme permet de :
**Évaluer instantanément** la probabilité de survie post-greffe à partir de 6 variables cliniques clés.
**Expliquer la prédiction** grâce à l'intelligence artificielle explicable (SHAP), garantissant une totale transparence médicale.
**Simuler des traitements** via une interface interactive où le médecin peut virtuellement ajuster les doses de cellules souches pour optimiser le pronostic du patient avant l'intervention réelle.

## Architecture et Organisation du Code

Ce projet a été structuré en suivant les standards de développement logiciel. L'objectif est de séparer clairement l'interface utilisateur , la logique d'Intelligence Artificielle et l'intégration continue.

```text
Pediatric-BMT-Decision-Support/
├── .github/workflows/   # CI/CD : Automatisation des tests à chaque "push" (GitHub Actions)
├── app/                 # Front-end : Application web médicale interactive
│   └── app.py           # Interface principale codée avec Streamlit
├── data/                # Base de données clinique anonymisée (.arff)
├── models/              # Modèles IA entraînés, sauvegardés et prêts à l'emploi (.joblib)
├── notebooks/           # Phase de recherche : Analyse exploratoire des données (EDA)
├── src/                 # Back-end ML : Cœur de l'algorithme
│   ├── data_processing.py # Nettoyage et préparation des données médicales
│   ├── evaluate_model.py  # Calcul des performances et métriques
│   └── train_model.py     # Pipeline d'entraînement des modèles (XGBoost, SVM...)
├── tests/               # Scripts de validation continue (exécutés via Pytest)
├── Dockerfile           # Fichier de configuration pour la conteneurisation
└── requirements.txt     # Liste stricte des dépendances Python
```

## Analyse Exploratoire des Données

Le fichier `notebooks/eda.ipynb` constitue la fondation scientifique de notre démarche. Avant d'entraîner le moindre algorithme, une analyse rigoureuse du jeu de données brut  a été menée pour comprendre les dynamiques cliniques et garantir la fiabilité du modèle.

Les étapes clés de cette analyse comprennent :

**Nettoyage et Prétraitement :** Décodage des chaînes de caractères brutes, traitement des valeurs manquantes: imputation par la médiane pour les variables continues et encodage des variables catégorielles.
**Prévention du Data Leakage :** C'est une étape critique du projet. Nous avons identifié et supprimé les variables "post-opératoires" comme le temps de récupération des plaquettes `PLTrecovery` ou la GvHD qui gonflaient artificiellement les scores de prédiction, car ces informations sont inconnues du médecin avant la greffe.
**Sélection des Caractéristiques (Feature Selection) :** Réduction de la dimensionnalité de 37 variables brutes à 6 indicateurs pré-opératoires clés (Âge, Masse corporelle, Maladie d'origine, Antécédent de rechute, Dosages CD34+ et CD3+) par des tests statistiques des lois de Chi 2 qui conserve les variables qualitatives qui influencent sur le succes et MANN-WHITNEY qui traite les variables quantitatives : continues afin d'en extraire celles qui ont un poids dans le taux de succes.
**Étude des Corrélations :** Visualisation des relations entre les biomarqueurs et la cible :statut de survie via des matrices de corrélation et des distributions statistiques.

##  Entraînement et Comparaison des Modèles

Le script `src/train_model.py` constitue le cœur de notre Intelligence Artificielle. Il automatise le flux de travail de Machine Learning, de la préparation des données jusqu'à l'exportation du modèle optimal pour le déploiement.

Notre pipeline de modélisation intègre les étapes critiques suivantes :

**Standardisation des Données :`StandardScaler`:** Mise à l'échelle des variables continues comme le poids ou les dosages cellulaires. Cette étape est indispensable pour garantir les performances des algorithmes basés sur les distances géométriques comme le SVM.
**Équilibrage des Classes `SMOTE`: :** Les jeux de données médicaux sont souvent déséquilibrés (sur-représentation des succès de greffe par rapport aux échecs). Nous utilisons la technique de sur-échantillonnage synthétique SMOTE sur les données d'entraînement pour éviter que l'IA ne développe un biais d'optimisme.
**Banc d'Essai Multi-Algorithmes :** Le script met en compétition 4 algorithmes de pointe reconnus pour leurs performances sur les données tabulaires de santé :
    * Random Forest
    * XGBoost
    * LightGBM
    * SVM
**Sérialisation et Déploiement :MLOps:** Le modèle obtenant la meilleure précision Accuracy / ROC-AUC sur le jeu de test est automatiquement élu. Le script exporte alors trois artefacts cruciaux dans le dossier `models/` : le modèle vainqueur `final_model.joblib`, le traducteur d'échelle `scaler.joblib` et le schéma des variables `features_list.joblib`, garantissant une intégration sans faille avec l'application Web.

## Interface Utilisateur & Dashboard Clinique 

Le fichier `app/app.py` contient l'application Web interactive développée avec le framework **Streamlit**. L'interface a été conçue sur mesure via du CSS injecté pour offrir une ergonomie digne d'un véritable logiciel hospitalier, évitant la surcharge cognitive des praticiens.

L'application est structurée autour de fonctionnalités avancées, développées spécifiquement pour ce projet :

* Authentification Sécurisée :Système de connexion et d'inscription  avec stockage sécurisé des utilisateurs via `users.json` et gestion des sessions (`st.session_state`).
* Design & Animations SVG Sur Mesure : Intégration de logos vectoriels SVG animés en CSS directement générés en Python, offrant un branding professionnel sans alourdir l'application.
* Gestionnaire de Rendez-vous Intégré : Développement d'un widget calendrier interactif permettant au médecin de planifier, visualiser et supprimer ses consultations directement depuis son tableau de bord.
* Moteur d'Explicabilité SHAP : Après la saisie des données patient, l'application ne donne pas qu'un score probabiliste. Elle génère dynamiquement des graphiques d'importance des variables barres horizontales à l'aide de Matplotlib et SHAP, justifiant ainsi la décision de l'algorithme.
* Générateur de Rapports Médico-Légaux ReportLab:Implémentation complète d'un moteur d'export PDF métier. En un clic, le médecin télécharge un rapport clinique traçable contenant les métadonnées de la session, le profil patient, le diagnostic IA, les graphiques SHAP et les avertissements légaux, prêt à être joint au dossier médical.





##Guide d'utilisation de l'interface : 

L'interface a été pensée pour s'intégrer naturellement dans le flux de travail d'un professionnel de santé. Voici les étapes pour réaliser une simulation clinique :

Étape 1 : Authentification Sécurisée
*Au lancement de l'application, l'écran de verrouillage PediaBMT apparaît.
*Si vous êtes un nouvel utilisateur :Allez dans l'onglet *Créer un compte*, remplissez vos informations :Nom, Prénom, Spécialité et définissez un mot de passe (min. 6 caractères).
Connexion :Entrez votre identifiant et votre mot de passe dans l'onglet  *Connexion* pour accéder à votre espace sécurisé.

Étape 2 : Le Tableau de Bord 
Une fois connecté, vous arrivez sur votre tableau de bord personnel.
*À droite :Gestion de cabinet : Vous disposez d'un calendrier interactif. Vous pouvez y consulter vos prochains rendez-vous, en ajouter de nouveaux  * Ajouter* ou supprimer des consultations passées.
*À gauche :Moteur de prédiction :C'est ici que se trouve le formulaire de saisie clinique.

Étape 3 : Saisie du Profil Patient
Dans le panneau de gauche, renseignez les données pré-opératoires du patient :
*Variables numériques :Ajustez l'âge, la masse corporelle, ainsi que les doses prévues de cellules souches CD34+ et de lymphocytes T CD3+.
*Variables catégorielles : Sélectionnez le type de maladie :AML, ALL, etc., l'antécédent de rechute, et les niveaux de compatibilité via les menus déroulants.
*Cliquez ensuite sur le bouton bleu **Analyser la prédiction**.

Étape 4 : Interprétation du Diagnostic IA
L'application bascule sur la page de résultats générée par le modèle d'Intelligence Artificielle.
*La prédiction :Un encadré visuel :Vert pour favorable, Rouge pour risque élevé,affiche le score de probabilité de survie à 1 an.
*L'Explicabilité SHAP : Un graphique détaillé vous explique le score. Les barres bleues représentent les paramètres qui ont favorisé la survie, tandis que les barres rouges soulignent les facteurs de risque spécifiques à ce patient.

Étape 5 : Traçabilité et Export PDF
Si le protocole simulé vous convient :
Cliquez sur le bouton **Télécharger le PDF** situé en haut à droite.
L'application génère instantanément un rapport médico-légal complet avec en-tête de l'hôpital, horodatage, profil patient, protocole retenu, et graphiques explicatifs.
Pour tester un nouveau dosage sur le même patient, cliquez sur **← Retour à l'analyse** et ajustez les variables.