# Description du projet
Ce projet est une application interactive basée sur Streamlit qui regroupe plusieurs fonctionnalités d'apprentissage automatique. Elle propose des modules pour la régression, la classification, ainsi qu'une détection d'objets (ongles). Le projet utilise des ensembles de données prédéfinis pour illustrer les concepts d'analyse exploratoire, de sélection de variables, et d'évaluation de modèles prédictifs.

## Modules

### 1. Régression => Cheick NGOM
**But** : Prédire une variable cible continue à partir de données médicales liées au diabète 

**Caractéristiques principales :**
- Chargement et prétraitement des données.
- Analyse exploratoire (corrélation, heatmap, tests ANOVA).
- Comparaison de plusieurs modèles de régression :
  - Régression linéaire
  - Random Forest Regressor
  - Support Vector Regression (SVR)
- Validation croisée avec 5-fold.
- Résultats comparés des métriques : R², MSE, Erreur Médiane.
- Hyperparamètres ajustables via la barre latérale (nombre d'arbres, profondeur, etc.).

### 2. Classification => Mourad ATTBIB
**But** : Classifier des données d'échantillons de vin à l'aide de modèles de machine learning.

**Caractéristiques principales :**
- Prétraitement et nettoyage des données.
- Sélection des variables les plus importantes via un modèle Random Forest.
- Création d'un modèle de classification basé sur la régression logistique.
- Affichage des résultats :
  - Précision (Accuracy)
  - Matrice de confusion
  - Rapport de classification.
- Graphique montrant l'évolution de la précision en fonction des itérations d'entraînement.

### 3. Détection d'ongles => Kieran SWEETMAN
**But** : Détecter des ongles dans des images via un modèle préentraîné hébergé sur Roboflow.

**Caractéristiques principales :**
- Téléchargement multiple d'images par l'utilisateur.
- Détection d'ongles à l'aide d'une API Roboflow.
- Affichage des résultats avec les boîtes délimitant les ongles détectés sur les images.

## Technologies utilisées
- **Python**
- **Bibliothèques :**
  - Streamlit : Interface utilisateur interactive.
  - Pandas : Manipulation des données.
  - Matplotlib / Seaborn : Visualisations.
  - Scikit-learn : Modèles de machine learning et métriques d'évaluation.
  - Joblib : Sauvegarde et chargement des modèles.
  - Roboflow : API pour la détection d'objets.
- Jupyter Notebooks (optionnel) : Explorations supplémentaires.

## Installation

### Prérequis
- Python 3.8 ou supérieur.
- Bibliothèques Python nécessaires (énumérées dans `requirements.txt`).

### Étapes d'installation
1. Clonez le dépôt :  
   ```bash
   git clone https://github.com/Supstoov81/DataScienceProject

 2 .  Installez les dépendances

 3 .  pip install -r requirements.txt

 4 .   Lancez l'application Streamlit 

    streamlit run app.py


### Structure du projet 
DataScienceProject/
├── data/                          # Dossier contenant les ensembles de données
├── sections/                      # Modules principaux
│   ├── regression/                # Module régression
│   │   ├── regression.py
│   ├── classification/            # Module classification
│   │   ├── classification.py
│   ├── detection/                 # Module détection d'ongles
│       ├── detection.py
├── app.py                         # Point d'entrée principal de l'application
├── requirements.txt               # Liste des dépendances Python
├── README.md                      # Documentation


