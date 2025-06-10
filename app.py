import os
import sys
import streamlit as st

# Ajouter le répertoire courant au PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importer les pages
from sections.classification.classification import classification_page
from sections.nailsdetection.nails import nail_page
from sections.regression.regression import regression_page

def home_page():
    st.title("Bienvenue dans notre Application de Data Science")
    
    st.markdown("""
    ### 🚀 À propos de l'application
    
    Cette application vous permet d'explorer et d'analyser des données à travers différentes fonctionnalités :
    
    - **Classification** : Analyse et prédiction de données catégorielles
    - **Régression** : Analyse et prédiction de données continues
    - **Détection d'ongles** : Analyse d'images pour la détection d'ongles
    
    ### 📊 Fonctionnalités principales
    
    1. **Traitement des données**
       - Nettoyage et préparation des données
       - Gestion des valeurs manquantes
       - Sélection des features importantes
    
    2. **Visualisation**
       - Graphiques interactifs
       - Analyse des corrélations
       - Exploration des données
    
    3. **Modélisation**
       - Entraînement de différents modèles
       - Évaluation des performances
       - Comparaison des résultats
    
    ### 🎯 Comment commencer ?
    
    1. Utilisez le menu latéral pour naviguer entre les différentes sections
    2. Choisissez le type d'analyse qui vous intéresse
    3. Suivez les étapes guidées pour chaque fonctionnalité
    
    ### 📈 Exemple de workflow
    
    Pour une analyse de classification :
    1. Chargez vos données
    2. Prétraitez les données
    3. Visualisez les corrélations
    4. Entraînez et évaluez les modèles
    """)

# Configuration de la page
st.set_page_config(
    page_title="Projet de Data Science",
    page_icon="📊",
    layout="wide"
)

# Message de débogage
st.write("Application démarrée avec succès!")

# Barre latérale pour la navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choisissez une page", ["Accueil", "Classification"])

if page == "Accueil":
    home_page()
elif page == "Classification":
    # Sous-menu pour les onglets de classification
    classification_tab = st.sidebar.radio(
        "Sélectionnez une section",
        ["Traitement des données", "Visualisation", "Entraînement et Evaluation"]
    )
    classification_page(classification_tab)

if __name__ == "__main__":
    main()
