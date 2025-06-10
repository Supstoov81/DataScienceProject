import os
import sys
import streamlit as st

# Ajouter le répertoire courant au PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importer les pages
from sections.classification.classification import classification_page
from sections.nailsdetection.nails import nail_page
from sections.regression.regression import regression_page

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
