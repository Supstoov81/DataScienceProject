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
    page_title="Playground ML",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Message de débogage
st.write("Application démarrée avec succès!")

# Choix dans la barre latérale
type_data = st.sidebar.radio(
    "Choisissez votre type de playground",
    ["Regression", "Classification", "NailsDetection"]
)

# Chargement des pages en fonction du choix
try:
    if type_data == "Regression":
        regression_page()
    elif type_data == "Classification":
        classification_page()
    elif type_data == "NailsDetection":
        nail_page()
    else:
        st.write("Choisissez une option")
except Exception as e:
    st.error(f"Une erreur est survenue : {str(e)}")
    st.write("Détails de l'erreur :")
    st.exception(e)
