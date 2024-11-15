
import streamlit as st
from sections.classification.classification import classification_page
from sections.nailsdetection.nails import nail_page
from sections.regression.regression import regression_page

# La première commande Streamlit dans le fichier
st.set_page_config(
    page_title="Playground ML",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Choix dans la barre latérale
type_data = st.sidebar.radio(
    "Choisissez votre type de playground",
    ["Regression", "Classification", "NailsDetection"]
)

# Chargement des pages en fonction du choix
if type_data == "Regression":
    regression_page()
elif type_data == "Classification":
    classification_page()
elif type_data == "NailsDetection":
    nail_page()
else:
    st.write("Choisissez une option")
