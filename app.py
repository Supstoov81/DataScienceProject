import streamlit as st
from sections.classification.classification import classification_page
from sections.nailsdetection.nails import nail_page
from sections.regression.regression import regression_page

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Playground ML",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Style CSS personnalisé
st.markdown("""
    <style>
    .main-header {
        font-size: 36px;
        font-weight: bold;
        color: #ffffff;
        background-color: #4CAF50;
        text-align: center;
        padding: 10px;
        border-radius: 5px;
    }
    .sidebar .sidebar-content {
        background-color: #f4f4f4;
        border-radius: 10px;
        padding: 20px;
    }
    .section-header {
        font-size: 24px;
        font-weight: bold;
        margin: 20px 0;
        color: #333333;
    }
    .custom-radio > label {
        font-size: 18px;
        font-weight: bold;
        color: #4CAF50;
    }
    .stButton button {
        background-color: #4CAF50;
        color: #ffffff;
        font-size: 16px;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 20px;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Titre principal
st.markdown('<div class="main-header">Playground Machine Learning</div>', unsafe_allow_html=True)

# Menu dans la barre latérale
st.sidebar.markdown('<div class="section-header">Options disponibles</div>', unsafe_allow_html=True)
type_data = st.sidebar.radio(
    "Choisissez une option ci-dessous :", 
    ["Régression", "Classification", "Détection des ongles"], 
    key="playground_option"
)

# Contenu principal basé sur l'option sélectionnée
if type_data == "Régression":
    st.markdown('<div class="section-header">Analyse de régression</div>', unsafe_allow_html=True)
    regression_page()
elif type_data == "Classification":
    st.markdown('<div class="section-header">Modèles de classification</div>', unsafe_allow_html=True)
    classification_page()
elif type_data == "Détection des ongles":
    st.markdown('<div class="section-header">Détection des ongles</div>', unsafe_allow_html=True)
    nail_page()
else:
    st.write("Veuillez sélectionner une option.")

# Footer
st.markdown("""
    <hr style="margin: 20px 0;">
    <div style="text-align: center; color: #777777; font-size: 14px;">
        &copy; 2024 Playground ML. Tous droits réservés.
    </div>
""", unsafe_allow_html=True)
