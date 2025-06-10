import os
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import sys
import os

# Ajouter le répertoire parent au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from sections.classification.dataCleaner import clean_data

def classification_page():
    st.markdown('<h1 style="color: blue;">Bienvenue dans notre modèle de prédiction</h1>', unsafe_allow_html=True)
    st.caption("Classification des vins avec traitement des données et sélection de modèles")

    tabs = st.tabs(["Traitement", "Visualisation", "Entrainement et Evaluation"])

    # Onglet Traitement
    with tabs[0]:
        st.subheader("Traitement des données")

        # Chargement des données
        default_file = "cleaned.csv"  # Fichier par défaut dans le même dossier
        file_path = st.text_input("Chemin vers le fichier CSV :", default_file)
        
        # Utiliser le chemin relatif au fichier courant
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(current_dir, file_path)
        
        st.write(f"Tentative de chargement du fichier : {path}")
        
        if os.path.exists(path):
            df_cleaned = clean_data(path)
            if df_cleaned is not None:
                st.success("Fichier chargé avec succès!")
                st.write("Aperçu des données :")
                st.write(df_cleaned.head())

                # Gestion des valeurs manquantes
                missing_value_option = st.selectbox("Choisissez comment traiter les valeurs manquantes",
                                                    ["Supprimer les lignes", "Supprimer les colonnes",
                                                     "Remplir avec une valeur spécifique",
                                                     "Remplir avec la moyenne",
                                                     "Remplir avec la médiane"])

                if missing_value_option == "Remplir avec une valeur spécifique":
                    fill_value = st.text_input("Entrez la valeur pour remplacer les valeurs manquantes :")

                if missing_value_option == "Supprimer les lignes":
                    df_cleaned = df_cleaned.dropna()
                elif missing_value_option == "Supprimer les colonnes":
                    df_cleaned = df_cleaned.dropna(axis=1)
                elif missing_value_option == "Remplir avec une valeur spécifique" and fill_value:
                    df_cleaned = df_cleaned.fillna(fill_value)
                elif missing_value_option == "Remplir avec la moyenne":
                    numeric_cols = df_cleaned.select_dtypes(include=['number']).columns
                    df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].mean())
                elif missing_value_option == "Remplir avec la médiane":
                    numeric_cols = df_cleaned.select_dtypes(include=['number']).columns
                    df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].median())

                st.write("Données après traitement des valeurs manquantes :")
                st.write(df_cleaned.head())

                # Suppression de colonnes inutiles
                if 'Index' in df_cleaned.columns:
                    df_cleaned = df_cleaned.drop(columns=['Index'])

                X = df_cleaned.drop(columns=['target'])
                y = df_cleaned['target']

                # Sélection des features importantes
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                rf.fit(X, y)
                feature_importances = pd.Series(rf.feature_importances_, index=X.columns)

                st.write("Importance des features :")
                st.write(feature_importances)

                threshold = st.slider("Seuil pour sélectionner les features importantes", 0.0, 1.0, 0.05)
                selected_features = feature_importances[feature_importances > threshold].index

                st.write("Features sélectionnées :")
                st.write(selected_features)

                X_selected = X[selected_features]

    # Onglet Visualisation
    with tabs[1]:

        # Corrélation avec la cible
        if file_path and 'target' in df_cleaned:
            correlation_data = X_selected.copy()
            correlation_data['target'] = y

            # Encodage de la colonne target (avant la création de la matrice de corrélation)
            if df_cleaned['target'].dtype == 'object':
                label_encoder = LabelEncoder()
                df_cleaned['target'] = label_encoder.fit_transform(df_cleaned['target'])

            # Filtrer les colonnes sélectionnées et la cible
            correlation_data = df_cleaned[selected_features.tolist() + ['target']]

            # Calculer la matrice de corrélation
            correlation_matrix = correlation_data.corr()

            # Extraire uniquement les corrélations avec la cible
            correlation_with_target = correlation_matrix[['target']].drop('target')

            # Afficher les corrélations dans une heatmap
            st.write("Heatmap des corrélations entre les features sélectionnées et la colonne target :")
            plt.figure(figsize=(8, 6))
            sns.heatmap(correlation_with_target, annot=True, cmap="coolwarm", cbar=True)
            st.pyplot(plt)

    # Onglet Évaluation
    with tabs[2]:

        # Division des données
        if file_path:
            test_size = st.slider("Taille de l'ensemble de test", 0.1, 0.5, 0.3)
            X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=test_size, random_state=42, stratify=y)

            # Validation croisée
            cv_choice = st.selectbox("Type de validation croisée", ["KFold", "StratifiedKFold"])
            n_splits = st.slider("Nombre de plis pour la validation croisée", 2, 10, 5)

            if st.button("Démarrer l'entraînement"):
                models = {
                    "Logistic Regression": LogisticRegression(max_iter=1000),
                    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                    "SVM": SVC(random_state=42)
                }

                param_grids = {
                    "Logistic Regression": {"C": [0.1, 1, 10]},
                    "Random Forest": {"n_estimators": [50, 100, 200]},
                    "SVM": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}
                }

                model_results = []

                for model_name, model in models.items():
                    grid_search = GridSearchCV(model, param_grids[model_name], cv=n_splits, scoring='accuracy')
                    grid_search.fit(X_train, y_train)
                    best_params = grid_search.best_params_
                    best_score = grid_search.best_score_
                    best_index = grid_search.best_index_

                    mean_accuracy, std_accuracy = evaluate_model(grid_search.best_estimator_, X_selected, y, cv_choice, n_splits, model_name)
                    model_results.append({
                        "Modèle": model_name,
                        "Précision moyenne": mean_accuracy,
                        "Écart-type": std_accuracy,
                        "Meilleur paramètre": best_params,
                        "Meilleur score": best_score,
                        "Meilleur index": best_index
                    })

                if model_results:
                    st.write("Comparaison des performances des modèles :")
                    results_df = pd.DataFrame(model_results)
                    st.write(results_df)  # Afficher le tableau des résultats des modèles

                    best_model = max(model_results, key=lambda x: x["Précision moyenne"])
                    best_model_name = best_model["Modèle"]
                    st.write(f"Le meilleur modèle est : {best_model_name} avec une précision moyenne de "
                             f"{best_model['Précision moyenne']:.2f}.")

def evaluate_model(model, X, y, cv_choice, n_splits, model_name):
    if cv_choice == "KFold":
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    cv_results = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

    st.write(f"Résultats pour {model_name}:")
    for i, score in enumerate(cv_results, start=1):
        st.write(f"- **Score du pli {i} :** {score:.2f}")
    st.write(f"- **Précision moyenne :** {cv_results.mean():.2f}")
    st.write(f"- **Écart-type :** {cv_results.std():.2f}")

    model.fit(X, y)
    st.write(f"Entraînement final sur l'ensemble complet pour {model_name}.")

    return cv_results.mean(), cv_results.std()
