import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from .dataCleaner import clean_data

def classification_page():
    st.header("Bienvenue dans notre modèle de prédiction")
    st.caption("Bienvenue dans la classification des vins")

    # Chemin vers le fichier CSV
    file_path = st.text_input("Chemin vers le fichier CSV :", r"C:\Users\mattb\Documents\projet\data\cleaned_Vin.csv")

    if file_path:
        # Appel de la fonction de nettoyage des données
        df_cleaned = clean_data(file_path)
        
        if df_cleaned is not None:
            st.write("Aperçu des données : ")
            st.write(df_cleaned.head())

            # Options de traitement des valeurs manquantes
            missing_value_option = st.selectbox("Choisissez comment traiter les valeurs manquantes", 
                                                ["Supprimer les lignes", "Supprimer les colonnes", 
                                                 "Remplir avec une valeur spécifique", 
                                                 "Remplir avec la moyenne", 
                                                 "Remplir avec la médiane"])
                                                 
            if missing_value_option == "Remplir avec une valeur spécifique":
                fill_value = st.text_input("Entrez la valeur avec laquelle remplir les valeurs manquantes :")

            # Appliquer le traitement des valeurs manquantes
            if missing_value_option == "Supprimer les lignes":
                df_cleaned = df_cleaned.dropna()
            elif missing_value_option == "Supprimer les colonnes":
                df_cleaned = df_cleaned.dropna(axis=1)
            elif missing_value_option == "Remplir avec une valeur spécifique":
                if fill_value:
                    df_cleaned = df_cleaned.fillna(fill_value)
            elif missing_value_option == "Remplir avec la moyenne":
                numeric_cols = df_cleaned.select_dtypes(include=['number']).columns
                df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].mean())
            elif missing_value_option == "Remplir avec la médiane":
                numeric_cols = df_cleaned.select_dtypes(include=['number']).columns
                df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].median())

            st.write("Données après traitement des valeurs manquantes :")
            st.write(df_cleaned.head())

            # Suppression de la colonne 'Index'
            if 'Index' in df_cleaned.columns:
                df_cleaned = df_cleaned.drop(columns=['Index'])
            
            X = df_cleaned.drop(columns=['target'])
            y = df_cleaned['target']

            # Sélection de features avec RandomForest
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)
            
            # Importance des features
            feature_importances = pd.Series(rf.feature_importances_, index=X.columns)
            st.write("Importance des features : ")
            st.write(feature_importances)

            # Seuil ajustable pour la sélection des features importantes
            threshold = st.slider("Sélectionner le seuil pour les features importantes", 0.0, 1.0, 0.05)
            
            # Sélection des features les plus importantes
            selected_features = feature_importances[feature_importances > threshold].index
            
            st.write("Features sélectionnées : ")
            st.write(selected_features)

            # Utiliser uniquement les features sélectionnées
            X_selected = X[selected_features]
            
            # Division des données en ensemble d'entraînement et de test avec stratification
            test_size = st.slider("Sélectionner la taille de l'ensemble de test", 0.1, 0.5, 0.3)
            X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=test_size, random_state=42, stratify=y)

            # Sélection du modèle
            model_choice = st.selectbox("Choisissez le modèle de classification", ["Logistic Regression", "Random Forest"])
            
            # Sélection du type de validation croisée
            cv_choice = st.selectbox("Choisissez le type de validation croisée", ["KFold", "StratifiedKFold"])
            n_splits = st.slider("Nombre de plis pour la validation croisée", 2, 10, 5)

            if model_choice == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)
                if st.button('Lancer l\'entraînement'):
                    entrainer_et_afficher_resultats(model, X_selected, y, cv_choice, n_splits)
            elif model_choice == "Random Forest":
                nombre_arbre = st.selectbox("Choisissez le nombre d'arbres souhaités dans le modèle : ", options=[10, 50, 100, 200, 500])
                if st.button('Lancer l\'entraînement'):
                    model = RandomForestClassifier(n_estimators=nombre_arbre, random_state=42)
                    entrainer_et_afficher_resultats(model, X_selected, y, cv_choice, n_splits)

def entrainer_et_afficher_resultats(model, X, y, cv_choice, n_splits):
    if cv_choice == "KFold":
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    elif cv_choice == "StratifiedKFold":
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Cross-validation
    cv_results = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    st.write(f"Résultats de la validation croisée ({cv_choice}) :")
    st.write(cv_results)
    st.write(f"Précision moyenne : {cv_results.mean():.2f}")
    st.write(f"Écart-type de la précision : {cv_results.std():.2f}")

    # Entraînement final sur l'ensemble complet
    model.fit(X, y)
    
    # Prédictions sur l'ensemble de test
    st.write("Entraînement final sur l'ensemble complet des données effectué.")

    # Affichage des importances des features pour le RandomForest
    if isinstance(model, RandomForestClassifier):
        feature_importances = pd.Series(model.feature_importances_, index=X.columns)
        st.write("Importance des features (modèle final) : ")
        st.write(feature_importances)

    # Graphique des résultats de validation croisée
    st.write("Graphique des résultats de validation croisée :")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_splits + 1), cv_results, marker='o')
    plt.xlabel('Pli')
    plt.ylabel('Précision')
    plt.title(f'Précision de la validation croisée ({cv_choice})')
    plt.grid(True)
    st.pyplot(plt)
