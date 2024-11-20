import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
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

            # Sélection du type de validation croisée
            cv_choice = st.selectbox("Choisissez le type de validation croisée", ["KFold", "StratifiedKFold"])
            n_splits = st.slider("Nombre de plis pour la validation croisée", 2, 10, 5)

            # Sélectionner le nombre d'arbres pour RandomForest
            n_estimators = st.slider("Sélectionner le nombre d'arbres pour Random Forest", 10, 500, 100)

            # Bouton pour démarrer l'entraînement
            if st.button("Démarrer l'entraînement"):
                # Initialisation des modèles et grilles de paramètres
                models = {
                    "Logistic Regression": (LogisticRegression(max_iter=1000), {'C': [0.1, 1, 10]}),
                    "Random Forest": (RandomForestClassifier(random_state=42, n_estimators=n_estimators), {'max_depth': [None, 10, 20, 30]}),
                    "SVM": (SVC(random_state=42), {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']})
                }

                model_results = {}

                # Evaluer chaque modèle avec la validation croisée et GridSearchCV
                for model_name, (model, param_grid) in models.items():
                    mean_accuracy, std_accuracy, best_params = entrainer_et_afficher_resultats(model, X_selected, y, cv_choice, n_splits, model_name, param_grid)
                    model_results[model_name] = {"mean": mean_accuracy, "std": std_accuracy, "best_params": best_params}

                # Comparaison des modèles
                if model_results:
                    st.write("Comparaison des performances des modèles :")
                    model_comparison_df = pd.DataFrame(model_results).T
                    model_comparison_df.columns = ['Précision moyenne', 'Écart-type', 'Meilleurs paramètres']
                    st.write(model_comparison_df)

                    # Affichage du meilleur modèle
                    best_model_name = max(model_results, key=lambda x: model_results[x]["mean"])
                    st.write(f"Le meilleur modèle est : {best_model_name} avec une précision moyenne de {model_results[best_model_name]['mean']:.2f}.")
                    st.write(f"Meilleurs paramètres pour {best_model_name} : {model_results[best_model_name]['best_params']}")

def entrainer_et_afficher_resultats(model, X, y, cv_choice, n_splits, model_name, param_grid):
    if cv_choice == "KFold":
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    elif cv_choice == "StratifiedKFold":
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # GridSearchCV
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)
    
    # Meilleurs paramètres et score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # Affichage des résultats avec le nom du modèle
    st.write(f"Résultats de la validation croisée ({cv_choice}) pour le modèle {model_name} :")
    st.write(f"Meilleurs paramètres : {best_params}")
    st.write(f"Meilleur score : {best_score:.2f}")

    # Entraînement final sur l'ensemble complet avec les meilleurs paramètres
    best_model = grid_search.best_estimator_
    best_model.fit(X, y)
    
    st.write(f"Entraînement final sur l'ensemble complet des données pour le modèle {model_name} effectué.")

    return best_score, grid_search.cv_results_['std_test_score'][grid_search.best_index_], best_params
