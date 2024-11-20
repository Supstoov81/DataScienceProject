import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from .dataCleaner import clean_data  # Assurez-vous que la fonction clean_data est bien définie dans le fichier 'dataCleaner.py'


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

            # Suppression de la colonne 'Index' si elle existe
            if 'Index' in df_cleaned.columns:
                df_cleaned = df_cleaned.drop(columns=['Index'])

            # Afficher les colonnes pour vérifier la présence de 'target'
            st.write("Colonnes disponibles :", df_cleaned.columns)

            # Vérification de la présence de la colonne 'target'
            if 'target' not in df_cleaned.columns:
                st.error("Erreur : la colonne 'target' est absente du DataFrame. Assurez-vous que les données sont correctement formatées.")
                return

            X = df_cleaned.drop(columns=['target'])
            y = df_cleaned['target']

            # Importance des features avec RandomForest
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)

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

            if not selected_features.empty:
                st.write("Tableau de corrélation avec la variable cible :")
                
                # Ajouter la cible pour calculer la corrélation
                correlation_data = X_selected.copy()
                correlation_data["target"] = y

                # Forcer la conversion de la colonne target en numérique si nécessaire
                if not pd.api.types.is_numeric_dtype(correlation_data["target"]):
                    correlation_data["target"] = pd.factorize(correlation_data["target"])[0]

                # Calcul de la matrice de corrélation
                correlation_matrix = correlation_data.corr()

                if "target" in correlation_matrix.columns:
                    correlation_with_target = correlation_matrix["target"].sort_values(ascending=False)
                    st.write(correlation_with_target)

                    # Afficher une heatmap
                    st.write("Heatmap des corrélations :")
                    plt.figure(figsize=(10, 6))
                    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
                    st.pyplot(plt)
                else:
                    st.error("Erreur : 'target' n'est pas présent dans la matrice de corrélation.")
            else:
                st.warning("Aucune feature importante sélectionnée.")

            # Division des données en ensemble d'entraînement et de test
            test_size = st.slider("Sélectionner la taille de l'ensemble de test", 0.1, 0.5, 0.3)
            X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=test_size, random_state=42, stratify=y)

            # Sélection du type de validation croisée
            cv_choice = st.selectbox("Choisissez le type de validation croisée", ["KFold", "StratifiedKFold"])
            n_splits = st.slider("Nombre de plis pour la validation croisée", 2, 10, 5)

            # Bouton pour démarrer l'entraînement
            if st.button("Démarrer l'entraînement"):
                # Initialisation des modèles
                models = {
                    "Logistic Regression": LogisticRegression(max_iter=1000),
                    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                    "SVM": SVC(random_state=42)
                }

                model_results = {}

                # Evaluer chaque modèle avec la validation croisée
                for model_name, model in models.items():
                    mean_accuracy, std_accuracy = entrainer_et_afficher_resultats(model, X_selected, y, cv_choice, n_splits, model_name)
                    model_results[model_name] = {"mean": mean_accuracy, "std": std_accuracy}

                # Comparaison des modèles
                if model_results:
                    st.write("Comparaison des performances des modèles :")
                    model_comparison_df = pd.DataFrame(model_results).T
                    model_comparison_df.columns = ['Précision moyenne', 'Écart-type']
                    st.write(model_comparison_df)

                    # Affichage du meilleur modèle
                    best_model_name = max(model_results, key=lambda x: model_results[x]["mean"])
                    st.write(f"Le meilleur modèle est : {best_model_name} avec une précision moyenne de {model_results[best_model_name]['mean']:.2f}.")


def entrainer_et_afficher_resultats(model, X, y, cv_choice, n_splits, model_name):
    if cv_choice == "KFold":
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    elif cv_choice == "StratifiedKFold":
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Cross-validation
    cv_results = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

    # Affichage des résultats
    st.write(f"Résultats de la validation croisée ({cv_choice}) pour le modèle {model_name} :")
    st.write(cv_results)
    st.write(f"Précision moyenne : {cv_results.mean():.2f}")
    st.write(f"Écart-type de la précision : {cv_results.std():.2f}")

    # Entraînement final sur l'ensemble complet
    model.fit(X, y)
    st.write(f"Entraînement final sur l'ensemble complet des données pour le modèle {model_name} effectué.")

    return cv_results.mean(), cv_results.std()
