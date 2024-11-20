import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from .dataCleaner import clean_data


def classification_page():
    st.header("Bienvenue")
    st.caption("Bienvenue dans la classification")

    # Chemin vers le fichier CSV
    file_path = r"C:\Users\mattb\Documents\projet\data\cleaned_Vin.csv"

    # Appel de la fonction de nettoyage des données
    df_cleaned = clean_data(file_path)

    if df_cleaned is not None:
        st.write("Aperçu des données : ")
        st.write(df_cleaned.head())

        # Préparation des données
        # Suppression de la colonne 'Index'
        if "Index" in df_cleaned.columns:
            df_cleaned = df_cleaned.drop(columns=["Index"])

        X = df_cleaned.drop(columns=["target"])
        y = df_cleaned["target"]

        # Sélection de features avec RandomForest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)

        # Importance des features
        feature_importances = pd.Series(rf.feature_importances_, index=X.columns)
        st.write("Importance des features : ")
        st.write(feature_importances)

        # Sélection des features les plus importantes
        threshold = 0.05  # Seuil à ajuster
        selected_features = feature_importances[feature_importances > threshold].index

        st.write("Features sélectionnées : ")
        st.write(selected_features)

        # Utiliser uniquement les features sélectionnées
        X_selected = X[selected_features]

        # Division des données en ensemble d'entraînement et de test avec stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.3, random_state=42, stratify=y
        )

        # Création du modèle LogisticRegression
        model = LogisticRegression(max_iter=1000)

        # Initialisation des listes pour stocker la précision
        train_accuracies = []
        test_accuracies = []

        # Vérification si les données d'entraînement contiennent plusieurs classes
        unique_classes_train = y_train.unique()
        if len(unique_classes_train) < 2:
            st.write(
                f"Erreur : L'échantillon d'entraînement contient une seule classe: {unique_classes_train[0]}."
            )
            return

        # Entraînement du modèle avec suivi de la précision à chaque itération
        for i in range(1, 101):  # Itération sur un nombre d'itérations maximum de 100
            # Sélectionner un sous-ensemble croissant des données d'entraînement
            subset_size = int(
                len(X_train) * (i / 100)
            )  # Augmenter progressivement la taille de l'échantillon

            # Si la taille du sous-ensemble est trop petite, on commence avec un sous-ensemble plus grand
            if subset_size < 10:
                subset_size = 10  # Vous pouvez ajuster ce seuil selon la taille de votre ensemble de données

            # S'assurer qu'on a au moins deux classes dans le sous-ensemble
            if len(y_train[:subset_size].unique()) < 2:
                st.write(
                    f"Erreur : L'échantillon d'entraînement pour l'itération {i} contient une seule classe."
                )
                break

            # Entraînement du modèle sur le sous-ensemble des données
            model.fit(
                X_train[:subset_size], y_train[:subset_size]
            )  # Entraîner sur un sous-ensemble des données

            # Prédictions
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            # Calcul de la précision
            train_acc = accuracy_score(y_train, train_pred)
            test_acc = accuracy_score(y_test, test_pred)

            # Ajout des résultats dans les listes
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)

            # Vérifier si des précisions ont été ajoutées
            if i % 10 == 0:  # Afficher chaque 10e itération
                st.write(
                    f"Iteration {i}, Précision entraînement : {train_acc:.2f}, Précision test : {test_acc:.2f}"
                )

        # Vérifier que les listes de précisions ne sont pas vides avant de tracer
        if len(train_accuracies) > 0 and len(test_accuracies) > 0:
            # Affichage des courbes de précision pour l'entraînement et le test
            st.write("Graphique de la précision d'entraînement et de test :")
            plt.figure(figsize=(10, 6))
            plt.plot(
                range(1, len(train_accuracies) + 1),
                train_accuracies,
                label="Précision entraînement",
            )
            plt.plot(
                range(1, len(test_accuracies) + 1),
                test_accuracies,
                label="Précision test",
                linestyle="--",
            )
            plt.xlabel("Nombre d'itérations")
            plt.ylabel("Précision")
            plt.title("Précision du modèle d'apprentissage")
            plt.legend()
            st.pyplot(plt)
        else:
            st.write(
                "Aucune précision n'a été enregistrée. Vérifiez vos itérations d'entraînement."
            )

        # Prédiction finale et évaluation
        # S'assurer que le modèle a été formé avant de prédire
        if hasattr(model, "coef_"):
            y_pred = model.predict(X_test)
            # Précision finale
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Précision du modèle sur les données de test : {accuracy:.2f}")

            # Matrice de confusion
            cm = confusion_matrix(y_test, y_pred)
            st.write("Matrice de confusion :")
            st.write(cm)

            # Rapport de classification
            report = classification_report(y_test, y_pred, output_dict=True)
            st.write("Rapport de classification :")
            st.write(report)
        else:
            st.write("Erreur : Le modèle n'a pas été correctement entraîné.")
    else:
        st.write("Erreur lors du nettoyage des données.")
