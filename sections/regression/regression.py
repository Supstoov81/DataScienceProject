import pandas as pd
import os
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, f_oneway
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def regression_page():
    st.header("Playground de Régression")
    st.caption("Sélection des variables explicatives basées sur la corrélation")

    # Charger le jeu de données
    csv_path = "data/diabetes_cleaned.csv"
    path = os.path.join(os.getcwd(), csv_path)
    # data_path = "C:/Users/lenovo/Desktop/FormationDigi/DataScienceProject/data/diabetes_cleaned.csv"
    try:
        data = pd.read_csv(path)
    except FileNotFoundError:
        st.error(f"Le fichier {path} n'a pas été trouvé.")
        return

    st.write("Aperçu des données :")
    st.write(data.head())

    # Vérification de la présence de colonnes
    if data.empty:
        st.error("Le jeu de données est vide.")
        return

    # Définir la colonne cible
    target_col = st.selectbox(
        "Sélectionnez la colonne cible pour la régression :",
        data.columns,
        key="target_col_1",
    )
    X = data.drop(columns=[target_col])
    y = data[target_col]

    # Table de corrélation
    st.subheader("Table de corrélation")
    corr_matrix = data.corr()
    st.write(corr_matrix)

    # Visualisation avec Seaborn
    st.subheader("Carte de corrélation (heatmap)")
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    st.pyplot(plt)
    plt.close()

    # Test de corrélation pour chaque colonne avec la cible
    st.subheader("Corrélation avec la cible")
    correlation_dict = {}

    for col in X.columns:
        try:
            corr, _ = pearsonr(X[col], y)
            correlation_dict[col] = corr
        except Exception as e:
            st.warning(
                f"Impossible de calculer la corrélation pour **{col}** : {str(e)}"
            )

    # Trier les résultats par ordre croissant de la corrélation
    sorted_corr = sorted(correlation_dict.items(), key=lambda item: item[1])

    # Afficher les résultats triés
    for col, corr in sorted_corr:
        st.write(f"Corrélation entre **{col}** et **{target_col}** : {corr:.2f}")

    # Test ANOVA pour les variables catégorielles
    st.subheader("Test ANOVA pour les variables catégorielles")
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns

    if categorical_cols.empty:
        st.write("Aucune variable catégorielle trouvée pour le test ANOVA.")
    else:
        for col in categorical_cols:
            try:
                # Séparer les données de la cible pour chaque catégorie
                groups = [y[X[col] == val] for val in X[col].dropna().unique()]

                # Appliquer le test ANOVA
                f_stat, p_val = f_oneway(*groups)

                # Afficher les résultats
                st.write(
                    f"ANOVA entre **{col}** et **{target_col}** : F-stat={f_stat:.2f}, p-val={p_val:.4f}"
                )

                # Interprétation des résultats
                if p_val < 0.05:
                    st.write(
                        f"Il y a une différence significative entre les groupes pour **{col}**."
                    )
                else:
                    st.write(
                        f"Aucune différence significative entre les groupes pour **{col}**."
                    )
            except Exception as e:
                st.warning(f"Erreur lors du test ANOVA pour **{col}** : {str(e)}")

    # Diviser le jeu de données en train et test
    test_size = (
        st.slider("Proportion de test (en %)", 10, 50, 30, key="test_size_slider") / 100
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Instanciation du modèle de régression
    model = LinearRegression()

    # Entraîner le modèle
    model.fit(X_train, y_train)

    # Prédictions
    y_pred = model.predict(X_test)

    # Afficher les résultats
    st.subheader("Évaluation du modèle")
    st.write("Erreur quadratique moyenne (MSE) :", mean_squared_error(y_test, y_pred))
    st.write("Score R2 :", r2_score(y_test, y_pred))

    # Afficher les coefficients du modèle
    st.write("Coefficients du modèle :", model.coef_)


# Appel de la fonction si nécessaire
if __name__ == "__main__":
    regression_page()
