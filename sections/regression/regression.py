

############# Regresssion avec les données diabete ############
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, f_oneway
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error, confusion_matrix
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from joblib import dump, load


def regression_page():
    st.header("Bienvenue ")
    st.caption("Analyse exploratoire, sélection des variables et évaluation d'un modèle de régression")

    # Charger le jeu de données
    data_path = "C:/Users/lenovo/Desktop/FormationDigi/DataScienceProject/data/diabetes_cleaned.csv"
    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        st.error(f"Le fichier {data_path} n'a pas été trouvé.")
        return

    st.write("Aperçu des données :")
    st.write(data.head())

    # Définir la colonne cible

    target = st.selectbox("Sélectionnez la colonne cible pour la régression :", data.columns)

    # Sélectionner les features (X) et la cible (y)
    X = data.drop(columns=[target])  # Features
    y = data[target]  # Target
    
    #  Splitting des données
    st.subheader("Séparation des données (Train/Test)")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    st.write(f"**Taille du Train Set :** {len(X_train)}")
    st.write(f"**Taille du Test Set :** {len(X_test)}")

    # Normalisation des features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    #  Entraînement et Évaluation
    st.subheader("Évaluation Comparée des Modèles")

    # Initialisation des modèles
    models = {
        "Régression Linéaire": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        "Support Vector Machine": SVR(kernel="rbf"),
    }

    results = []

    # Entraînement et évaluation pour chaque modèle
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        med_error = median_absolute_error(y_test, y_pred)

        results.append({"Modèle": name, "R²": r2, "MSE": mse, "Erreur Médiane": med_error})

    # Création d'un tableau récapitulatif
    results_df = pd.DataFrame(results)
    st.write("**Résultats Comparés des Modèles :**")
    st.dataframe(results_df)

    # Configuration du modèle dans la sidebar
    st.sidebar.header("Configuration du modèle")
    n_estimators = st.sidebar.slider("Nombre d'arbres dans Random Forest", 10, 200, 100)
    max_depth = st.sidebar.slider("Profondeur maximale de l'arbre", 2, 20, 10)

    # Normalisation des features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Division en train/test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Entraînement d'un modèle
    if st.sidebar.button("Entraîner le modèle"):
        st.subheader("Résultats de la Régression")
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Précision : {accuracy:.2f}")
        st.text("Rapport de Regression :")
        st.text(classification_report(y_test, y_pred))
        st.subheader("Résultats de Régression")
        reg_model = LinearRegression()
        reg_model.fit(X_train, y_train)
        y_pred_reg = reg_model.predict(X_test)

    # Normalisation des features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # Transformation des features

    # Table de corrélation
    st.subheader("Table de corrélation")
    corr_matrix = data.corr()
    st.write(corr_matrix)

    # Heatmap de corrélation
    st.subheader("Carte de corrélation (heatmap)")
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    st.pyplot(plt)
    plt.close()

    # Corrélation avec la cible
    st.subheader("Corrélation avec la cible")
    correlation_dict = {}
    for col in X.columns:
        try:
            corr, _ = pearsonr(X[col], y)
            correlation_dict[col] = corr
        except Exception as e:
            st.warning(f"Impossible de calculer la corrélation pour **{col}** : {str(e)}")
    sorted_corr = sorted(correlation_dict.items(), key=lambda item: item[1])
    for col, corr in sorted_corr:
        st.write(f"Corrélation entre **{col}** et **{target}** : {corr:.2f}")

    # Test ANOVA pour les variables catégorielles
    st.subheader("Test ANOVA pour les variables catégorielles")
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns
    for col in categorical_cols:
        # Séparer les données de la cible pour chaque catégorie
        groups = [y[X[col] == val] for val in X[col].unique()]
    
        # Appliquer le test ANOVA
        f_stat, p_val = f_oneway(*groups)
    
        # Afficher les résultats
        st.write(f"ANOVA entre **{col}** et **{target}** : F-stat={f_stat:.2f}, p-val={p_val:.4f}")

    # Validation croisée (5-fold)
    st.subheader("Validation croisée (5-fold)")
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    mse_scores = []

    for train_index, val_index in kfold.split(X_scaled):
        # Séparer les données en train et validation
        X_train_k, X_val_k = X_scaled[train_index], X_scaled[val_index]
        y_train_k, y_val_k = y.iloc[train_index], y.iloc[val_index]
    
        # Initialiser un modèle de régression à chaque itération
        reg_model = LinearRegression()
        reg_model.fit(X_train_k, y_train_k)
    
        # Faire des prédictions
        y_val_pred_k = reg_model.predict(X_val_k)
    
        # Calculer le MSE pour cette itération
        mse = mean_squared_error(y_val_k, y_val_pred_k)
        mse_scores.append(mse)

    # Calculer le MSE moyen sur les folds
    mean_mse = sum(mse_scores) / len(mse_scores)
    st.write(f"Erreur quadratique moyenne (MSE) moyenne sur les folds : {mean_mse:.2f}")