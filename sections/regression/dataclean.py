import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from joblib import dump, load

# Charger le dataset
st.title("Analyse interactive du Dataset Diabetes")
diabetes = pd.read_csv("C:/Users/lenovo/Desktop/FormationDigi/DataScienceProject/data/diabete.csv")

# Nettoyage des données
for col in diabetes.columns:
    diabetes[col].fillna(diabetes[col].mode()[0], inplace=True)

# Afficher le DataFrame
if st.checkbox("Afficher le DataFrame complet"):
    st.write(diabetes.head())

# Vérifier les valeurs manquantes dans le diabete
for col in diabetes.columns:
    # Compte le nombre de valeurs nulles dans chaque colonne
    print(col + ":", diabetes[diabetes[col].isnull()].count())

# Identifier les valeurs manquantes
print("Nombre de NaN par colonne :")
print(diabetes.isna().sum())

# Remplir les NaN avec la valeur la plus fréquente par colonne
for col in diabetes.columns:
    diabetes[col].fillna(diabetes[col].mode()[0], inplace=True)

# créer une fonction pour compter le nombre de valeurs 0 et leur pourcentage par colonne
def count_zeros(df, columns):
    for col in columns:
        num_zeros = (df[col] == 0).sum()  # Nombre de zéros dans la colonne
        total_rows = len(df[col])  # Nombre total de lignes dans la colonne
        percentage = (num_zeros / total_rows) * 100  # Pourcentage
        print("{} : {} ({:.2f}%)".format(col, num_zeros, percentage))

liste_cols = ['age', 'sex', 'bmi', 'bp', 's1']
count_zeros(diabetes, liste_cols)

print("Imputation par valeur fréquente terminée.")

# Distribution de l'âge
plt.figure(figsize=(10, 6))
plt.hist(diabetes['age'], bins=20, color='skyblue', edgecolor='black')
plt.title("Distribution de l'âge")
plt.xlabel("Âge")
plt.ylabel("Fréquence")
plt.show()

#  Boxplot de l'indice de masse corporelle (BMI)
plt.figure(figsize=(10, 6))
plt.boxplot(diabetes['bmi'], vert=False, patch_artist=True, boxprops=dict(facecolor='lightgreen'))
plt.title("Boxplot de l'indice de masse corporelle (BMI)")
plt.xlabel("BMI")
plt.show()

#  Répartition des sexes
plt.figure(figsize=(8, 6))
sex_counts = diabetes['sex'].value_counts()
plt.bar(sex_counts.index, sex_counts.values, color=['blue', 'pink'])
plt.title("Répartition des sexes")
plt.xlabel("Sexe (1 = Homme, 2 = Femme)")
plt.ylabel("Nombre")
plt.xticks(ticks=sex_counts.index, labels=["Homme", "Femme"])
plt.show()

#  Nuage de points entre BP et BMI
plt.figure(figsize=(10, 6))
plt.scatter(diabetes['bp'], diabetes['bmi'], alpha=0.7, c='orange', edgecolor='black')
plt.title("Relation entre la pression artérielle (BP) et le BMI")
plt.xlabel("Pression Artérielle (BP)")
plt.ylabel("Indice de Masse Corporelle (BMI)")
plt.show()

# Exploration des données
st.sidebar.header("Exploration des données")
exploration_option = st.sidebar.radio(
    "Choisissez une option d'exploration",
    ["Distribution de l'âge", "Boxplot de BMI", "Répartition des sexes", "Relation BP-BMI"]
)

if exploration_option == "Distribution de l'âge":
    st.subheader("Distribution de l'âge")
    fig, ax = plt.subplots()
    ax.hist(diabetes['age'], bins=20, color='skyblue', edgecolor='black')
    ax.set_title("Distribution de l'âge")
    ax.set_xlabel("Âge")
    ax.set_ylabel("Fréquence")
    st.pyplot(fig)

elif exploration_option == "Boxplot de BMI":
    st.subheader("Boxplot de BMI")
    fig, ax = plt.subplots()
    ax.boxplot(diabetes['bmi'], vert=False, patch_artist=True, boxprops=dict(facecolor='lightgreen'))
    ax.set_title("Boxplot de l'indice de masse corporelle (BMI)")
    ax.set_xlabel("BMI")
    st.pyplot(fig)

elif exploration_option == "Répartition des sexes":
    st.subheader("Répartition des sexes")
    sex_counts = diabetes['sex'].value_counts()
    fig, ax = plt.subplots()
    ax.bar(sex_counts.index, sex_counts.values, color=['blue', 'pink'])
    ax.set_title("Répartition des sexes")
    ax.set_xlabel("Sexe (1 = Homme, 2 = Femme)")
    ax.set_ylabel("Nombre")
    ax.set_xticks(sex_counts.index)
    ax.set_xticklabels(["Homme", "Femme"])
    st.pyplot(fig)

elif exploration_option == "Relation BP-BMI":
    st.subheader("Relation entre la pression artérielle (BP) et le BMI")
    fig, ax = plt.subplots()
    ax.scatter(diabetes['bp'], diabetes['bmi'], alpha=0.7, c='orange', edgecolor='black')
    ax.set_title("Relation entre BP et BMI")
    ax.set_xlabel("Pression Artérielle (BP)")
    ax.set_ylabel("Indice de Masse Corporelle (BMI)")
    st.pyplot(fig)
# Sauvegarder les données dans un nouveau fichier CSV
    diabetes_cleaned_file = "diabetes_cleaned.csv"
    diabetes.to_csv(diabetes_cleaned_file, index=False)

    print("Dataset nettoyé sauvegardé dans 'diabetes_cleaned.csv'")

    
