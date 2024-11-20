# Package basiques
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load


# lecture
import warnings
warnings.filterwarnings('ignore')

# Chargez le fichier CSV dans un dataframe Spark
diabetes=pd.read_csv("C:/Users/lenovo/Desktop/FormationDigi/37 - PROJET Data Science/projet/data/diabete.csv")

# Afficher les premières lignes du DataFrame
print(diabetes.head())
# Afficher des informations détaillées sur le DataFrame
print(diabetes.info())

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

    print("Imputation par valeur fréquente terminée.")

# créer une fonction pour compter le nombre de valeurs 0 et leur pourcentage par colonne
def count_zeros(df, columns):
    """
    Compte le nombre de zéros dans les colonnes spécifiées et affiche le pourcentage.

    :param df: pd.DataFrame, le DataFrame à analyser
    :param columns: list, liste des colonnes à analyser
    """
    for col in columns:
        num_zeros = (df[col] == 0).sum()  # Nombre de zéros dans la colonne
        total_rows = len(df[col])  # Nombre total de lignes dans la colonne
        percentage = (num_zeros / total_rows) * 100  # Pourcentage
        print("{} : {} ({:.2f}%)".format(col, num_zeros, percentage))

liste_cols = ['age', 'sex', 'bmi', 'bp', 's1']
count_zeros(diabetes, liste_cols)
# Afficher la valeur moyenne pour chaque colonne et faire le remplacement
for i in diabetes.columns[1:6]:
    mean_val = diabetes[i].mean()  # Calculer la moyenne de la colonne
    print("La valeur moyenne de la colonne {} est : {:.2f}".format(i, mean_val))
    
    # Mettre à jour les valeurs si elles sont égales à 0
    df=diabetes[i] = diabetes[i].replace(0, mean_val)

# Sauvegarder les données dans un nouveau fichier CSV
diabetes.to_csv("diabetes_cleaned.csv", index=False)

print("Dataset nettoyé sauvegardé dans 'diabetes_cleaned.csv'")
