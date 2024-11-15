import pandas as pd
from unidecode import unidecode

# Chemin vers le fichier CSV
file_path = r"C:\Users\mattb\Documents\projet\data\vin.csv"
# Lecture du fichier CSV
df = pd.read_csv(file_path ,sep=',',encoding='utf-8')

# Affichage des premières lignes du dataframe
print("Fichier chargé avec succès.")
print(df.head())

# Vérifier les valeurs manquantes dans chaque colonne
missing_values = df.isna().sum()
print("\nValeurs manquantes par colonne :")
print(missing_values)
    
# Afficher les colonnes avec des valeurs manquantes
columns_with_missing_values = missing_values[missing_values > 0].index
print("\nColonnes avec des valeurs manquantes :")
print(columns_with_missing_values)

#Afficher le type de données des colonnes
print(df.dtypes)

# Renommer la colonne "Unnamed: 0" en "Index"
df = df.rename(columns={'Unnamed: 0': 'Index'})

# Afficher les nouvelles colonnes
print(df.columns)

# Supprimer les accents dans la colonne "target"
df['target'] = df['target'].apply(lambda x: unidecode(x) if isinstance(x, str) else x)

# Vérifier les valeurs après suppression des accents
print(df['target'].tail())

# Appliquer une fonction pour convertir toutes les valeurs de la colonne en majuscule
df['target'] = df['target'].apply(lambda x: x.upper())
print(df.head())

# Filtrer en prenant les données de la colonne proline supérieures à 1400
df_filtred = df[df["proline"] > 1400]
print("\ndf avec collonne proline supérieure à 1400 :")
print(df_filtred)

# Vérifier s'il y a des valeurs manquantes
if df.isna().sum().sum() > 0:
    print("Remplace les valeurs manquantes")
else:
    print("Aucune valeur manquante")

# Compter les occurrences uniques dans une colonne spécifique
print(df['proline'].value_counts())

# Boucler sur les lignes d'une colonne 'target' et afficher l'index et la valeur
for index, value in enumerate(df['target']):
    print(f"Index: {index}, Valeur: {value}")

# Sauvegarder le DataFrame nettoyé dans un nouveau fichier CSV
df.to_csv("cleaned.csv", index=False)

print("Fichier nettoyé sauvegardé sous 'cleaned.csv'.")