import pandas as pd
from unidecode import unidecode

def clean_data(file_path):
    try:
        # Lecture du fichier CSV
        df = pd.read_csv(file_path, sep=',', encoding='utf-8')
        print("Fichier chargé avec succès.")
        
        # Affichage des premières lignes du dataframe
        print(df.head())
        
        # Vérifier les valeurs manquantes dans chaque colonne
        missing_values = df.isna().sum()
        print("\nValeurs manquantes par colonne :")
        print(missing_values)

        # Afficher les colonnes avec des valeurs manquantes
        columns_with_missing_values = missing_values[missing_values > 0].index
        print("\nColonnes avec des valeurs manquantes :")
        print(columns_with_missing_values)

        # Renommer la colonne "Unnamed: 0" en "Index"
        df = df.rename(columns={'Unnamed: 0': 'Index'})
        
        # Supprimer les accents dans la colonne "target"
        df['target'] = df['target'].apply(lambda x: unidecode(x) if isinstance(x, str) else x)

        # Appliquer une fonction pour convertir toutes les valeurs de la colonne en majuscule
        df['target'] = df['target'].apply(lambda x: x.upper())
        
        # Filtrer en prenant les données de la colonne proline supérieures à 1400
        df_filtred = df[df["proline"] > 1400]
        
        # Sauvegarder le DataFrame nettoyé dans un nouveau fichier CSV
        df.to_csv("cleaned.csv", index=False)
        print("Fichier nettoyé sauvegardé sous 'cleaned.csv'.")

        return df  # Retourne le DataFrame nettoyé
    
    except Exception as e:
        print("Erreur lors du nettoyage des données:", e)
        return None  # Retourne None en cas d'erreur