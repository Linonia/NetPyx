import pandas as pd

import utils.eda as eda
import utils.preprocessing as preprocessing

#stampe = True
stampe = False

# import del dataset nel programma
dataframe = pd.read_csv("dataset/netflix_titles_corrected.csv")


# EDA

# stampa nel terminale le informazioni generali del dataset
eda.general_informations(dataframe)

# stampa del grafico che mostra la differenza tra Film e Serie TV
if stampe:
    eda.bar_product_subdivision(dataframe)
    eda.bar_plot_series(dataframe)
    eda.bar_plot_movie(dataframe)

# stampa del grafico che mostra la distribuzione di generi tra serie e film
if stampe:
    eda.plot_product_genres(dataframe)

# Fine EDA


# Preprocessing

# preprocessing per permutare i generi
dataframe = preprocessing.unify_genres(dataframe)

# Mostra i generi aggiornati
print("\nGeneri unificati:")
print(dataframe[['title', 'listed_in']].head(10))
if stampe:
    eda.plot_unified_product_genres(dataframe)

# rinomina delle features
print("\nDataset attuale:\n")
dataframe.info()
dataframe = preprocessing.rename_features(dataframe)
print("\nDataset con features rinominate:\n")
dataframe.info()

# Mapping delle categorie
all_ratings = dataframe['Categoria'].value_counts().index.to_list()
print("\nRatings presenti nel dataset:", sorted(all_ratings))
dataframe['Categoria'] = dataframe['Categoria'].apply(preprocessing.map_rating)
print("\nPrime 10 righe del dataset con i ratings mappati:")
print(dataframe[['ID', 'Titolo', 'Descrizione', 'Categoria']].head(10))

# gestione valori nulli
preprocessing.find_null_values(dataframe)
dataframe = preprocessing.manage_null_values(dataframe)
preprocessing.find_null_values(dataframe)

# rimozione dei duplicati
dataframe = preprocessing.remove_duplicates(dataframe)

dataframe = preprocessing.permutazione_generi_numerici(dataframe)

print(dataframe[["Generi", "Vettori_Generi"]].head(20))

print(f"\nFine fase di preprocessing, righe e colonne presenti:\n{dataframe.shape}")
dataframe.info()
# Fine Preprocessing
dataframe.to_csv("fine_preprocesso.csv", index=False)




