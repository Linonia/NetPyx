import pandas as pd

import utils.eda as eda
import utils.preprocessing as preprocessing

#stampe = True
stampe = False

# import del dataset nel programma
dataframe = pd.read_csv("dataset/dataset_netflix.csv")

# stampa nel terminale le informazioni generali del dataset
eda.general_informations(dataframe)

# stampa del grafico che mostra la differenza tra Film e Serie TV
if stampe:
    eda.bar_suddivisione_prodotti(dataframe)
    eda.bar_plot_tv_show(dataframe)
    eda.bar_plot_movie(dataframe)

# stampa del grafico che mostra la distribuzione di generi tra serie e film
if stampe:
    eda.plot_generi_prodotti(dataframe)

# preprocessing per permutare i generi
dataframe = preprocessing.unifica_generi(dataframe)

# Mostra i generi aggiornati
print("\nGeneri unificati:")
print(dataframe[['title', 'listed_in']].head(10))
