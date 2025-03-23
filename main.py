import pandas as pd

import utils.eda as eda
import utils.preprocessing as preprocessing
import utils.addestramento_word2vec_WordNet as w2v_wn
import utils.addestramento_word2vec as w2v
import utils.addestramento_supervisionato as add_sup
import random
import numpy as np

#stampe = True
stampe = False



# import del dataset nel programma
dataframe = pd.read_csv("dataset/netflix_titles_corrected.csv")


# EDA

# stampa nel terminale le informazioni generali del dataset
eda.general_informations(dataframe)

# stampa del grafico che mostra la differenza tra Film e Serie TV
if stampe:
    print("\n\n")
    # stampa del grafico che mostra la distribuzione di film e serie nel dataset
    eda.bar_product_subdivision(dataframe)
    # stampa del grafico che mostra la distribuzione di categoria età per serie
    eda.bar_plot_series(dataframe)
    # stampa del grafico che mostra la distribuzione di categoria età per film
    eda.bar_plot_movie(dataframe)
    # stampa del grafico che mostra la distribuzione di generi tra serie e film
    eda.plot_product_genres(dataframe)

# Fine EDA

#

#

# Preprocessing

# preprocessing per permutare i generi
dataframe = preprocessing.unify_genres(dataframe)

# Mostra i generi aggiornati
numero_esempi = 10
print(f"\n\nUnificazione dei generi eseguita. Stampa dei primi {numero_esempi} elementi del dataframe:")
print(dataframe[['title', 'listed_in']].head(numero_esempi))
if stampe:
    print("\n")
    eda.plot_unified_product_genres(dataframe)

# rinomina delle features
print("\n\nRinomina dellle features. dataset attuale:")
dataframe.info()
dataframe = preprocessing.rename_features(dataframe)
print("\n\nDataset con features rinominate:")
dataframe.info()

# Mapping delle categorie
all_ratings = dataframe['Categoria'].value_counts().index.to_list()
print(f"\n\nRinomina delle categorie di eta' presenti nel dataset.\nValori attuali: {sorted(all_ratings)}")
dataframe['Categoria'] = dataframe['Categoria'].apply(preprocessing.map_rating)
all_ratings = dataframe['Categoria'].value_counts().index.to_list()
print(f"\n\nValori presenti dopo la rinomina delle categorie di età: {sorted(all_ratings)}")
numero_esempi = 10
print(f"\nPrime {numero_esempi} righe del dataset con i ratings mappati:")
print(dataframe[['ID', 'Titolo', 'Descrizione', 'Categoria']].head(numero_esempi))

# modifica ai valori dei tipi
dataframe = preprocessing.permutazione_tipo(dataframe)

if stampe:
    print("\n\n")
    eda.bar_plot_categories(dataframe)

# gestione valori nulli
preprocessing.find_null_values(dataframe)
print("\nRiempimento valori...")
dataframe = preprocessing.manage_null_values(dataframe)
preprocessing.find_null_values(dataframe)

# rimozione dei duplicati
print("\n\nRimozione eventuali duplicati...\n")
dataframe = preprocessing.remove_duplicates(dataframe)

# dataframe per addestramento non supervisionato
non_sup_dataframe = dataframe.copy()

print("\n\nTrasformazione dei generi in vettori numerici per futuri addestramenti:")
dataframe = preprocessing.permutazione_generi_numerici(dataframe)

print("\nRisultato della permutazione:\n")
print(dataframe[["Generi", "Vettori_Generi"]].head(20))

print(f"\n\n\nFine fase di preprocessing, righe e colonne presenti:\n{dataframe.shape}")
dataframe.info()


# Fine Preprocessing

#

#

# Inizio Test Addestramento Supervisionato

add_sup.simulate_testing_sup_train(dataframe, stampe=stampe)

# Fine Test addestramento Supervisionato

#

#

# Inizio Test Addestramento non Supervisionato

no_sup_model = w2v.simulate_testing_non_sup_train(dataframe, stampe=stampe)

# Fine Test Addestramento non Supervisionato

#

#

# Inizio fase utente

while True:
    print("\n" + "=" * 50)
    print("📌 MENU PRINCIPALE")
    print("=" * 50)
    print("1️⃣  🔍 Cercare film per parole chiave (Apprendimento NON supervisionato)")
    print("2️⃣  🎭 Cercare film per preferenze personali (Apprendimento supervisionato)")
    print("3️⃣  ❌ Uscire")
    print("=" * 50)

    scelta = input("👉 Inserisci il numero dell'opzione desiderata: ").strip().lower()

    if scelta == "1":
        print("\n🔄 Avvio ricerca per parole chiave...")
        w2v.search_movies_by_user_input(non_sup_dataframe, no_sup_model, stampe=stampe)
    elif scelta == "2":
        print("\n🛠️ Avvio ricerca basata sulle preferenze personali...")
        add_sup.user_testing_sup_train(dataframe, stampe=stampe)
    elif scelta == "3":
        print("\n👋 Uscita dal programma... Grazie per aver usato il sistema di raccomandazione!")
        break
    else:
        print("\n⚠️ Scelta non valida! Inserisci un numero tra 1 e 3.")




