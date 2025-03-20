import pandas as pd

import utils.eda as eda
import utils.preprocessing as preprocessing
import utils.addestramento_non_supervisionato as add_non_sup
import utils.addestramento_word2vec_WordNet as w2v_wn
import utils.addestramento_word2vec as w2v


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

print("\n\nTrasformazione dei generi in vettori numerici per futuri addestramenti:")
dataframe = preprocessing.permutazione_generi_numerici(dataframe)

print("\nRisultato della permutazione:\n")
print(dataframe[["Generi", "Vettori_Generi"]].head(20))

print(f"\n\n\nFine fase di preprocessing, righe e colonne presenti:\n{dataframe.shape}")
dataframe.info()
# Fine Preprocessing
# dataframe.to_csv("fine_preprocesso.csv", index=False)


print("\n\n\n\nProva senza Wordnet")

word_searching = [["scary", "paranormal"], ["challenge", "death"], ["anime", "village"]]

# Prova senza WordNet
model = w2v.train_word2vec(dataframe)
for words in word_searching:
    print(f"\n\nRisultati per {words} senza WordNet:\n")
    w2v.get_similar_words(model, words, topn=3)
    suggestions = w2v.get_similar_movies(dataframe, model, words, topn=5)
    print(suggestions[['Titolo', 'Generi']].head(5))

print("\n\n\n\nProva con Wordnet")

# Prova con WordNet
model_wn = w2v_wn.train_word2vec(dataframe)
for words in word_searching:
    print(f"\n\nRisultati per {words} con WordNet:\n")
    w2v_wn.get_similar_words(model_wn, words, topn=3)
    suggestions = w2v_wn.get_similar_movies(dataframe, model_wn, words, topn=5)
    print(suggestions[['Titolo', 'Generi']].head(5))

