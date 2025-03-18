import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from gensim.models import Word2Vec

STOP_WORDS = {"movies", "movie", "tv", "shows", "show", "series"}


def preprocess_genres(genres):
    """ Rimuove stopword e converte in minuscolo """
    processed = []
    for genre in genres:
        words = genre.lower().split()
        filtered_words = [word for word in words if word not in STOP_WORDS]
        processed.append(" ".join(filtered_words))
    return processed


def find_longest_common_phrase(strings):
    """ Trova la sequenza di parole più lunga in comune tra le stringhe """
    if not strings:
        return "Unknown"

    words_lists = [s.split() for s in strings if s]
    if not words_lists:
        return "Unknown"

    common_phrases = set(words_lists[0])
    for words in words_lists[1:]:
        common_phrases.intersection_update(words)

    if not common_phrases:
        return "Unknown"

    longest_common_phrase = max(common_phrases, key=len, default="Unknown")
    return longest_common_phrase.title()


def cluster_genres(genres, n_clusters=20):
    """ Clustering dei generi usando KMeans """
    processed_genres = preprocess_genres(genres)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(processed_genres)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    clustered = {}
    for genre, label in zip(genres, labels):
        clustered.setdefault(label, []).append(genre)

    return clustered


def assign_cluster_names(clusters):
    """ Assegna un nome rappresentativo a ogni cluster """
    cluster_names = {}
    for cluster_id, genres in clusters.items():
        processed_genres = preprocess_genres(genres)
        valid_genres = [g for g in processed_genres if g]
        if "" in processed_genres:
            cluster_names[cluster_id] = "Unknown"
        else:
            cluster_names[cluster_id] = find_longest_common_phrase(valid_genres)
    return cluster_names


def build_genre_mapping(genres):
    """ Costruisce la mappa dei generi originali verso i nuovi generi """
    clusters = cluster_genres(genres, n_clusters=30)
    cluster_names = assign_cluster_names(clusters)
    return {genre: cluster_names[cluster] for cluster, genres in clusters.items() for genre in genres}


def unify_genres(df):
    """ Applica la trasformazione dei generi al dataset """
    if 'listed_in' not in df.columns:
        raise ValueError("Il dataframe non contiene la colonna 'listed_in'")

    generi_unici = set()
    df['listed_in'].dropna().str.split(', ').apply(generi_unici.update)
    genre_mapping = build_genre_mapping(list(generi_unici))

    def map_new_genres(genres):
        mapped_genres = {genre_mapping.get(g, g) for g in genres.split(', ')}
        return ', '.join(mapped_genres)

    df['listed_in'] = df['listed_in'].apply(lambda x: map_new_genres(x) if pd.notna(x) else x)
    return df


def rename_features(dataframe: pd.DataFrame):
    """
    Rinomina le colonne del files per migliorarne la leggibilità.

    :param dataframe: DataFrame di Pandas contenente il files da modificare.
    :return: DataFrame con le colonne rinominate.
    """
    nuovi_nomi_features = {
        'show_id': 'ID',
        'type': 'Tipo',
        'title': 'Titolo',
        'director': 'Regista',
        'description': 'Descrizione',
        'cast': 'Cast',
        'country': 'Nazione',
        'date_added': 'Data_Aggiunta',
        'release_year': 'Data_Uscita',
        'rating': 'Categoria',
        'duration': 'Durata',
        'listed_in': 'Generi'
    }

    dataframe.rename(columns=nuovi_nomi_features, inplace=True)
    return dataframe


# Funzione per il mapping dei ratings in una categoria corrispondente.
# Considerate 7 categorie differenti: Kids, Children, Family, Teens, Mature Teens, Adults, Unrated
def map_rating(rating):
    # Dizionario che mappa i ratings alle categorie
    categories = {
        'Kids': ['G', 'TV-G', 'TV-Y'],  # Bambini piccoli
        'Children': ['TV-Y7', 'TV-Y7-FV'],  # Bambini sopra i 7 anni
        'Family': ['PG', 'TV-PG'],  # Contenuti adatti a tutta la famiglia
        'Teens': ['PG-13', 'TV-14'],  # Adolescenti
        'Mature Teens': ['R'],  # Adolescenti Maggiorenni
        'Adults': ['NC-17', 'TV-MA'],  # Contenuti per adulti
        'Unrated': ['NR', 'Not Rated', 'UR']  # Non classificati
    }

    # Ricerca del rating nella categoria
    for category, ratings in categories.items():
        if rating in ratings:
            return category.capitalize()


def find_null_values(dataframe: pd.DataFrame):
    """
    Stampa la quantità di valori mancanti per colonna nel files.

    :param dataframe: DataFrame di Pandas contenente il files da analizzare.
    :return: None
    """
    print(f"\nQuantità di valori mancanti per colonna nel files:\n{dataframe.isnull().sum()}")


def manage_null_values(dataframe: pd.DataFrame):
    """
    Gestisce i valori mancanti nel files riempiendo i campi nulli con valori predefiniti.

    :param dataframe: DataFrame di Pandas contenente il files da modificare.
    :return: DataFrame con i valori nulli gestiti.
    """
    dataframe = dataframe.copy()  # senza questa riga risultava in errore
    dataframe['Categoria'] = dataframe['Categoria'].fillna('Unrated')
    dataframe['Regista'] = dataframe['Regista'].fillna('Sconosciuto')
    dataframe['Nazione'] = dataframe['Nazione'].fillna('Sconosciuto')
    dataframe['Data_Aggiunta'] = dataframe['Data_Aggiunta'].fillna('Sconosciuto')
    dataframe['Cast'] = dataframe['Cast'].fillna('Sconosciuto')
    return dataframe


def remove_duplicates(dataframe: pd.DataFrame):
    """
    Rimuove i duplicati presenti nel files ignorando la colonna 'ID',
    senza alterare i tipi di dato nelle altre colonne.

    :param dataframe: DataFrame di Pandas contenente il files da modificare.
    :return: DataFrame senza duplicati.
    """
    df_temp = dataframe.copy()

    # Converti temporaneamente colonne problematiche in stringa per individuare i duplicati
    for col in df_temp.columns:
        if df_temp[col].apply(lambda x: isinstance(x, (list, dict, set, pd.Series, pd.DataFrame))).any():
            df_temp[col] = df_temp[col].astype(str)

    # Trova gli ID delle righe duplicate (ignorando la colonna 'ID')
    duplicati = df_temp[df_temp.duplicated(subset=[col for col in df_temp.columns if col != 'ID'], keep='first')]['ID']

    # Elimina le righe duplicate dal dataframe originale
    dataframe = dataframe[~dataframe['ID'].isin(duplicati)].copy()

    return dataframe


def permutazione_generi_numerici(df):
    """
    Converte i generi dei Film e delle Serie TV in vettori numerici utilizzando Word2Vec.

    :param dataframe: DataFrame di Pandas contenente il files da modificare.
    :return: DataFrame con i generi convertiti in vettori numerici.
    """
    # Convertire i generi in liste
    df['Generi'] = df['Generi'].apply(lambda x: x.split(', ') if isinstance(x, str) else [])

    # Creare una lista complessiva di generi
    generi_totali = df['Generi'].tolist()

    # Creare il modello Word2Vec
    modello = Word2Vec(sentences=generi_totali, vector_size=100, window=5, min_count=1, workers=4, sg=0)
    vocabolario = list(modello.wv.key_to_index.keys())
    print(f"\nGeneri presenti nel modello di Word2Vec:\n{vocabolario}")

    # Creare i vettori dei generi
    def genera_vettore(generi):
        vettori = [modello.wv[genere] for genere in generi if genere in modello.wv]
        return np.mean(vettori, axis=0) if vettori else np.zeros(modello.vector_size)

    df['Vettori_Generi'] = df['Generi'].apply(genera_vettore)

    return df


