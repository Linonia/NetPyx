import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from gensim.models import Word2Vec

STOP_WORDS = {"movies", "movie", "tv", "shows", "show", "series"}


def preprocess_genres(genres):
    """
    Pulisce i generi rimuovendo stopword e convertendo i testi in minuscolo.

    :param genres: Lista di stringhe contenenti i generi.
    :return: Lista di generi processati.
    """
    processed = []

    # Iterazione sui generi per la pulizia del testo
    for genre in genres:
        words = genre.lower().split()

        # Rimozione delle stopword
        filtered_words = [word for word in words if word not in STOP_WORDS]

        # Ricostruzione della stringa pulita
        processed.append(" ".join(filtered_words))

    return processed


def find_longest_common_phrase(strings):
    """
    Trova la sequenza di parole più lunga in comune tra le stringhe fornite.

    :param strings: Lista di stringhe da confrontare.
    :return: Frase più lunga in comune tra le stringhe, oppure 'Unknown' se non esiste.
    """
    if not strings:
        return "Unknown"

    # Suddivisione delle stringhe in liste di parole
    words_lists = [s.split() for s in strings if s]

    if not words_lists:
        return "Unknown"

    # Trova le parole comuni tra tutte le stringhe
    common_phrases = set(words_lists[0])
    for words in words_lists[1:]:
        common_phrases.intersection_update(words)

    if not common_phrases:
        return "Unknown"

    # Trova la parola più lunga tra quelle in comune
    longest_common_phrase = max(common_phrases, key=len, default="Unknown")

    return longest_common_phrase.title()


def cluster_genres(genres, n_clusters=30):
    """
    Raggruppa i generi utilizzando l'algoritmo KMeans.

    :param genres: Lista di stringhe contenenti i generi.
    :param n_clusters: Numero di cluster da creare (default: 30).
    :return: Dizionario con i cluster e i relativi generi associati.
    """
    # Pre-elaborazione dei generi
    processed_genres = preprocess_genres(genres)

    # Creazione della matrice TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(processed_genres)

    # Addestramento del modello KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    # Creazione del dizionario per raccogliere i generi raggruppati
    clustered = {}
    for genre, label in zip(genres, labels):
        clustered.setdefault(label, []).append(genre)

    return clustered


def assign_cluster_names(clusters):
    """
    Assegna un nome rappresentativo a ogni cluster di generi.

    :param clusters: Dizionario con cluster e generi associati.
    :return: Dizionario con i nomi assegnati ai cluster.
    """
    cluster_names = {}

    # Iterazione sui cluster per assegnare un nome rappresentativo
    for cluster_id, genres in clusters.items():
        processed_genres = preprocess_genres(genres)

        # Filtra i generi validi eliminando stringhe vuote
        valid_genres = [g for g in processed_genres if g]

        # Assegna "Unknown" se ci sono generi vuoti, altrimenti trova il nome più rappresentativo
        if "" in processed_genres:
            cluster_names[cluster_id] = "Unknown"
        else:
            cluster_names[cluster_id] = find_longest_common_phrase(valid_genres)

    return cluster_names


def build_genre_mapping(genres):
    """
    Costruisce la mappatura dei generi originali verso i nuovi generi raggruppati.

    :param genres: Lista di stringhe contenenti i generi originali.
    :return: Dizionario che mappa ogni genere originale al nome del cluster assegnato.
    """
    # Clustering dei generi
    clusters = cluster_genres(genres)

    # Assegnazione dei nomi ai cluster
    cluster_names = assign_cluster_names(clusters)

    # Creazione della mappatura dai generi originali ai nuovi nomi di cluster
    return {genre: cluster_names[cluster] for cluster, genres in clusters.items() for genre in genres}


def unify_genres(dataframe):
    """
    Trasforma i generi del dataset raggruppandoli in categorie unificate.

    :param dataframe: DataFrame contenente la colonna 'listed_in' con i generi originali.
    :return: DataFrame con i generi aggiornati.
    """
    # Verifica la presenza della colonna 'listed_in'
    if 'listed_in' not in dataframe.columns:
        raise ValueError("Il dataframe non contiene la colonna 'listed_in'")

    # Estrazione dei generi unici presenti nel dataset
    generi_unici = set()
    dataframe['listed_in'].dropna().str.split(', ').apply(generi_unici.update)

    # Creazione della mappatura generi originali → generi unificati
    genre_mapping = build_genre_mapping(list(generi_unici))

    # Funzione interna per applicare la nuova categorizzazione ai generi
    def map_new_genres(genres):
        mapped_genres = {genre_mapping.get(g, g) for g in genres.split(', ')}
        return ', '.join(mapped_genres)

    # Applicazione della trasformazione alla colonna 'listed_in'
    dataframe['listed_in'] = dataframe['listed_in'].apply(lambda x: map_new_genres(x) if pd.notna(x) else x)

    return dataframe


def rename_features(dataframe):
    """
    Rinomina le colonne del DataFrame per migliorarne la leggibilità.

    :param dataframe: DataFrame contenente i dati originali.
    :return: DataFrame con le colonne rinominate.
    """
    # Dizionario di mappatura per la rinomina delle colonne
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

    # Applicazione della rinomina
    dataframe.rename(columns=nuovi_nomi_features, inplace=True)

    return dataframe


def map_rating(rating):
    """
    Mappa il rating in una categoria generica di età.

    :param rating: Valutazione originale del contenuto.
    :return: Categoria di età corrispondente.
    """
    # Dizionario di mappatura delle categorie di età
    categories = {
        'Kids': ['G', 'TV-G', 'TV-Y'],  # Bambini piccoli
        'Children': ['TV-Y7', 'TV-Y7-FV'],  # Bambini sopra i 7 anni
        'Family': ['PG', 'TV-PG'],  # Contenuti adatti a tutta la famiglia
        'Teens': ['PG-13', 'TV-14'],  # Adolescenti
        'Mature Teens': ['R'],  # Adolescenti Maggiorenni
        'Adults': ['NC-17', 'TV-MA'],  # Contenuti per adulti
        'Unrated': ['NR', 'Not Rated', 'UR']  # Non classificati
    }

    # Ricerca della categoria corrispondente al rating
    for category, ratings in categories.items():
        if rating in ratings:
            return category

    return "Unknown"


def find_null_values(dataframe: pd.DataFrame):
    """
    Stampa il numero di valori mancanti per colonna nel dataset.

    :param dataframe: DataFrame di Pandas da analizzare.
    :return:
    """
    # Calcola e stampa la quantità di valori nulli per colonna
    print(f"Quantità di valori mancanti per colonna nel dataset:\n{dataframe.isnull().sum()}")


def manage_null_values(dataframe: pd.DataFrame):
    """
    Gestisce i valori mancanti nel dataset riempiendo i campi nulli con valori predefiniti.

    :param dataframe: DataFrame di Pandas da modificare.
    :return: DataFrame con i valori nulli gestiti.
    """
    dataframe = dataframe.copy()  # Evita modifiche dirette al DataFrame originale

    # Riempie i valori nulli con valori predefiniti
    dataframe['Categoria'] = dataframe['Categoria'].fillna('Unrated')
    dataframe['Regista'] = dataframe['Regista'].fillna('Sconosciuto')
    dataframe['Nazione'] = dataframe['Nazione'].fillna('Sconosciuto')
    dataframe['Data_Aggiunta'] = dataframe['Data_Aggiunta'].fillna('Sconosciuto')
    dataframe['Cast'] = dataframe['Cast'].fillna('Sconosciuto')

    return dataframe


def remove_duplicates(dataframe: pd.DataFrame):
    """
    Rimuove i duplicati presenti nel dataset ignorando la colonna 'ID',
    senza alterare i tipi di dato nelle altre colonne.

    :param dataframe: DataFrame di Pandas da modificare.
    :return: DataFrame senza duplicati.
    """
    dataframe_temp = dataframe.copy()

    # Converti temporaneamente in stringa le colonne che contengono dati complessi
    for col in dataframe_temp.columns:
        if dataframe_temp[col].apply(lambda x: isinstance(x, (list, dict, set, pd.Series, pd.DataFrame))).any():
            dataframe_temp[col] = dataframe_temp[col].astype(str)

    # Trova gli ID delle righe duplicate (ignorando la colonna 'ID')
    duplicati = dataframe_temp[dataframe_temp.duplicated(subset=[col for col in dataframe_temp.columns if col != 'ID'],
                                                         keep='first')]['ID']

    # Rimuove i duplicati dal DataFrame originale
    dataframe = dataframe[~dataframe['ID'].isin(duplicati)].copy()

    return dataframe


def permutazione_generi_numerici(dataframe, vector_size=100, window=5, min_count=1, workers=4, sg=0):
    """
    Converte i generi dei Film e delle Serie TV in vettori numerici utilizzando Word2Vec.

    :param dataframe: DataFrame di Pandas contenente il dataset da modificare.
    :param vector_size: Dimensione dei vettori generati da Word2Vec.
    :param window: Dimensione della finestra di contesto per Word2Vec.
    :param min_count: Frequenza minima per includere una parola nel modello.
    :param workers: Numero di thread per il training.
    :param sg: Metodo di addestramento (0=CBOW, 1=Skip-gram).
    :return: DataFrame con i generi convertiti in vettori numerici.
    """

    # Convertire i generi in liste
    dataframe['Generi'] = dataframe['Generi'].apply(lambda x: x.split(', ') if isinstance(x, str) else [])

    # Creare una lista complessiva di generi
    generi_totali = dataframe['Generi'].tolist()

    # Addestrare il modello Word2Vec
    modello = Word2Vec(sentences=generi_totali, vector_size=vector_size,
                       window=window, min_count=min_count, workers=workers, sg=sg)

    vocabolario = list(modello.wv.key_to_index.keys())
    print(f"Generi presenti nel modello di Word2Vec:\n{vocabolario}\n")

    # Otteniamo la dimensione dei vettori
    vector_dim = modello.vector_size

    # Funzione per generare i vettori dei generi
    def genera_vettore(generi):
        vettori = [modello.wv[genere] for genere in generi if genere in modello.wv]
        return np.mean(vettori, axis=0) if vettori else np.zeros(vector_dim)

    # Applicare la funzione al DataFrame
    dataframe['Vettori_Generi'] = dataframe['Generi'].apply(genera_vettore)

    return dataframe


def permutazione_tipo(dataframe):
    """
    Converte i valori della colonna 'Tipo' per uniformare la nomenclatura:
    - 'Movie' → 'Film'
    - 'TV Show' → 'Serie TV'

    :param dataframe: DataFrame di Pandas contenente il dataset da modificare.
    :return: DataFrame con la colonna 'Tipo' aggiornata.
    """

    # Controlla se la colonna 'Tipo' esiste nel DataFrame prima di modificarla
    if "Tipo" in dataframe.columns:
        dataframe["Tipo"] = dataframe["Tipo"].replace({"Movie": "Film", "TV Show": "Serie TV"})

    return dataframe
