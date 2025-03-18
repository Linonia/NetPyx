import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

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


def unifica_generi(df):
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
