import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from itertools import combinations

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


def replace_genres(dataset_genres, cluster_map):
    """ Sostituisce i generi originali con quelli nuovi """
    replaced_genres = [cluster_map.get(genre, genre) for genre in dataset_genres]
    return replaced_genres


# Esempio d'uso
genres = [
    "Movies", "TV Dramas", "Sports Movies", "Stand-Up Comedy", "Horror Movies", "LGBTQ Movies", "TV Thrillers",
    "Stand-Up Comedy & Talk Shows", "Romantic TV Shows", "Docuseries", "Action & Adventure", "International Movies",
    "Cult Movies", "TV Shows", "Classic & Cult TV", "Crime TV Shows", "International TV Shows", "Music & Musicals",
    "Comedies", "Teen TV Shows", "Classic Movies", "Dramas", "Kids' TV", "British TV Shows", "Reality TV", "Thrillers",
    "TV Comedies", "TV Horror", "Children & Family Movies", "Faith & Spirituality", "TV Mysteries",
    "Spanish-Language TV Shows", "TV Action & Adventure", "Romantic Movies", "Independent Movies", "Anime Features",
    "TV Sci-Fi & Fantasy", "Documentaries", "Sci-Fi & Fantasy", "Science & Nature TV", "Korean TV Shows", "Anime Series"
]

# Clusterizzazione e assegnazione dei nomi
clusters = cluster_genres(genres, n_clusters=30)
cluster_names = assign_cluster_names(clusters)

# Creare una mappa "genere originale → nuovo genere"
genre_mapping = {genre: cluster_names[cluster] for cluster, genres in clusters.items() for genre in genres}

# Sostituire i generi nel dataset
new_genres = replace_genres(genres, genre_mapping)

# Stampa i risultati
print("Mappa dei nuovi generi:")
for old, new in genre_mapping.items():
    print(f"{old} → {new}")

print("\nGeneri aggiornati nel dataset:")
print(new_genres)