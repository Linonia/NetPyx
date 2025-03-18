import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


def preprocess_genres(genres):
    stop_words = {"movies", "movie", "tv", "shows", "show", "series"}
    processed = [" ".join([word for word in genre.lower().split() if word not in stop_words]) for genre in genres]
    return processed


def cluster_genres(genres, n_clusters=20):  # Numero di cluster stimato
    processed_genres = preprocess_genres(genres)

    # Convertire i generi in vettori numerici
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(processed_genres)

    # Clustering automatico con KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    # Creare i gruppi basati sui cluster
    clustered = {}
    for genre, label in zip(genres, labels):
        clustered.setdefault(label, []).append(genre)

    return clustered


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

clusters = cluster_genres(genres, n_clusters=30)

# Stampa i risultati
for cluster, grouped_genres in clusters.items():
    print(f"Cluster {cluster}: {grouped_genres}")
