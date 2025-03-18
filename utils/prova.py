from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
import numpy as np


def preprocess_genres(genres):
    stop_words = {"movies", "movie", "tv", "shows", "show"}  # Termini da ignorare
    processed = [" ".join([word for word in genre.lower().split() if word not in stop_words]) for genre in genres]
    return processed


def cluster_genres(genres, n_clusters=None):
    processed_genres = preprocess_genres(genres)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(processed_genres)

    if n_clusters is None:
        n_clusters = int(np.sqrt(len(genres)))  # Stima del numero di cluster

    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    labels = clustering.fit_predict(X.toarray())

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

clusters = cluster_genres(genres)
for cluster, grouped_genres in clusters.items():
    print(f"Cluster {cluster}: {grouped_genres}")


def group_genres(genres):
    clusters = {
        "Romance": {"Romantic Movies", "Romantic TV Shows"},
        "Horror": {"Horror Movies", "TV Horror"},
        "Comedy": {"Comedies", "Stand-Up Comedy", "Stand-Up Comedy & Talk Shows", "TV Comedies"},
        "Drama": {"Dramas", "TV Dramas"},
        "Thriller": {"Thrillers", "TV Thrillers"},
        "Mystery": {"TV Mysteries"},
        "Crime": {"Crime TV Shows"},
        "Action & Adventure": {"Action & Adventure", "TV Action & Adventure"},
        "Science Fiction & Fantasy": {"Sci-Fi & Fantasy", "TV Sci-Fi & Fantasy"},
        "Documentary": {"Documentaries", "Docuseries"},
        "Reality": {"Reality TV"},
        "Classic & Cult": {"Classic Movies", "Cult Movies", "Classic & Cult TV"},
        "International": {"International Movies", "International TV Shows", "Spanish-Language TV Shows", "British TV Shows", "Korean TV Shows"},
        "Children & Family": {"Children & Family Movies", "Kids' TV"},
        "Faith & Spirituality": {"Faith & Spirituality"},
        "Anime": {"Anime Features", "Anime Series"},
        "Teen": {"Teen TV Shows"},
        "Music & Musicals": {"Music & Musicals"},
        "Sports": {"Sports Movies"},
        "Science & Nature": {"Science & Nature TV"}
    }