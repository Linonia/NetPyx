import re
import nltk
import numpy as np
import seaborn
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt

# Inizializza NLTK
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def preprocess_text(row):
    """
    Pre-elabora il testo:
    - Converte in minuscolo
    - Rimuove punteggiatura e numeri
    - Tokenizza con NLTK
    - Rimuove stopwords
    - Applica lemmatizzazione
    """
    text = f"{row['Generi']} {row['Descrizione']}"
    if not isinstance(text, str) or text.strip() == "":
        return []
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Rimuove numeri e punteggiatura
    words = word_tokenize(text)  # Tokenizzazione
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 2]
    return words  # Restituiamo una lista di parole per Word2Vec


def train_word2vec(df, vector_size=250, window=6, min_count=1, workers=4, epochs=40, genre_weight=1.5):
    """
    Addestra un modello Word2Vec sulle descrizioni dei film, dando più peso ai generi.
    """
    sentences = df.dropna(subset=['Descrizione']).apply(preprocess_text, axis=1).tolist()

    # Creiamo un set con tutti i generi presenti nel dataset
    unique_genres = set()
    for genres in df['Generi'].dropna():
        unique_genres.update(genres)

    model = Word2Vec(
        sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=1,  # Skip-gram (meglio per testi brevi)
        hs=1,  # Hierarchical Softmax per un modello più preciso
        alpha=0.01,
        min_alpha=0.0001,
        epochs=epochs
    )

    # Creiamo un dizionario dei pesi per le parole
    weights = {}
    for sentence in sentences:
        for word in sentence:
            if word in unique_genres:
                weights[word] = genre_weight  # Peso maggiore per i generi
            else:
                weights[word] = 1.0  # Peso normale

    # Modifica dei vettori in base ai pesi
    for word in model.wv.index_to_key:
        if word in weights:
            model.wv.vectors[model.wv.key_to_index[word]] *= weights[word]

    return model


def get_similar_movies(df, model, keywords, topn=10):
    """
    Restituisce i film con descrizioni più simili alle parole chiave fornite.
    """
    valid_keywords = [lemmatizer.lemmatize(word) for word in keywords if word in model.wv]
    if not valid_keywords:
        return df.iloc[0:0]  # Restituisce un DataFrame vuoto

    try:
        # Ottieni parole simili con punteggio alto
        similar_words = []
        for word in valid_keywords:
            try:
                similar = model.wv.most_similar(word, topn=topn)
                similar_words.extend([w for w, score in similar if score > 0.60])
            except KeyError:
                continue

        all_words = set(valid_keywords + similar_words)

        def cosine_similarity(description):
            words = description.lower().split()
            similarity_scores = [model.wv.similarity(w, k) for w in words for k in all_words if w in model.wv]
            return np.mean(similarity_scores) if similarity_scores else 0.01

        df['similarity'] = df['Descrizione'].apply(cosine_similarity)

        # BOOST: Aggiungiamo un bonus se il genere del film è tra le parole chiave
        def apply_genre_boost(row):
            genres = [g.lower() for g in row['Generi']] if isinstance(row['Generi'], list) else row[
                'Generi'].lower().split(', ')
            if any(genre in valid_keywords for genre in genres):
                return row['similarity'] * 1.15  # Aumento del 15%
            return row['similarity']

        df['similarity'] = df.apply(apply_genre_boost, axis=1)

        return df.sort_values(by='similarity', ascending=False).head(topn)
    except KeyError:
        return df.iloc[0:0]  # Restituisce un DataFrame vuoto


def get_similar_movies_with_plot(df, model, keywords, topn=10, stampe=True):
    """
    Restituisce i film con descrizioni più simili alle parole chiave e genera un grafico con i punteggi se richiesto.
    """
    valid_keywords = [lemmatizer.lemmatize(word) for word in keywords if word in model.wv]
    if not valid_keywords:
        print("Nessuna parola chiave valida trovata nel modello.")
        return df.iloc[0:0]  # Restituisce un DataFrame vuoto

    try:
        # Trova parole simili con score > 0.60
        similar_words = []
        for word in valid_keywords:
            try:
                similar = model.wv.most_similar(word, topn=topn)
                similar_words.extend([w for w, score in similar if score > 0.60])
            except KeyError:
                continue

        all_words = set(valid_keywords + similar_words)

        # Calcola la similarità media tra descrizione del film e parole chiave
        def cosine_similarity(description):
            words = description.lower().split()
            similarity_scores = [model.wv.similarity(w, k) for w in words for k in all_words if w in model.wv]
            return np.mean(similarity_scores) if similarity_scores else 0.01

        df['similarity'] = df['Descrizione'].apply(cosine_similarity)

        # BOOST: Aggiungiamo un bonus se il genere del film è tra le parole chiave
        def apply_genre_boost(row):
            genres = [g.lower() for g in row['Generi']] if isinstance(row['Generi'], list) else row[
                'Generi'].lower().split(', ')
            if any(genre in valid_keywords for genre in genres):
                return row['similarity'] * 1.2  # Aumento del 15%
            return row['similarity']

        df['similarity'] = df.apply(apply_genre_boost, axis=1)

        # Ordina i risultati
        df_results = df.sort_values(by='similarity', ascending=False).head(topn)

        if stampe:
            # **GRAFICO**
            labels = df_results['Titolo'].tolist()
            scores = df_results['similarity'].tolist()

            plt.figure(figsize=(10, 7.5))
            plt.bar(labels, scores, color=seaborn.color_palette("Purples", len(scores)))
            plt.xlabel('Film consigliati')
            plt.ylabel('Score di similarità')
            plt.title('Qualità delle Raccomandazioni')
            plt.ylim(0, 1)
            plt.xticks(rotation=20, ha="center", fontsize=10)
            plt.subplots_adjust(bottom=0.3)  # Aggiunge più spazio sotto per i titoli
            plt.show()

        return df_results[['Titolo', 'similarity', 'Generi', 'Descrizione']]

    except KeyError:
        return df.iloc[0:0]  # Restituisce un DataFrame vuoto


def get_similar_words(model, words, topn=5):
    """
    Restituisce le parole più simili nel modello addestrato.
    """
    for word in words:
        lemma_word = lemmatizer.lemmatize(word)  # Lemmatizzazione della parola
        if lemma_word in model.wv:
            print(f"Parole simili a {lemma_word}: {model.wv.most_similar(lemma_word, topn=topn)}")
        else:
            print(f"{lemma_word} non è nel vocabolario del modello.")


def avg_similarity_top3(df, model, keywords):
    """
    Calcola la similarità media solo con le 3 parole più simili della descrizione.
    """
    valid_keywords = [lemmatizer.lemmatize(w) for w in keywords if w in model.wv]
    if not valid_keywords:
        return 0

    suggested_movies = get_similar_movies(df, model, keywords)
    descriptions = suggested_movies['Descrizione'].dropna().tolist()

    if not descriptions:
        return 0

    total_score = 0
    count = 0

    for desc in descriptions:
        words = desc.lower().split()
        similarities = []

        for word in words:
            if word in model.wv:
                sim_scores = [model.wv.similarity(word, k) for k in valid_keywords if k in model.wv]
                if sim_scores:
                    similarities.append(max(sim_scores))  # Prendiamo solo il massimo

        top3_similarities = sorted(similarities, reverse=True)[:3]  # Consideriamo solo le 3 parole migliori

        if top3_similarities:
            total_score += np.mean(top3_similarities)
            count += 1

    return total_score / count if count > 0 else 0


def plot_keyword_coherence(df, model, keywords_list):
    scores = [avg_similarity_top3(df, model, kw) for kw in keywords_list]
    labels = ['-'.join(kw) for kw in keywords_list]

    plt.figure(figsize=(10, 7.5))
    plt.bar(labels, scores, color=seaborn.color_palette("Blues", len(scores)))
    plt.xlabel('Liste di parole chiave')
    plt.ylabel('Similarità media TF-IDF')
    plt.title('Coerenza delle Raccomandazioni')
    plt.ylim(0, 1)
    plt.xticks(rotation=20, ha="center", fontsize=10)
    plt.subplots_adjust(bottom=0.2)
    plt.show()
