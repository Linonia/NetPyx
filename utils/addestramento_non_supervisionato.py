import re
import nltk
import gensim
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec

def setup_nltk():
    """
    Scarica le risorse NLTK solo se non già presenti.
    """
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

# Scarica le risorse NLTK prima di usarle
setup_nltk()

# Inizializzare il lemmatizzatore e le stopwords
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
    - Aggiunge il titolo e i generi alla descrizione
    """
    text = f"{row['Generi']} {row['Descrizione']}"
    if not isinstance(text, str) or text.strip() == "":
        return []
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Rimuove numeri e punteggiatura
    words = word_tokenize(text)  # Tokenizzazione migliorata
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 2]
    return words  # Restituiamo una lista di parole per Word2Vec


def train_word2vec(df, vector_size=150, window=5, min_count=1, workers=4, epochs=20):
    """
    Addestra un modello Word2Vec sulle descrizioni dei film con parametri più restrittivi.
    """
    sentences = df.dropna(subset=['Descrizione']).apply(preprocess_text, axis=1).tolist()

    model = Word2Vec(
        sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=1,  # Skip-gram (meglio per testi brevi)
        hs=1,  # Hierarchical Softmax per un modello più preciso
        epochs=epochs
    )

    return model


def get_similar_movies(df, model, keywords, topn=10):
    """
    Restituisce i film con descrizioni più simili alle parole chiave fornite.
    """
    # Lemmatizza le parole chiave
    valid_keywords = [lemmatizer.lemmatize(word) for word in keywords if word in model.wv]

    if not valid_keywords:
        return df.iloc[0:0]  # Restituisce un DataFrame vuoto con le stesse colonne

    try:
        # Ottieni le parole simili con un punteggio alto
        similar_words = []
        for word in valid_keywords:
            try:
                similar = model.wv.most_similar(word, topn=topn)
                similar_words.extend([w for w, score in similar if score > 0.6])  # Soglia di similarità
            except KeyError:
                continue

        all_words = set(valid_keywords + similar_words)

        # Funzione per calcolare la similarità tra la descrizione e le parole chiave
        def cosine_similarity(description):
            words = description.lower().split()  # Tokenizzazione
            similarity_scores = [model.wv.similarity(w, k) for w in words for k in all_words if w in model.wv]
            return np.mean(similarity_scores) if similarity_scores else 0

        # Applica la funzione e filtra per punteggio di similarità alto
        df['similarity'] = df['Descrizione'].apply(cosine_similarity)
        return df.sort_values(by='similarity', ascending=False).head(topn)

    except KeyError:
        return df.iloc[0:0]  # Restituisce un DataFrame vuoto con le stesse colonne


def get_similar_words(model, words):
    for word in words:
        lemma_word = lemmatizer.lemmatize(word)  # Lemmatizzazione della parola
        if lemma_word in model.wv:
            print(f"Parole simili a {lemma_word}: {model.wv.most_similar(lemma_word, topn=5)}")
        else:
            print(f"{lemma_word} non è nel vocabolario del modello.")

