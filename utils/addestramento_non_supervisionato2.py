import re
import nltk
import gensim
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec, KeyedVectors


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
    text = f"{row['Titolo']} {row['Generi']} {row['Descrizione']}"
    if not isinstance(text, str) or text.strip() == "":
        return []
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Rimuove numeri e punteggiatura
    words = word_tokenize(text)  # Tokenizzazione migliorata
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 2]
    return words  # Restituiamo una lista di parole per Word2Vec


def load_pretrained_word2vec():
    """
    Carica un modello Word2Vec pre-addestrato.
    """
    try:
        model = KeyedVectors.load_word2vec_format('models/GoogleNews-vectors-negative300.bin', binary=True)
        return model
    except Exception as e:
        print("Errore nel caricamento del modello pre-addestrato:", e)
        return None


def fine_tune_word2vec(pretrained_model, df, vector_size=300, window=5, min_count=3, workers=4):
    """
    Riaddestra un modello pre-addestrato di Word2Vec sulle descrizioni dei film.
    """
    if pretrained_model is None:
        print("Errore: il modello pre-addestrato non è stato caricato correttamente.")
        return None

    sentences = df.dropna(subset=['Descrizione']).apply(preprocess_text, axis=1).tolist()

    # Convertire il modello KeyedVectors in Word2Vec
    model = Word2Vec(vector_size=vector_size, window=window, min_count=min_count, workers=workers)
    model.build_vocab_from_freq(pretrained_model.key_to_index)  # Usa il vocabolario del modello pre-addestrato
    model.build_vocab(sentences, update=True)  # Aggiunge parole dal dataset
    model.train(sentences, total_examples=len(sentences), epochs=10)

    return model


def train_word2vec(df, vector_size=100, window=5, min_count=3, workers=4):
    """
    Addestra un modello Word2Vec sulle descrizioni dei film.
    """
    sentences = df.dropna(subset=['Descrizione']).apply(preprocess_text, axis=1).tolist()
    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, workers=workers,
                     sg=1)  # sg=1 per Skip-gram
    return model


def get_similar_movies(df, model, keywords, topn=10):
    """
    Restituisce le righe del dataframe corrispondenti ai film più simili alle parole chiave fornite.
    """
    valid_keywords = [word for word in keywords if word in model.wv]

    if not valid_keywords:
        return df.iloc[0:0]  # Restituisce un DataFrame vuoto con le stesse colonne

    try:
        # Ordina le parole chiave per frequenza nel dataset
        sorted_keywords = sorted(valid_keywords, key=lambda w: model.wv.get_vecattr(w, "count"), reverse=True)

        # Trova parole simili
        similar_words = model.wv.most_similar(sorted_keywords, topn=topn)
        similar_words = [word for word, score in similar_words]

        # Filtra il dataframe per le parole simili nelle descrizioni
        mask = df['Descrizione'].apply(lambda x: any(word in x.lower() for word in similar_words))

        return df.loc[mask].copy()  # Mantiene la struttura originale
    except KeyError:
        return df.iloc[0:0]  # Restituisce un DataFrame vuoto con le stesse colonne


def get_similar_words(model, words):
    """
    Trova parole simili nel modello Word2Vec.
    """
    for word in words:
        lemma_word = lemmatizer.lemmatize(word)  # Lemmatizzazione della parola
        if lemma_word in model.wv:
            print(f"Parole simili a {lemma_word}: {model.wv.most_similar(lemma_word, topn=5)}")
        else:
            print(f"{lemma_word} non è nel vocabolario del modello.")
