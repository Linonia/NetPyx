import re
import nltk
import numpy as np
import seaborn
import pandas as pd
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import shutil
import textwrap

# Inizializza NLTK
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def preprocess_text(row):
    """
    Pre-elabora il testo combinando generi e descrizione di un film o serie TV.

    La funzione esegue le seguenti operazioni:
    - Concatena i generi con la descrizione.
    - Converte il testo in minuscolo.
    - Rimuove numeri e punteggiatura.
    - Tokenizza il testo con NLTK.
    - Rimuove stopwords e parole troppo corte.
    - Applica la lemmatizzazione per ridurre le parole alla loro radice.

    :param row: Riga del DataFrame contenente i campi 'Generi' e 'Descrizione'.
    :return: Lista di parole pre-elaborate, utile per Word2Vec.
    """

    # Combiniamo i generi e la descrizione in un unico testo
    text = f"{row['Generi']} {row['Descrizione']}"

    # Controlliamo se il testo è valido
    if not isinstance(text, str) or text.strip() == "":
        return []  # Se è vuoto o non è una stringa, restituiamo una lista vuota

    # Convertiamo in minuscolo
    text = text.lower()

    # Rimuoviamo numeri e punteggiatura, mantenendo solo lettere e spazi
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenizziamo il testo
    words = word_tokenize(text)

    # Rimuoviamo stopwords e lemmatizziamo le parole (solo parole con più di 2 lettere)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 2]

    return words  # Restituiamo la lista di parole pre-elaborate


def train_word2vec(dataframe, vector_size=250, window=6, min_count=1, workers=4, epochs=40, genre_weight=1.5):
    """
    Addestra un modello Word2Vec basato sulle descrizioni dei film, assegnando un peso maggiore ai generi.

    Il modello viene addestrato su testi pre-elaborati che combinano generi e descrizione dei film.
    Per migliorare la qualità dei vettori, i generi vengono enfatizzati aumentando il loro peso.

    :param dataframe: DataFrame contenente i dati dei film.
    :param vector_size: Dimensione dei vettori generati dal modello Word2Vec.
    :param window: Dimensione della finestra di contesto per Word2Vec.
    :param min_count: Frequenza minima per includere una parola nel vocabolario.
    :param workers: Numero di thread per il training.
    :param epochs: Numero di epoche per l'addestramento.
    :param genre_weight: Peso maggiore assegnato ai generi per enfatizzarli nei vettori.
    :return: Modello Word2Vec addestrato.
    """

    # Creiamo una lista di frasi tokenizzate (liste di parole) dalle descrizioni pre-elaborate
    sentences = dataframe.dropna(subset=['Descrizione']).apply(preprocess_text, axis=1).tolist()

    # Creiamo un set con tutti i generi presenti nel dataset per identificare le parole da enfatizzare
    unique_genres = set()
    for genres in dataframe['Generi'].dropna():
        unique_genres.update(genres)

    # Inizializziamo il modello Word2Vec con i parametri scelti
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=1,  # Skip-gram, adatto per testi brevi come le descrizioni
        hs=1,  # Hierarchical Softmax per un miglior apprendimento su vocabolari piccoli
        alpha=0.01,
        min_alpha=0.0001,
        epochs=epochs
    )

    # Creiamo un dizionario dei pesi per enfatizzare i generi nei vettori del modello
    weights = {}
    for sentence in sentences:
        for word in sentence:
            if word in unique_genres:
                weights[word] = genre_weight  # Peso maggiore per i generi
            else:
                weights[word] = 1.0  # Peso standard per le altre parole

    # Modifichiamo i vettori del modello applicando i pesi assegnati
    for word in model.wv.index_to_key:
        if word in weights:
            model.wv.vectors[model.wv.key_to_index[word]] *= weights[word]

    return model


def get_similar_movies(dataframe, model, keywords, topn=10):
    """
    Trova i film con descrizioni più simili alle parole chiave fornite, utilizzando un modello Word2Vec.

    La funzione confronta le parole chiave con i vettori delle descrizioni dei film.
    Inoltre, espande la ricerca includendo parole simili e applica un bonus ai film il cui genere corrisponde alle parole chiave.

    :param dataframe: DataFrame contenente i dati dei film.
    :param model: Modello Word2Vec addestrato sulle descrizioni dei film.
    :param keywords: Lista di parole chiave inserite dall'utente per trovare film correlati.
    :param topn: Numero di film da restituire ordinati per rilevanza.
    :return: DataFrame con i film più simili alle parole chiave, ordinati per similarità.
    """

    # Lemmatizziamo le parole chiave e manteniamo solo quelle presenti nel vocabolario del modello Word2Vec
    valid_keywords = [lemmatizer.lemmatize(word) for word in keywords if word in model.wv]
    if not valid_keywords:
        return dataframe.iloc[0:0]  # Restituisce un DataFrame vuoto se nessuna parola chiave è valida

    try:
        # Troviamo parole semanticamente simili per ampliare la ricerca
        similar_words = []
        for word in valid_keywords:
            try:
                similar = model.wv.most_similar(word, topn=topn)  # Ottiene parole correlate
                similar_words.extend([w for w, score in similar if score > 0.60])  # Tiene solo le più rilevanti
            except KeyError:
                continue  # Se una parola chiave non è nel vocabolario, la ignoriamo

        # Uniamo parole chiave originali e parole simili
        all_words = set(valid_keywords + similar_words)

        # Calcoliamo la similarità tra la descrizione di ogni film e il set di parole chiave
        def cosine_similarity(description):
            words = description.lower().split()
            similarity_scores = [model.wv.similarity(w, k) for w in words for k in all_words if w in model.wv]
            return np.mean(similarity_scores) if similarity_scores else 0.01  # Evita errori con descrizioni vuote

        dataframe['similarity'] = dataframe['Descrizione'].apply(cosine_similarity)

        # Aggiungiamo un bonus alla similarità se il genere del film è tra le parole chiave
        def apply_genre_boost(row):
            genres = [g.lower() for g in row['Generi']] if isinstance(row['Generi'], list) else row['Generi'].lower().split(', ')
            if any(genre in valid_keywords for genre in genres):
                return row['similarity'] * 1.15  # Aumento del 15% se il genere corrisponde
            return row['similarity']

        dataframe['similarity'] = dataframe.apply(apply_genre_boost, axis=1)

        # Restituiamo i film ordinati per similarità, prendendo solo i top_n più rilevanti
        return dataframe.sort_values(by='similarity', ascending=False).head(topn)

    except KeyError:
        return dataframe.iloc[0:0]  # Restituisce un DataFrame vuoto in caso di errore


def get_similar_movies_with_plot(dataframe, model, keywords, topn=10, stampe=False):
    """
    Trova i film con descrizioni più simili alle parole chiave e, se richiesto, genera un grafico con i punteggi.

    La funzione utilizza un modello Word2Vec per calcolare la similarità tra le parole chiave fornite dall'utente e le descrizioni dei film.
    Inoltre, espande la ricerca includendo parole semanticamente simili e applica un bonus ai film il cui genere corrisponde alle parole chiave.

    :param dataframe: DataFrame contenente i dati dei film.
    :param model: Modello Word2Vec addestrato sulle descrizioni dei film.
    :param keywords: Lista di parole chiave inserite dall'utente per trovare film correlati.
    :param topn: Numero massimo di film da restituire ordinati per rilevanza.
    :param stampe: Se True, genera un grafico a barre con i punteggi di similarità.
    :return: DataFrame con i film più simili alle parole chiave, ordinati per similarità.
    """

    # Lemmatizziamo le parole chiave e filtriamo quelle presenti nel vocabolario del modello
    valid_keywords = [lemmatizer.lemmatize(word) for word in keywords if word in model.wv]
    if not valid_keywords:
        print("Nessuna parola chiave valida trovata nel modello.")
        return dataframe.iloc[0:0]  # Restituisce un DataFrame vuoto se nessuna parola chiave è valida

    try:
        # Troviamo parole semanticamente simili per ampliare la ricerca
        similar_words = []
        for word in valid_keywords:
            try:
                similar = model.wv.most_similar(word, topn=topn)  # Ottiene parole correlate
                similar_words.extend([w for w, score in similar if score > 0.60])  # Tiene solo le più rilevanti
            except KeyError:
                continue  # Se una parola chiave non è nel vocabolario, la ignoriamo

        # Uniamo parole chiave originali e parole simili
        all_words = set(valid_keywords + similar_words)

        # Calcoliamo la similarità tra la descrizione di ogni film e il set di parole chiave
        def cosine_similarity(description):
            words = description.lower().split()
            similarity_scores = [model.wv.similarity(w, k) for w in words for k in all_words if w in model.wv]
            return np.mean(similarity_scores) if similarity_scores else 0.01  # Evita errori con descrizioni vuote

        dataframe['similarity'] = dataframe['Descrizione'].apply(cosine_similarity)

        # Aggiungiamo un bonus alla similarità se il genere del film è tra le parole chiave
        def apply_genre_boost(row):
            genres = [g.lower() for g in row['Generi']] if isinstance(row['Generi'], list) else row['Generi'].lower().split(', ')
            if any(genre in valid_keywords for genre in genres):
                return row['similarity'] * 1.2  # Aumento del 20% se il genere corrisponde
            return row['similarity']

        dataframe['similarity'] = dataframe.apply(apply_genre_boost, axis=1)

        # Ordiniamo i risultati per similarità e selezioniamo i topn più rilevanti
        dataframe_results = dataframe.sort_values(by='similarity', ascending=False).head(topn)

        if stampe:
            # **GRAFICO DELLE SIMILARITÀ**
            labels = dataframe_results['Titolo'].tolist()
            scores = dataframe_results['similarity'].tolist()

            plt.figure(figsize=(10, 7.5))
            plt.bar(labels, scores, color=seaborn.color_palette("Purples", len(scores)))
            plt.xlabel('Film consigliati')
            plt.ylabel('Score di similarità')
            plt.title('Qualità delle Raccomandazioni')
            plt.ylim(0, 1)
            plt.xticks(rotation=20, ha="center", fontsize=10)
            plt.subplots_adjust(bottom=0.3)  # Aggiunge più spazio sotto per i titoli
            plt.show()

        return dataframe_results[['Titolo', 'similarity', 'Generi', 'Descrizione', 'Tipo', 'Durata']]

    except KeyError:
        return dataframe.iloc[0:0]  # Restituisce un DataFrame vuoto in caso di errore


def get_similar_words(model, words, topn=5):
    """
    Trova e stampa le parole più simili a quelle fornite, basandosi sul modello Word2Vec.

    La funzione lemmatizza le parole in input e verifica se sono presenti nel vocabolario del modello.
    Se una parola è nel modello, stampa le parole più simili con il rispettivo punteggio di similarità.

    :param model: Modello Word2Vec addestrato.
    :param words: Lista di parole per cui trovare termini simili.
    :param topn: Numero di parole simili da restituire per ogni input.
    """

    for word in words:
        lemma_word = lemmatizer.lemmatize(word)  # Lemmatizzazione della parola

        if lemma_word in model.wv:
            # Ottiene le parole più simili con i relativi punteggi
            similar_words = model.wv.most_similar(lemma_word, topn=topn)

            print(f"\n🔍 Parole simili a '{lemma_word}':")
            for i, (similar_word, score) in enumerate(similar_words, start=1):
                print(f"   {i}. {similar_word} (similarità: {score:.2f})")
        else:
            print(f"\n❌ '{lemma_word}' non è nel vocabolario del modello.")  # Messaggio se la parola non è presente


def avg_similarity_top3(dataframe, model, keywords):
    """
    Calcola la similarità media tra le parole chiave e le tre parole più simili nelle descrizioni dei film suggeriti.

    La funzione:
    - Lemmatizza le parole chiave e verifica che siano nel vocabolario del modello.
    - Recupera i film più simili basandosi sulle parole chiave.
    - Per ogni descrizione dei film suggeriti:
      - Tokenizza il testo e calcola la similarità tra ogni parola e le parole chiave.
      - Considera solo le tre parole con la similarità più alta.
      - Calcola la media di questi valori.
    - Restituisce la similarità media tra le tre parole migliori di tutte le descrizioni.

    :param dataframe: DataFrame contenente i film.
    :param model: Modello Word2Vec addestrato.
    :param keywords: Lista di parole chiave per la ricerca dei film.
    :return: Similarità media basata sulle tre parole più affini nelle descrizioni.
    """

    # Lemmatizziamo le parole chiave e selezioniamo solo quelle nel modello
    valid_keywords = [lemmatizer.lemmatize(w) for w in keywords if w in model.wv]
    if not valid_keywords:
        return 0  # Se nessuna parola è valida, restituiamo 0

    # Otteniamo i film suggeriti basandoci sulle parole chiave
    suggested_movies = get_similar_movies(dataframe, model, keywords)
    descriptions = suggested_movies['Descrizione'].dropna().tolist()

    if not descriptions:
        return 0  # Se non ci sono descrizioni, restituiamo 0

    total_score = 0
    count = 0

    # Per ogni descrizione, calcoliamo la similarità con le parole chiave
    for desc in descriptions:
        words = desc.lower().split()
        similarities = []

        for word in words:
            if word in model.wv:
                # Calcoliamo la similarità tra la parola e ogni parola chiave
                sim_scores = [model.wv.similarity(word, k) for k in valid_keywords if k in model.wv]
                if sim_scores:
                    similarities.append(max(sim_scores))  # Prendiamo solo il massimo punteggio per ogni parola

        # Consideriamo solo le 3 parole con la similarità più alta
        top3_similarities = sorted(similarities, reverse=True)[:3]

        if top3_similarities:
            total_score += np.mean(top3_similarities)  # Facciamo la media delle migliori 3
            count += 1

    # Restituiamo la similarità media complessiva
    return total_score / count if count > 0 else 0


def plot_keyword_coherence(dataframe, model, keywords_list):
    """
    Genera un grafico a barre per visualizzare la coerenza delle raccomandazioni basate sulle parole chiave.

    La funzione:
    - Calcola la similarità media (basata sulle 3 parole più simili) per ogni lista di parole chiave.
    - Crea un grafico a barre con i punteggi di similarità ottenuti.
    - Salva il grafico come immagine e lo visualizza.

    :param dataframe: DataFrame contenente i film.
    :param model: Modello Word2Vec addestrato.
    :param keywords_list: Lista di liste di parole chiave da analizzare.
    """

    # Calcoliamo il punteggio di similarità media per ogni lista di parole chiave
    scores = [avg_similarity_top3(dataframe, model, kw) for kw in keywords_list]

    # Creiamo le etichette del grafico unendo le parole chiave con un trattino
    labels = ['-'.join(kw) for kw in keywords_list]

    # Creazione del grafico a barre
    plt.figure(figsize=(10, 7.5))
    plt.bar(labels, scores, color=seaborn.color_palette("Blues", len(scores)))
    plt.xlabel('Liste di parole chiave')
    plt.ylabel('Similarità media')
    plt.title('Coerenza delle Raccomandazioni')
    plt.ylim(0, 1)  # Impostiamo il limite dell'asse Y tra 0 e 1
    plt.xticks(rotation=20, ha="center", fontsize=10)  # Ruotiamo le etichette per leggibilità
    plt.subplots_adjust(bottom=0.2)  # Aggiungiamo spazio per le etichette lunghe

    # Salviamo il grafico come immagine
    plt.savefig("plots/apprendimento_non_supervisionato_plot.jpg")

    # Mostriamo il grafico
    plt.show()


def simulate_testing_non_sup_train(dataframe, stampe=False):
    """
    Simula il test dell'addestramento non supervisionato con Word2Vec e verifica la qualità delle raccomandazioni.

    La funzione:
    - Addestra un modello Word2Vec sulle descrizioni dei film.
    - Interroga il modello con diverse liste di parole chiave.
    - Stampa i risultati delle parole più simili e dei film consigliati.
    - Genera un grafico sulla coerenza delle raccomandazioni se richiesto.

    :param dataframe: DataFrame contenente i film e le descrizioni.
    :param stampe: Se True, genera stampe aggiuntive e grafici.
    :return: Modello Word2Vec addestrato.
    """

    print("\n[INFO] Avvio della simulazione di addestramento non supervisionato...\n")

    # Liste di parole chiave per testare il modello
    word_searching = [["scary", "paranormal"], ["challenge", "death"], ["anime", "village"]]
    print(f"[INFO] Liste di parole chiave utilizzate per il test: {word_searching}\n")

    # Addestriamo il modello Word2Vec sui dati del dataset
    print("[INFO] Addestramento del modello Word2Vec in corso...\n")
    model = train_word2vec(dataframe)
    print("[OK] Modello Word2Vec addestrato con successo.\n")

    # Iteriamo su ogni lista di parole chiave per testare il modello
    for words in word_searching:
        print("\n=========================================")
        print(f"[INFO] Avvio test con parole chiave: {words}")
        print("=========================================\n")

        # Stampiamo le parole più simili nel modello
        get_similar_words(model, words, topn=3)

        # Recuperiamo i film con descrizioni più affini alle parole chiave
        print("\n[INFO] Recupero dei film consigliati...\n")
        suggestions = get_similar_movies_with_plot(dataframe, model, words, topn=5, stampe=stampe)
        print("[OK] Film suggeriti con successo.\n")

        # Stampiamo i risultati principali
        print_recommended_movies(suggestions.head(5))

    # Se richiesto, generiamo un grafico di coerenza delle parole chiave
    if stampe:
        print("\n[INFO] Generazione grafico di coerenza delle parole chiave...\n")
        plot_keyword_coherence(dataframe, model, word_searching)
        print("[OK] Grafico generato con successo.\n")

    print("\n[OK] Simulazione di addestramento non supervisionato completata.\n")
    return model


def search_movies_by_user_input(dataframe, model, stampe=False):
    """
    Permette all'utente di inserire parole chiave per cercare film simili nel dataset.

    La funzione:
    - Richiede all'utente di inserire parole chiave separate da una virgola.
    - Lemmatizza le parole chiave usando la lemmatizzazione già implementata.
    - Trova le parole più simili nel modello Word2Vec.
    - Recupera i film con descrizioni più affini alle parole chiave inserite.
    - Stampa i risultati e genera un grafico sulla coerenza delle raccomandazioni, se richiesto.

    :param dataframe: DataFrame contenente i film e le descrizioni.
    :param model: Modello Word2Vec già addestrato.
    :param stampe: Se True, genera stampe aggiuntive e grafici.
    """

    # Richiediamo all'utente di inserire parole chiave separate da virgole
    words = input("Inserisci parole chiave separate da una virgola: ").strip().split(',')
    words = [word.strip().lower() for word in words]

    print("\n🔄 Lemmatizzazione delle parole chiave in corso...\n")

    # Usa la lemmatizzazione già presente nel codice esistente
    words = [lemmatizer.lemmatize(word) for word in words]

    print(f"\n✅ Parole lemmatizzate: {words}\n")
    print(f"\n\nRisultati per {words}:\n")

    # Stampiamo le parole più simili trovate dal modello
    get_similar_words(model, words, topn=3)

    # Recuperiamo i film con descrizioni più affini alle parole chiave inserite
    suggestions = get_similar_movies_with_plot(dataframe, model, words, topn=5, stampe=stampe)

    # Stampiamo i risultati principali
    print_recommended_movies(suggestions.head(5))

    # Se richiesto, generiamo un grafico di coerenza delle parole chiave inserite
    if stampe:
        plot_keyword_coherence(dataframe, model, [words])


def print_recommended_movies(recommended_movies):
    """
    Stampa i film consigliati in un formato leggibile e strutturato.

    :param recommended_movies: DataFrame contenente i film consigliati con le seguenti colonne:
        - 'Titolo' (str): Nome del film o della serie TV.
        - 'Tipo' (str, opzionale): Tipologia (es. "Film", "Serie TV").
        - 'Durata' (str, opzionale): Durata del film o degli episodi.
        - 'Generi' (str): Elenco dei generi del film.
        - 'Descrizione' (str): Breve sinossi del contenuto.
    """

    terminal_width = shutil.get_terminal_size().columns  # Ottiene la larghezza attuale del terminale
    wrap_width = max(100, terminal_width - 20)  # Evita righe troppo lunghe, garantendo almeno 80 caratteri

    if recommended_movies.empty:
        print("\n❌ Nessun film trovato con le parole chiave fornite.\n")
        return

    print("\n\n📌 Ecco i film consigliati per te:\n")
    for i, (_, row) in enumerate(recommended_movies.iterrows(), start=1):
        # Recupera la durata, se disponibile, altrimenti mostra "N/A"
        durata = row["Durata"] if "Durata" in row and pd.notna(row["Durata"]) else "N/A"

        # Stampa titolo, tipo e durata del contenuto
        print(f"🎬 {i}. {row['Titolo']} ({row.get('Tipo', 'N/A')} - {durata})")

        # Stampa i generi del contenuto
        print(f"   📂 Generi: {row['Generi']}")

        # Formatta la descrizione adattandola alla larghezza del terminale
        prefix = "   📝 Descrizione: "
        adjusted_width = wrap_width - len(prefix)

        wrapped_description = textwrap.fill(
            row["Descrizione"],
            width=adjusted_width,
            initial_indent=prefix,  # Mantiene "📝 Descrizione: " sulla stessa riga
            subsequent_indent=" " * (len(prefix) + 1)  # Allinea il testo sotto la "D"
        )

        print(wrapped_description)
        print("   ----------------------------------------")  # Separatore tra i risultati
