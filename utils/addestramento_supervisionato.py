import time
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity
import shutil
import textwrap


def prepare_dataset(dataframe):
    """
    Prepara il dataset trasformando i generi in vettori numerici e applicando TF-IDF alle descrizioni.

    - I generi vengono convertiti in vettori numerici tramite Word2Vec.
    - Le descrizioni vengono trasformate in vettori TF-IDF.
    - Le due rappresentazioni vengono concatenate in un'unica matrice di feature.

    :param dataframe: DataFrame di Pandas contenente il dataset da preparare.
    :return: Una matrice numpy con le feature del dataset e il DataFrame originale.
    """

    # Convertiamo i vettori dei generi da stringhe in array numpy, se necessario
    dataframe["Vettori_Generi"] = dataframe["Vettori_Generi"].apply(
        lambda x: np.fromstring(x.strip("[]"), sep=" ") if isinstance(x, str) else x)

    # Applichiamo TF-IDF alle descrizioni per estrarre caratteristiche testuali
    vectorizer = TfidfVectorizer(max_features=100)  # Limitiamo a 100 feature per efficienza
    tfidf_matrix = vectorizer.fit_transform(dataframe["Descrizione"].fillna(""))

    # Convertiamo i dati TF-IDF in array numpy
    X_tfidf = tfidf_matrix.toarray()

    # Combiniamo i vettori dei generi con quelli TF-IDF delle descrizioni
    X_genres = np.vstack(dataframe["Vettori_Generi"].values)  # Stack dei vettori di generi
    X_final = np.hstack((X_genres, X_tfidf))  # Concatenazione delle feature

    return X_final, dataframe


def ask_user_ratings(dataframe, num_ratings=10):
    """
    Chiede all'utente di valutare un numero specifico di film, raccogliendo sia valutazioni effettive che giudizi di ispirazione.

    - Se l'utente ha visto il film, può dare un voto da 1 a 10 con peso pieno (1.0).
    - Se l'utente non ha visto il film, valuta quanto lo ispira con un voto da 1 a 10, ma con peso ridotto (0.3).
    - Se l'utente scrive "skip", il film viene ignorato e ne viene proposto un altro.
    - Se l'utente scrive "stop", il processo termina immediatamente.

    :param dataframe: DataFrame contenente il dataset dei film.
    :param num_ratings: Numero di film da sottoporre all'utente per la valutazione.
    :return: Dizionario {indice_film: (voto, peso)}, dove il peso è 1.0 per i film visti e 0.6 per quelli solo ispirati.
    """

    user_ratings = {}
    available_indices = list(range(len(dataframe)))  # Lista di indici disponibili per la selezione
    random.shuffle(available_indices)  # Mescola gli indici per proporre film casualmente

    print("Valutazione dei film. Rispondi con un numero da 1 a 10, scrivi 'skip' per saltare o 'stop' per terminare.")

    while len(user_ratings) < num_ratings and available_indices:
        idx = available_indices.pop(0)  # Estrae un indice casuale
        film = dataframe.iloc[idx]

        print(f"\nTitolo: {film['Titolo']}")
        print(f"Generi: {film['Generi']}")
        print(f"Descrizione: {film['Descrizione']}")

        visto = input("Hai visto questo film? (sì/no/skip/stop): ").strip().lower()

        if visto == "stop":
            break  # L'utente interrompe il processo

        if visto == "skip":
            continue  # Non conta la valutazione e passa al film successivo

        if visto == "sì":
            while True:
                voto = input("Dai un voto da 1 a 10: ")
                if voto.isdigit() and 1 <= int(voto) <= 10:
                    user_ratings[idx] = (int(voto), 1.0)  # Peso pieno per film visti
                    break
                else:
                    print("Inserisci un numero valido da 1 a 10.")
        else:
            while True:
                voto = input("Quanto ti ispira, da 1 a 10?: ")
                if voto.isdigit() and 1 <= int(voto) <= 10:
                    user_ratings[idx] = (int(voto), 0.6)  # Peso ridotto per valutazioni di ispirazione
                    break
                else:
                    print("Inserisci un numero valido da 1 a 10.")

    return user_ratings


def simulate_user_ratings(dataframe, num_ratings=10):
    """
    Simula le valutazioni dell'utente assegnando punteggi casuali ai film.

    - Il sistema sceglie casualmente un sottoinsieme di film dal dataset.
    - Per ogni film selezionato, decide casualmente se l'utente lo ha visto o meno.
    - Se lo ha visto, assegna un voto da 1 a 10 con peso 1.0.
    - Se non lo ha visto, assegna un voto da 1 a 10 con peso ridotto (0.3).

    :param dataframe: DataFrame contenente il dataset dei film.
    :param num_ratings: Numero di film da valutare.
    :return: Dizionario {indice_film: (voto, peso)}, dove il peso è 1.0 per i film visti e 0.6 per quelli solo ispirati.
    """

    # Imposta un seed per la riproducibilità dei risultati
    random.seed(42)
    np.random.seed(42)

    # Seleziona casualmente gli indici dei film da valutare
    sample_indices = random.sample(range(len(dataframe)), num_ratings)
    user_ratings = {}

    for idx in sample_indices:
        # Decide casualmente se l'utente ha visto il film
        visto = random.choice([True, False])

        # Assegna un voto casuale da 1 a 10
        rating = random.randint(1, 10)

        # Imposta il peso (1.0 per film visti, 0.6 per quelli non visti ma ispiranti)
        weight = 1.0 if visto else 0.6

        # Salva la valutazione nel dizionario
        user_ratings[idx] = (rating, weight)

    return user_ratings


def train_model(X, user_ratings):
    """
    Addestra un modello di regressione Ridge utilizzando le valutazioni dell'utente.

    - Crea un vettore target `y` in cui i film valutati hanno il punteggio assegnato dall'utente,
      mentre gli altri rimangono a 0.
    - Crea un vettore di pesi `weights` che assegna maggiore importanza ai film visti rispetto a quelli ispirati.
    - Addestra un modello di regressione Ridge per prevedere le valutazioni su tutti i film.

    :param X: Matrice delle feature dei film (descrizione + generi numerici).
    :param user_ratings: Dizionario delle valutazioni dell'utente {indice_film: (voto, peso)}.
    :return: Modello addestrato di regressione Ridge.
    """

    # Inizializza i vettori target (y) e pesi (weights) con zeri
    y = np.zeros(len(X))
    weights = np.zeros(len(X))

    # Assegna i voti e i pesi ai film valutati dall'utente
    for film_id, (rating, weight) in user_ratings.items():
        y[film_id] = rating
        weights[film_id] = weight

    # Addestramento del modello di regressione Ridge con i pesi forniti
    model = Ridge(alpha=1.0)
    model.fit(X, y, sample_weight=weights)

    return model


def recommend_movies(model, X, dataframe, top_n=5):
    """
    Genera raccomandazioni di film basate sul modello addestrato.

    - Utilizza il modello di regressione per prevedere il punteggio di ogni film.
    - Ordina i film in base al punteggio previsto, dal più alto al più basso.
    - Seleziona i top_n film con il punteggio più alto e li stampa con dettagli.

    :param model: Modello di regressione addestrato.
    :param X: Matrice delle feature dei film.
    :param dataframe: DataFrame contenente i dati dei film.
    :param top_n: Numero di film da raccomandare.
    :return: DataFrame con i film consigliati.
    """

    # Predice i voti per tutti i film basandosi sulle feature fornite
    predicted_ratings = model.predict(X)

    # Ordina gli indici dei film in base ai voti previsti in ordine decrescente
    recommended_indices = np.argsort(predicted_ratings)[::-1]

    # Seleziona i primi top_n film raccomandati
    recommended_movies = dataframe.iloc[recommended_indices[:top_n]]

    # Ottiene la larghezza attuale del terminale e imposta un valore minimo di 80 caratteri
    terminal_width = shutil.get_terminal_size().columns
    wrap_width = max(120, terminal_width - 20)

    # Stampa i risultati
    print("📌 Ecco i film consigliati per te:\n")
    for i, (_, row) in enumerate(recommended_movies.iterrows(), start=1):
        # Recupera la durata, se disponibile, altrimenti mostra "N/A"
        durata = row["Durata"] if pd.notna(row["Durata"]) else "N/A"

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
            subsequent_indent=" " * (len(prefix) + 1) # Allinea il testo sotto la "D"
        )

        print(wrapped_description)
        print("   ----------------------------------------")  # Separatore tra i risultati

    return recommended_movies


def evaluate_recommendations(X, user_ratings, recommended_indices, dataframe):
    """
    Valuta la similarità tra i film consigliati e il profilo ideale dell'utente.

    - Crea un vettore "profilo ideale" dell'utente basato sulle valutazioni date.
    - Calcola la similarità coseno tra i film consigliati e il profilo ideale.
    - Genera un grafico a barre per visualizzare la similarità dei film consigliati.

    :param X: Matrice delle feature dei film.
    :param user_ratings: Dizionario delle valutazioni date dall'utente.
    :param recommended_indices: Indici dei film consigliati.
    :param dataframe: DataFrame contenente i film.
    """

    # Creazione del vettore ideale dell'utente basato sulle sue preferenze
    user_vector = np.zeros(X.shape[1])
    total_weight = 0

    # Somma ponderata delle caratteristiche dei film valutati
    for film_id, (rating, weight) in user_ratings.items():
        user_vector += X[film_id] * rating * weight
        total_weight += rating * weight

    # Normalizzazione del vettore ideale per ottenere un valore medio
    if total_weight > 0:
        user_vector /= total_weight

    # Estrazione dei vettori corrispondenti ai film consigliati
    recommended_vectors = X[recommended_indices]

    # Calcolo della similarità coseno tra i film consigliati e il profilo utente
    similarities = cosine_similarity(recommended_vectors, user_vector.reshape(1, -1)).flatten()

    # Otteniamo i titoli dei film consigliati per il grafico
    recommended_movies = dataframe.iloc[recommended_indices]

    # Creazione del grafico a barre per visualizzare le similarità
    plt.figure(figsize=(10, 7.5))
    plt.subplots_adjust(left=0.35)  # Spazio extra per titoli lunghi
    plt.barh(recommended_movies['Titolo'], similarities, color="lightcoral")
    plt.xlabel("Similarità con il profilo utente")
    plt.ylabel("Film Consigliati")
    plt.title("Quanto i film consigliati rispettano le preferenze utente")
    plt.gca().invert_yaxis()  # Invertiamo l'asse per mettere il film più simile in alto

    # Salviamo e mostriamo il grafico
    plt.savefig("plots/apprendimento_supervisionato_plot.jpg")
    plt.show()


def user_testing_sup_train(dataframe, stampe=False):
    """
    Esegue la fase di testing supervisionato chiedendo direttamente all'utente le valutazioni.

    - Genera un seed casuale per rendere il test riproducibile.
    - Prepara il dataset trasformando generi e descrizioni in vettori numerici.
    - Chiede all'utente di valutare un certo numero di film.
    - Addestra un modello supervisionato sulla base delle valutazioni raccolte.
    - Genera raccomandazioni e le valuta in base alla similarità con il profilo utente.

    :param stampe: Booleano che determina se stampare o no il grafico delle similarità.
    :param dataframe: DataFrame contenente i film da analizzare.
    """

    # Generazione di un seed casuale per garantire ripetibilità nei test
    seed = time.time()   # random.randint(0, 10000)
    random.seed(seed)
    np.random.seed(seed)

    print(f"Seed usato per il test: {seed}")

    # Preparazione del dataset (trasformazione di generi e descrizioni)
    X, sup_dataframe = prepare_dataset(dataframe)

    # Raccolta delle valutazioni dell'utente
    user_ratings = ask_user_ratings(sup_dataframe)

    # Addestramento del modello supervisionato basato sulle valutazioni dell'utente
    model = train_model(X, user_ratings)

    # Generazione delle raccomandazioni basate sul modello addestrato
    recommended_movies = recommend_movies(model, X, sup_dataframe)

    # Se richiesto, valutiamo le raccomandazioni con un grafico
    if stampe:
        recommended_indices = recommended_movies.index.to_numpy()
        evaluate_recommendations(X, user_ratings, recommended_indices, sup_dataframe)


def simulate_testing_sup_train(dataframe, stampe=False):
    """
    Simula la fase di testing supervisionato generando valutazioni casuali dell'utente.

    - Prepara il dataset trasformando generi e descrizioni in vettori numerici.
    - Simula valutazioni casuali dell'utente su un numero predefinito di film.
    - Addestra un modello supervisionato basato sulle valutazioni simulate.
    - Genera raccomandazioni in base al modello addestrato.
    - (Opzionale) Valuta la similarità tra i film consigliati e il profilo utente.

    :param dataframe: DataFrame contenente i film da analizzare.
    :param stampe: Se True, visualizza il grafico di valutazione delle raccomandazioni.
    """

    # Preparazione del dataset per l'addestramento (trasformazione generi e descrizioni)
    print("\n[INFO] Preparazione del dataset per l'addestramento supervisionato...\n")
    X, sup_dataframe = prepare_dataset(dataframe)

    # Simulazione delle valutazioni dell'utente
    print("\n[INFO] Simulazione della votazione utente in corso...\n")
    user_ratings = simulate_user_ratings(sup_dataframe, num_ratings=20)
    print("[OK] Simulazione completata.\n")

    # Addestramento del modello supervisionato basato sulle valutazioni simulate
    print("\n[INFO] Addestramento del modello supervisionato...\n")
    model = train_model(X, user_ratings)
    print("[OK] Modello addestrato con successo.\n")

    # Generazione delle raccomandazioni basate sul modello addestrato
    print("\n[INFO] Generazione raccomandazioni...\n")
    recommended_movies = recommend_movies(model, X, sup_dataframe)
    print("[OK] Raccomandazioni completate.\n")

    # Se richiesto, valutiamo le raccomandazioni con un grafico
    if stampe:
        print("\n[INFO] Valutazione della qualità delle raccomandazioni...\n")
        recommended_indices = recommended_movies.index.to_numpy()
        evaluate_recommendations(X, user_ratings, recommended_indices, sup_dataframe)
        print("[OK] Valutazione completata.\n")

