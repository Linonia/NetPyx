import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity


def prepare_dataset(df):
    """
    Prepara il dataset trasformando i generi in vettori numerici e applicando TF-IDF alle descrizioni.
    """
    # Convertiamo i vettori di generi in array numpy solo se sono stringhe
    df["Vettori_Generi"] = df["Vettori_Generi"].apply(
        lambda x: np.fromstring(x.strip("[]"), sep=" ") if isinstance(x, str) else x)

    # Trasformiamo le descrizioni in vettori TF-IDF
    vectorizer = TfidfVectorizer(max_features=100)  # Limitiamo a 100 caratteristiche per semplicità
    tfidf_matrix = vectorizer.fit_transform(df["Descrizione"].fillna(""))

    # Concatenazione delle feature (Generi + TF-IDF Descrizione)
    X_genres = np.vstack(df["Vettori_Generi"].values)  # Stack dei vettori di generi
    X_tfidf = tfidf_matrix.toarray()  # Convertiamo la matrice TF-IDF in array

    # Creiamo la matrice finale di feature
    X_final = np.hstack((X_genres, X_tfidf))

    return X_final, df


def ask_user_ratings(df, num_ratings=10):
    """
    Chiede all'utente di valutare un numero specifico di film.
    """
    sample_indices = random.sample(range(len(df)), num_ratings)
    user_ratings = {}

    print("Valutazione dei film. Rispondi con un numero da 1 a 10 o scrivi 'skip' per saltare.")

    for idx in sample_indices:
        film = df.iloc[idx]
        print(f"\nTitolo: {film['Titolo']}")
        print(f"Generi: {film['Generi']}")
        print(f"Descrizione: {film['Descrizione']}")
        visto = input("Hai visto questo film? (sì/no): ").strip().lower()

        if visto == "sì":
            while True:
                voto = input("Dai un voto da 1 a 10: ")
                if voto.isdigit() and 1 <= int(voto) <= 10:
                    user_ratings[idx] = (int(voto), 1.0)
                    break
                else:
                    print("Inserisci un numero valido da 1 a 10.")
        else:
            while True:
                voto = input("Quanto ti ispira, da 1 a 10?: ")
                if voto.isdigit() and 1 <= int(voto) <= 10:
                    user_ratings[idx] = (int(voto), 0.3)
                    break
                else:
                    print("Inserisci un numero valido da 1 a 10.")

    return user_ratings


def simulate_user_ratings(df, num_ratings=10):
    """
    Simula le valutazioni dell'utente assegnando punteggi casuali ai film.
    """
    random.seed(42)
    np.random.seed(42)
    sample_indices = random.sample(range(len(df)), num_ratings)
    user_ratings = {}

    for idx in sample_indices:
        visto = random.choice([True, False])
        if visto:
            rating = random.randint(1, 10)
            weight = 1.0
        else:
            rating = random.randint(1, 10)
            weight = 0.3

        user_ratings[idx] = (rating, weight)

    return user_ratings


def train_model(X, user_ratings):
    """
    Addestra un modello di regressione Ridge utilizzando le valutazioni dell'utente.
    """
    # Creazione del vettore target
    y = np.zeros(len(X))
    weights = np.zeros(len(X))

    for film_id, (rating, weight) in user_ratings.items():
        y[film_id] = rating
        weights[film_id] = weight

    # Addestriamo il modello di regressione Ridge
    model = Ridge(alpha=1.0)
    model.fit(X, y, sample_weight=weights)

    return model


def recommend_movies(model, X, df, top_n=10):
    """
    Genera raccomandazioni di film basate sul modello addestrato.
    """
    # Prediciamo i voti per tutti i film
    predicted_ratings = model.predict(X)

    # Ordiniamo i film in base ai voti previsti
    recommended_indices = np.argsort(predicted_ratings)[::-1]  # Ordine decrescente

    # Prendiamo i top_n film
    recommended_movies = df.iloc[recommended_indices[:top_n]]

    print("\n\n📌 Ecco i film consigliati per te:\n")
    for i, (_, row) in enumerate(recommended_movies.iterrows(), start=1):
        print(f"🎬 {i}. {row['Titolo']} ({row['Tipo']})")
        print(f"   📂 Generi: {row['Generi']}")
        print(f"   📝 Descrizione: {row['Descrizione']}")  # Stampa completa della descrizione
        print("   ----------------------------------------")

    return recommended_movies


def evaluate_recommendations(X, user_ratings, recommended_indices, df):
    """
    Valuta la similarità tra i film consigliati e il profilo ideale dell'utente.
    """
    # Creiamo un vettore ideale basato sulle preferenze dell'utente
    user_vector = np.zeros(X.shape[1])
    total_weight = 0
    for film_id, (rating, weight) in user_ratings.items():
        user_vector += X[film_id] * rating * weight
        total_weight += rating * weight
    if total_weight > 0:
        user_vector /= total_weight

    # Calcoliamo la similarità coseno tra i film consigliati e il vettore ideale
    recommended_vectors = X[recommended_indices]
    similarities = cosine_similarity(recommended_vectors, user_vector.reshape(1, -1)).flatten()

    # Otteniamo i titoli dei film consigliati
    recommended_movies = df.iloc[recommended_indices]

    # Plot della similarità con i titoli
    plt.figure(figsize=(10, 7.5))
    plt.subplots_adjust(left=0.35)
    plt.barh(recommended_movies['Titolo'], similarities, color="lightcoral")
    plt.xlabel("Similarità con il profilo utente")
    plt.ylabel("Film Consigliati")
    plt.title("Quanto i film consigliati rispettano le preferenze utente")
    plt.gca().invert_yaxis()
    plt.savefig("plots/apprendimento_supervisionato_plot.jpg")
    plt.show()


def user_testing_sup_train(dataframe):
    """
    Esegue la fase di testing supervisionato chiedendo direttamente all'utente le valutazioni.
    """
    seed = random.randint(0, 10000)
    random.seed(seed)
    np.random.seed(seed)

    print(f"Seed usato per il test: {seed}")

    X, sup_dataframe = prepare_dataset(dataframe)
    user_ratings = ask_user_ratings(sup_dataframe)
    model = train_model(X, user_ratings)
    recommended_movies = recommend_movies(model, X, sup_dataframe)
    recommended_indices = recommended_movies.index.to_numpy()
    evaluate_recommendations(X, user_ratings, recommended_indices, sup_dataframe)


def simulate_testing_sup_train(dataframe, stampe=False):
    # Addestramento supervisionato

    # preparazione del dataset per l'addestramento (creo una copia)
    X, sup_dataframe = prepare_dataset(dataframe)

    # richiediamo opinioni utente
    user_ratings = simulate_user_ratings(sup_dataframe, num_ratings=20)

    # addestriamo il modello
    model = train_model(X, user_ratings)

    # generiamo raccomandazioni
    recommended_movies = recommend_movies(model, X, sup_dataframe)

    if stampe:
        recommended_indices = recommended_movies.index.to_numpy()
        evaluate_recommendations(X, user_ratings, recommended_indices, sup_dataframe)
