import matplotlib.pyplot as plt
import pandas as pd
import seaborn


def general_informations(dataframe):
    """
    Fornisce informazioni generali sul dataset.

    :param dataframe: DataFrame da analizzare.
    :return:
    """
    numero_righe = 5  # Numero di righe da visualizzare

    # Stampa delle prime righe del dataset
    print(f"\nVerifica dell'avvenuto caricamento del dataset tramite la stampa"
          f" delle prime {numero_righe} righe:\n\n{dataframe.head(numero_righe)}\n")

    # Stampa della forma del dataset (numero di righe e colonne)
    print(f"Numero di righe e colonne presenti nel dataset: {dataframe.shape}\n")

    # Stampa di informazioni generali sul dataset
    print(f"Altre informazioni generali presenti nel dataset:")
    dataframe.info()


def bar_product_subdivision(dataframe):
    """
    Genera un grafico a barre per visualizzare la suddivisione tra Film e Serie TV.

    :param dataframe: DataFrame contenente la colonna 'type' con le categorie 'Movie' e 'TV Show'.
    :return:
    """
    # Creazione del grafico a barre con il conteggio dei tipi di prodotto
    dataframe['type'].value_counts().iloc[::-1].plot(kind='bar',
                                                     title="Numero di Film (Movie) e Serie TV (TV Show)",
                                                     figsize=(8, 6),
                                                     color=['red', 'blue'])

    # Configurazione degli assi
    plt.xticks(rotation=0)
    plt.xlabel("Tipo")
    plt.ylabel("Numero di prodotti")

    # Salvataggio del grafico
    plt.savefig("plots/suddivisione_prodotti.jpg")

    # Visualizzazione del grafico
    plt.show()

    # Messaggio di conferma
    print(f"\nGrafico del numero di Film e Serie TV a confronto stampato.\n")


def bar_plot_series(dataframe):
    """
    Genera un grafico a barre per visualizzare la distribuzione delle Serie TV in base alla classificazione per età.

    :param dataframe: DataFrame contenente le colonne 'type' e 'rating'.
    :return:
    """
    # Selezione delle sole Serie TV
    dataframe_selection = dataframe[dataframe['type'] == "TV Show"]

    # Creazione della figura
    plt.figure(figsize=(10, 7.5))

    # Grafico a barre con il conteggio delle serie TV per categoria di età
    ax = seaborn.countplot(x="rating",
                           data=dataframe_selection,
                           hue="rating",
                           palette="Set1",
                           order=dataframe_selection['rating'].value_counts().index[0:12],
                           legend=False)

    # Configurazione del grafico
    ax.set_title(f"Serie TV - Divisione per target di età")
    ax.set_ylabel(f"Numero di Serie TV")
    ax.set_xlabel("Categoria di età")

    # Salvataggio del grafico
    plt.savefig("plots/categoria_eta_serie.jpg")

    # Visualizzazione del grafico
    plt.show()

    # Messaggio di conferma
    print(f"\nGrafico della classificazione delle Serie TV per target di età stampato.\n")


def bar_plot_movie(dataframe):
    """
    Genera un grafico a barre per visualizzare la distribuzione dei Film in base alla classificazione per età.

    :param dataframe: DataFrame contenente le colonne 'type' e 'rating'.
    :return:
    """
    # Selezione dei soli Film
    dataframe_selection = dataframe[dataframe['type'] == "Movie"]

    # Creazione della figura
    plt.figure(figsize=(10, 7.5))

    # Grafico a barre con il conteggio dei film per categoria di età
    ax = seaborn.countplot(x="rating",
                           data=dataframe_selection,
                           hue="rating",
                           palette="Set1",
                           order=dataframe_selection['rating'].value_counts().index[0:12],
                           legend=False)

    # Configurazione del grafico
    ax.set_title(f"Film - Divisione per target di età")
    ax.set_ylabel(f"Numero di Film")
    ax.set_xlabel("Categoria di età")

    # Salvataggio del grafico
    plt.savefig("plots/categoria_eta_film.jpg")

    # Visualizzazione del grafico
    plt.show()

    # Messaggio di conferma
    print(f"\nGrafico della classificazione dei Film per target di età stampato.\n")


def plot_product_genres(dataframe):
    """
    Genera un grafico a barre per confrontare la distribuzione dei generi tra Serie TV e Film.

    :param dataframe: DataFrame contenente le colonne 'type' e 'listed_in'.
    :return:
    """
    # Estrazione e conteggio dei generi per Serie TV
    genere_serie_tv = (dataframe[dataframe['type'] == 'TV Show']['listed_in']
                       .str.split(', ', expand=True).stack().value_counts())

    # Estrazione e conteggio dei generi per Film
    genere_film = (dataframe[dataframe['type'] == 'Movie']['listed_in']
                   .str.split(', ', expand=True).stack().value_counts())

    # Creazione di un DataFrame combinato con i generi di entrambi i tipi di prodotto
    unione_generi = pd.concat([genere_serie_tv, genere_film], axis=1)
    unione_generi.columns = ['Genere Serie TV', 'Genere Film']
    unione_generi.fillna(0, inplace=True)

    # Trasformazione del DataFrame per facilitare la visualizzazione
    unione_generi = unione_generi.reset_index().melt(id_vars='index', var_name='Genre Type', value_name='Count')

    # Creazione del grafico a barre
    plt.figure(figsize=(12, 7))
    plt.subplots_adjust(left=0.3)
    seaborn.barplot(x='Count',
                    y='index',
                    hue='Genre Type',
                    data=unione_generi,
                    palette=['red', 'blue'])

    # Configurazione del grafico
    plt.title('Differenze sui generi tra Serie TV e Film')
    plt.xlabel('Quantità')
    plt.ylabel('Generi')
    plt.legend(title="Tipologia di prodotto")

    # Salvataggio e visualizzazione del grafico
    plt.savefig("plots/generi_separati.jpg")
    plt.show()

    # Messaggio di conferma
    print(f"\nGrafico della suddivisione di Film e Serie TV per generi stampato.\n")


def plot_unified_product_genres(dataframe):
    """
    Genera un grafico a barre per visualizzare la distribuzione dei generi nell'intero dataset.

    :param dataframe: DataFrame contenente la colonna 'listed_in' con i generi dei prodotti.
    :return:
    """
    # Estrazione e conteggio dei generi complessivi
    genere = dataframe['listed_in'].str.split(', ', expand=True).stack().value_counts()

    # Creazione del DataFrame con i generi e la loro frequenza
    genere_dataframe = genere.reset_index()
    genere_dataframe.columns = ['Genere', 'Count']

    # Creazione del grafico a barre
    plt.figure(figsize=(12, 7))
    seaborn.barplot(y='Genere', x='Count', hue='Genere', data=genere_dataframe, palette='Reds_r', legend=False)

    # Configurazione del grafico
    plt.title('Generi più presenti:')
    plt.xlabel('Quantità')
    plt.ylabel('Generi')

    # Salvataggio e visualizzazione del grafico
    plt.savefig("plots/generi_unificati.jpg")
    plt.show()

    # Messaggio di conferma
    print("\nGrafico della distribuzione dei Generi dell'intero dataset stampato.\n")


def bar_plot_categories(dataframe):
    """
    Genera un grafico a barre per visualizzare la distribuzione delle categorie di età nell'intero dataset.

    :param dataframe: DataFrame contenente la colonna 'Categoria' con le classificazioni per età.
    :return:
    """
    # Creazione della figura
    plt.figure(figsize=(10, 7.5))

    # Grafico a barre con il conteggio delle categorie di età
    ax = seaborn.countplot(x="Categoria",
                           data=dataframe,
                           hue="Categoria",
                           palette="Set1",
                           order=dataframe['Categoria'].value_counts().index[0:12],
                           legend=False)

    # Configurazione del grafico
    ax.set_title(f"Divisione per target di età dell'intero dataset")
    ax.set_ylabel(f"Numero di Prodotti")
    ax.set_xlabel("Categoria di età")

    # Salvataggio e visualizzazione del grafico
    plt.savefig("plots/categoria_eta_dataset.jpg")
    plt.show()

    # Messaggio di conferma
    print(f"\nGrafico della classificazione dell'intero dataset per target di età stampato.\n")
