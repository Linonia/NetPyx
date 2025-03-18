import matplotlib.pyplot as plt
import pandas as pd
import seaborn
from scipy.stats import skew, kurtosis


def general_informations(df):
    print(f"\nVerifica dell'avvenuto caricamento del dataset tramite la stampa delle prime 5 righe:\n{df.head(5)}")
    print(f"\nNumero di righe e colonne presenti nel dataset: {df.shape}")
    print(f"\nAltre informazioni generali presenti nel dataset:\n")
    df.info()


def bar_suddivisione_prodotti(df):
    """
    Genera un grafico a barre che mostra la suddivisione tra Film e Serie TV nel files.

    :param df: df di Pandas contenente il files da analizzare.
    :return: None
    """
    print(f"\nStampa del numero di Film e Serie TV a confronto eseguita.\n")
    df['type'].value_counts().iloc[::-1].plot(kind='bar',
                                          title="Numero di Film (Movie) e Serie TV (TV Show)",
                                          figsize=(8, 6),
                                          color=['red', 'blue'])
    plt.xticks(rotation=0)
    plt.xlabel("Tipo")
    plt.ylabel("Numero di prodotti")
    plt.show()


def bar_plot_tv_show(df):
    """
    Genera un grafico a barre che mostra la classificazione delle Serie TV in base al target di età.

    :param df: df di Pandas contenente il files da analizzare.
    :return: None
    """
    print(f"\nStampa della classificazione delle Serie TV per target di età eseguita.\n")
    df_selection = df[df['type'] == "TV Show"]
    plt.figure(figsize=(10, 7.5))
    ax = seaborn.countplot(x="rating",
                           data=df_selection,
                           hue="rating",
                           palette="Set1",
                           order=df_selection['rating'].value_counts().index[0:12],
                           legend=False)
    ax.set_title(f"Serie TV - Divisione per target di età")
    ax.set_ylabel(f"Numero di Serie TV")
    ax.set_xlabel("Categoria di età")
    plt.show()


def bar_plot_movie(df):
    """
    Genera un grafico a barre che mostra la classificazione dei Film in base al target di età.

    :param df: df di Pandas contenente il files da analizzare.
    :return: None
    """
    print(f"\nStampa della classificazione dei Film per target di età eseguita.\n")
    df_selection = df[df['type'] == "Movie"]
    plt.figure(figsize=(10, 7.5))
    ax = seaborn.countplot(x="rating",
                           data=df_selection,
                           hue="rating",
                           palette="Set1",
                           order=df_selection['rating'].value_counts().index[0:12],
                           legend=False)
    ax.set_title(f"Film - Divisione per target di età")
    ax.set_ylabel(f"Numero di Film")
    ax.set_xlabel("Categoria di età")
    plt.show()


def plot_generi_prodotti(df):
    """
    Genera un grafico a barre che mostra la suddivisione di Film e Serie TV in base ai generi.

    :param df: df di Pandas contenente il files da analizzare.
    :return: None
    """
    print(f"\nStampa della suddivisione di Film e Serie TV per generi eseguita.\n")
    genere_serie_tv = (df[df['type'] == 'TV Show']['listed_in']
                       .str.split(', ', expand=True).stack().value_counts())
    genere_film = (df[df['type'] == 'Movie']['listed_in']
                   .str.split(', ', expand=True).stack().value_counts())
    unione_generi = pd.concat([genere_serie_tv, genere_film], axis=1)
    unione_generi.columns = ['Genere Serie TV', 'Genere Film']
    unione_generi.fillna(0, inplace=True)
    unione_generi = unione_generi.reset_index().melt(id_vars='index', var_name='Genre Type', value_name='Count')
    plt.figure(figsize=(12, 7))
    plt.subplots_adjust(left=0.3)
    seaborn.barplot(x='Count',
                    y='index',
                    hue='Genre Type',
                    data=unione_generi,
                    palette=['red', 'blue'])
    plt.title('Differenze sui generi tra Serie TV e Film')
    plt.xlabel('Quantità')
    plt.ylabel('Generi')
    plt.legend(title="Tipologia di prodotto")
    plt.show()


def plot_generi_prodotti_unificati(df):
    print("\nStampa della suddivisione dei generi eseguita.\n")
    genere_totale = df['listed_in'].str.split(', ', expand=True).stack().value_counts()
    genere_totale = genere_totale.reset_index()
    genere_totale.columns = ['Genere', 'Count']

    plt.figure(figsize=(12, 7))
    plt.subplots_adjust(left=0.3)
    seaborn.barplot(x='Count', y='Genere', data=genere_totale, palette='coolwarm')
    plt.title('Distribuzione dei Generi')
    plt.xlabel('Quantità')
    plt.ylabel('Generi')
    plt.show()
