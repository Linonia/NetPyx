import matplotlib.pyplot as plt
import pandas as pd
import seaborn


def general_informations(df):
    numero_righe = 5
    print(f"\nVerifica dell'avvenuto caricamento del dataset tramite la stampa"
          f" delle prime {numero_righe} righe:\n\n{df.head(numero_righe)}\n")
    print(f"Numero di righe e colonne presenti nel dataset: {df.shape}\n")
    print(f"Altre informazioni generali presenti nel dataset:")
    df.info()


def bar_product_subdivision(df):
    df['type'].value_counts().iloc[::-1].plot(kind='bar',
                                          title="Numero di Film (Movie) e Serie TV (TV Show)",
                                          figsize=(8, 6),
                                          color=['red', 'blue'])
    plt.xticks(rotation=0)
    plt.xlabel("Tipo")
    plt.ylabel("Numero di prodotti")
    plt.savefig("plots/suddivisione_prodotti.jpg")
    plt.show()
    print(f"\nGrafico del numero di Film e Serie TV a confronto stampato.\n")


def bar_plot_series(df):
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
    plt.savefig("plots/categoria_eta_serie.jpg")
    plt.show()
    print(f"\nGrafico della classificazione delle Serie TV per target di età stampato.\n")


def bar_plot_movie(df):
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
    plt.savefig("plots/categoria_eta_film.jpg")
    plt.show()
    print(f"\nGrafico della classificazione dei Film per target di età stampato.\n")


def plot_product_genres(df):
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
    plt.savefig("plots/generi_separati.jpg")
    plt.show()
    print(f"\nGrafico della suddivisione di Film e Serie TV per generi stampato.\n")


def plot_unified_product_genres(df):
    genere = df['listed_in'].str.split(', ', expand=True).stack().value_counts()
    genere_df = genere.reset_index()
    genere_df.columns = ['Genere', 'Count']
    plt.figure(figsize=(12, 7))
    seaborn.barplot(y='Genere', x='Count', hue='Genere', data=genere_df, palette='Reds_r', legend=False)
    plt.title('Generi più presenti:')
    plt.xlabel('Quantità')
    plt.ylabel('Generi')
    plt.savefig("plots/generi_unificati.jpg")
    plt.show()
    print("\nGrafico della distribuzione dei Generi dell'intero dataset stampato.\n")


def bar_plot_categories(df):
    plt.figure(figsize=(10, 7.5))
    ax = seaborn.countplot(x="Categoria",
                           data=df,
                           hue="Categoria",
                           palette="Set1",
                           order=df['Categoria'].value_counts().index[0:12],
                           legend=False)
    ax.set_title(f"Divisione per target di età dell'intero dataset")
    ax.set_ylabel(f"Numero di Prodotti")
    ax.set_xlabel("Categoria di età")
    plt.savefig("plots/categoria_eta_dataset.jpg")
    plt.show()
    print(f"\nGrafico della classificazione dell'intero dataset per target di età stampato.\n")
