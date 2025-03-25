import pandas as pd
import utils.eda as eda
import utils.preprocessing as preprocessing
import utils.addestramento_word2vec as w2v
import utils.addestramento_supervisionato as add_sup


# Attivazione/disattivazione stampe dettagliate
stampe = False

# ==============================================================================
# IMPORT DEL DATASET
# ==============================================================================

print("\n\n==================================================")
print("              IMPORTAZIONE DEL DATASET             ")
print("==================================================\n")

print("[INFO] Importazione del dataset...\n")
dataframe = pd.read_csv("dataset/netflix_titles.csv")
print("[OK] Dataset importato.\n")

# ==============================================================================
# EDA - ANALISI ESPLORATIVA DEI DATI
# ==============================================================================

print("\n\n==================================================")
print("        ANALISI ESPLORATIVA DEL DATASET (EDA)      ")
print("==================================================\n")

print("[INFO] Avvio analisi esplorativa del dataset...\n")
eda.general_informations(dataframe)

if stampe:
    print("\n[INFO] Generazione dei grafici di analisi esplorativa...\n")
    eda.bar_product_subdivision(dataframe)
    eda.bar_plot_series(dataframe)
    eda.bar_plot_movie(dataframe)
    eda.plot_product_genres(dataframe)
    print("[OK] Grafici EDA generati con successo.\n")

# ==============================================================================
# PREPROCESSING
# ==============================================================================

print("\n\n==================================================")
print("                PREPROCESSING DATI                 ")
print("==================================================\n")

print("[INFO] Avvio preprocessing del dataset...\n")

# Unificazione generi
print("[INFO] Unificazione dei generi in corso...\n")
dataframe = preprocessing.unify_genres(dataframe)
print("[OK] Generi unificati.\n")

if stampe:
    print("[INFO] Generazione del grafico di unificazione dei generi...\n")
    eda.plot_unified_product_genres(dataframe)
    print("[OK] Grafico generato con successo.\n")

# Rinomina features
print("\n[INFO] Rinomina delle colonne in corso...\n")
dataframe = preprocessing.rename_features(dataframe)
print("[OK] Rinomina completata.\n")
dataframe.info()

# Eliminazione di features non utili al programma
print("\n[INFO] Eliminazione delle features non utili per il programma...\n")
dataframe = preprocessing.delete_features(dataframe)
print("[OK] Eliminazione completata.\n")


# Modifica valori tipo
print("\n[INFO] Mappatura dei valori di tipo del prodotto in corso...\n")
dataframe = preprocessing.type_permutation(dataframe)
print("[OK] Tipologia di prodotto mappata.\n")

# Mapping categorie età
print("\n[INFO] Mappatura categorie di età...\n")
dataframe['Categoria'] = dataframe['Categoria'].apply(preprocessing.map_rating)
print("[OK] Categorie di età mappate.\n")

if stampe:
    print("[INFO] Generazione del grafico di distribuzione delle categorie di età...\n")
    eda.bar_plot_categories(dataframe)
    print("[OK] Grafico generato con successo.\n")

# Gestione valori nulli
print("\n[INFO] Identificazione e gestione dei valori nulli...\n")
preprocessing.find_null_values(dataframe)
dataframe = preprocessing.manage_null_values(dataframe)
preprocessing.find_null_values(dataframe)
print("[OK] Valori nulli gestiti correttamente.\n")

# Rimozione duplicati
print("\n[INFO] Rimozione duplicati...\n")
dataframe = preprocessing.remove_duplicates(dataframe)
print("[OK] Duplicati rimossi.\n")

# Permutazione generi in vettori numerici
print("\n[INFO] Conversione generi in vettori numerici...\n")
dataframe = preprocessing.create_genres_vector(dataframe)
print("[OK] Conversione completata.\n")

print("\n\n[OK] Preprocessing completato. Dataset finale:\n")
dataframe.info()

# Creazione copia dataset per addestramento non supervisionato
non_sup_dataframe = dataframe.copy()

# ==============================================================================
# APPRENDIMENTO SUPERVISIONATO - TEST
# ==============================================================================

print("\n\n==================================================")
print("        TEST APPRENDIMENTO SUPERVISIONATO          ")
print("==================================================\n")

print("[INFO] Avvio test apprendimento supervisionato...\n")
add_sup.simulate_testing_sup_train(dataframe, stampe=stampe)
print("[OK] Test completato.\n")

# ==============================================================================
# APPRENDIMENTO NON SUPERVISIONATO - TEST
# ==============================================================================

print("\n\n==================================================")
print("      TEST APPRENDIMENTO NON SUPERVISIONATO        ")
print("==================================================\n")

print("[INFO] Avvio test apprendimento non supervisionato...\n")
no_sup_model = w2v.simulate_testing_non_sup_train(dataframe, stampe=stampe)
print("[OK] Test completato.\n")

# ==============================================================================
# INTERAZIONE UTENTE
# ==============================================================================

while True:
    print("\n\n==================================================")
    print("                    MENU PRINCIPALE                ")
    print("==================================================\n")
    print("1 - Cercare film per parole chiave (Apprendimento non supervisionato)")
    print("2 - Cercare film per preferenze personali (Apprendimento supervisionato)")
    print("3 - Uscire")
    print("==================================================\n")

    scelta = input("Inserire il numero dell'opzione desiderata: ").strip()

    if scelta == "1":
        print("\n[INFO] Avvio ricerca per parole chiave...")
        w2v.search_movies_by_user_input(non_sup_dataframe, no_sup_model, stampe=stampe)
    elif scelta == "2":
        print("\n[INFO] Avvio ricerca basata sulle preferenze personali...\n")
        add_sup.user_testing_sup_train(dataframe, stampe=stampe)
    elif scelta == "3":
        print("\n[INFO] Uscita dal programma...\n")
        break
    else:
        print("\n[ERRORE] Scelta non valida. Inserire un numero tra 1 e 3.\n")
