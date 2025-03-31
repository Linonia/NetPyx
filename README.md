# NetPyx

Repository del progetto di Ingegneria della Conoscenza, A.A. 2023/2024.

Gruppo di lavoro:

- Michele Grieco (757059, [m.grieco31@studenti.uniba.it](mailto:m.grieco31@studenti.uniba.it))
- Daniele Gentile (758352, [d.gentile32@studenti.uniba.it](mailto:d.gentile32@studenti.uniba.it))

## Descrizione del Progetto

Questo progetto è un sistema di raccomandazione di film che utilizza tecniche di apprendimento supervisionato e non supervisionato per suggerire film in base alle preferenze degli utenti e alle parole chiave inserite. Il sistema sfrutta librerie di machine learning e NLP per preprocessare i dati, addestrare i modelli e generare raccomandazioni.

## Indice

- [Installazione](#installazione)
- [Utilizzo](#utilizzo)
- [Struttura del Progetto](#struttura-del-progetto)
- [Moduli](#moduli)

## Installazione

Per installare le dipendenze richieste, eseguire:
```sh
pip install -r requirements.txt
```

Se non funziona pip, ma è installato, provare come segue:
```sh
python -m pip install -r requirements.txt
```


## Utilizzo

Per avviare il programma principale, eseguire:

```sh
python main.py
```

Il programma guida l'utente nell'importazione del dataset, l'analisi esplorativa (EDA), la pre-elaborazione dei dati e l'addestramento dei modelli. Successivamente, permette di ottenere raccomandazioni basate sulle preferenze o parole chiave inserite.

## Struttura del Progetto

```
├── dataset/
│   └── netflix_titles.csv
├── plots/
│   └── (grafici generati)
├── utils/
│   ├── addestramento_word2vec.py
│   ├── addestramento_supervisionato.py
│   ├── eda.py
│   └── preprocessing.py
├── main.py
└── requirements.txt
```

## Moduli

- utils/addestramento_word2vec.py: Addestra un modello Word2Vec sulle descrizioni dei film e genera suggerimenti basati su parole chiave.
- utils/addestramento_supervisionato.py: Prepara il dataset, raccoglie valutazioni utenti e addestra un modello supervisionato per raccomandazioni personalizzate.
- utils/eda.py: Analizza il dataset con statistiche descrittive e visualizzazioni.
- utils/preprocessing.py: Pre-elabora i dati, pulisce generi, gestisce valori nulli e trasforma le informazioni in vettori numerici.
- main.py: Coordina il flusso del sistema, dall'importazione dei dati alla generazione delle raccomandazioni.
