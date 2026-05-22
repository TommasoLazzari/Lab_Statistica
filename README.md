# Anomaly Detection su Grafi Eterogenei per la Pubblica  Amministrazione Italiana

#### Università degli Studi di Padova - RE-ACT
#### Gomiero Guido - Bortolato Alberto - Lazzari Tommaso

## Informazioni sul progetto

Questo progetto è stato sviluppato nell’ambito di un bando promosso dall’Università degli Studi di Padova, in collaborazione con RE-ACT. Contiene una pipeline completa in Python che parte dalla pulizia dei dati aperti dell'ANAC (Autorità Nazionale Anticorruzione), passando per il preprocessing e la costruzione di un grafo eterogeneo, fino all'addestramento di un modello di **Graph Neural Network (GNN)** per l'Anomaly Detection. 

Include anche tool per l'estrazione e l'interpretazione umana delle anomalie rilevate e due dashboard esplorative.

---

## Struttura della Directory

Non è necessario creare manualmente le cartelle prima dell'esecuzione: gli script genereranno automaticamente l'alberatura necessaria man mano che elaborano i dati. La struttura finale sarà la seguente:

```text
.
├── data/
│   ├── raw/                 # File CSV grezzi (es. CIG, aggiudicatari, ecc.)
│   └── processed/           # File ripuliti e ottimizzati (formati .parquet e .json)
├── models/                  # File di salvataggio del modello:
│   │                        #  - PA_graph.pt (Grafo eterogeneo PyTorch Geometric)
│   │                        #  - level4_minibatch_outputs.pt (Pesi del modello GNN)
│   │                        #  - scaler_params.json (Parametri di standardizzazione)
├── reports/                 # Output dell'Anomaly Detection
│   ├── level4_results/      # Score tabulari delle anomalie (.csv) e history di training
│   └── interpretazioni/     # Dettagli testuali (.txt) ed Excel (.xlsx) per le top anomalie
├── scripts/                 # Codice sorgente della pipeline
│   ├── 1_DataPreprocessing.py
│   ├── 2_HeteroGraph.py
│   ├── 3_AnomalyDetection.py
│   ├── 4_Interpreta_Top_Anomalyes.py
│   ├── 5_Dashboard_Grafo.py
│   └── 6_Dashboard_Anomalie.py
├── docs/
│   ├── Bando.pdf
│   ├── Report.pdf                # Relazione finale del progetto
│   └── Report_appendice.pdf      # Rapporto tecnico sul processo di pulizia dei dati
└── README.md
```

---

## Requisiti

Per eseguire l'intero progetto, assicurati di avere installato un ambiente Python 3.x con le seguenti librerie:

```bash
pip install pandas numpy requests scipy openpyxl pyarrow fastparquet
pip install torch torchvision torchaudio  # Scegli la versione corretta per il tuo OS/CUDA
pip install torch_geometric
pip install streamlit
```
*(Nota: L'engine `pyarrow` o `fastparquet` è necessario per la gestione dei file `.parquet` di pandas; `openpyxl` è richiesto per esportare l'Excel delle interpretazioni).*

---

## Istruzioni per l'Esecuzione

Per garantire il corretto funzionamento, è necessario:

- popolare la cartella LabStatistica/data/raw con i file contenuti nella cartella Drive raggiungibile al seguente link: https://drive.google.com/drive/folders/1ZAqTPAGTKu3UOiHKoHtLwQ2FAvN394Un?usp=sharing

- eseguire gli script Python **in ordine sequenziale**. In particolare, avvia un terminale nella cartella root del progetto (`Lab_Statistica`) e lancia:

**1. Esecuzione pipeline base:**
```bash
python scripts/1_DataPreprocessing.py
python scripts/2_HeteroGraph.py
python scripts/3_AnomalyDetection.py
python scripts/4_Interpreta_Top_Anomalyes.py
```

**2. Esecuzione delle Dashboard Interattive (Streamlit):**
```bash
# Per esplorare i dati pre-processati / grafo:
streamlit run scripts/5_Dashboard_Grafo.py

# Per visualizzare in modo interattivo le anomalie calcolate:
streamlit run scripts/6_Dashboard_Anomalie.py
```

---

## Spiegazione degli Script

Qui trovi nel dettaglio cosa fa ogni singolo file all'interno della cartella `scripts`:

### `1_DataPreprocessing.py`
Carica i file CSV da `data/raw` e li ripulisce per poter essere processati. Le operazioni includono l'eliminazione dei duplicati, il casting dei tipi di dato appropriati, l'imputazione dei dati mancanti e la standardizzazione delle feature continue. Salva il tutto in formato compresso `.parquet` e memorizza dizionari di mappatura chiavi in `.json` all'interno della cartella `data/processed`.

### `2_HeteroGraph.py`
Prende i dati pre-processati e li usa per comporre un grafo eterogeneo utilizzando la libreria **PyTorch Geometric**. Crea 4 tipologie di nodi (`SA`, `CIG`, `AGZ`, `AG`) e stabilisce gli archi direzionati basati sulle entità di partecipazione agli appalti. Salva il tensore dati strutturato come `PA_graph.pt` in `models`.

### `3_AnomalyDetection.py`
È il core algoritmico del progetto. Definisce un modello GNN composto da *HeteroEncoder*, *AttributeDecoder* e *RelationDecoder*. Lo script esegue il training in modalità *mini-batch* utilizzando campionamenti per via dell'alta mole di dati. Calcola le loss strutturali ed attributive, sfrutta l'early stopping e valuta tutto il grafo in fase di inferenza per stilare un ranking di anomalia (salvato poi nei `reports/level4_results/`).

### `4_Interpreta_Top_Anomalyes.py`
Legge le top anomalie (rank) e gli ID anonimizzati dal report del modello e li incrocia con i dizionari `JSON` originari. In questo modo riconduce l'ID di un nodo anomalo al rispettivo "Codice Fiscale", "Denominazione Impresa", o "Codice CIG" in chiaro, offrendo al contempo lo specchietto interpretativo di quali feature (attribuzionali o relazionali) siano state artefici dell'anomalia rilevata. L'output testuale e .xlsx si troverà in `reports/interpretazioni`.

### `5_Dashboard_Grafo.py`
Una dashboard sviluppata con Streamlit progettata per fornire statistiche esplorative del database (distribuzione dei CIG, informazioni grafiche del dataset).

### `6_Dashboard_Anomalie.py`
Un'altra applicazione Streamlit volta alla navigazione e visualizzazione immediata dei risultati dell'Anomaly Detection. Mostra in maniera user-friendly quali enti o appalti hanno ottenuto gli anomaly score più elevati e le logiche ricostruite dal modello.

---

Aprile, 2026
