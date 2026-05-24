# Anomaly Detection on Heterogeneous Graphs for the Italian Public Administration

#### Università di Padova - RE-ACT
#### Gomiero Guido - Bortolato Alberto - Lazzari Tommaso

## Project Information

This project was developed within a research initiative promoted by the University of Padua, in collaboration with RE-ACT. It contains a complete Python pipeline starting from the cleaning of ANAC (Italian National Anti-Corruption Authority) open data, through preprocessing and heterogeneous graph construction, up to the training of a **Graph Neural Network (GNN)** model for Anomaly Detection.

The project also includes tools for anomaly extraction and human interpretation, together with two exploratory dashboards.

> **Note:** The original project was entirely developed in Italian. Therefore, the original technical reports and supporting documentation are available in Italian, while the English versions currently included in the repository are translations of the original documents.

---

## Directory Structure

It is not necessary to manually create the folders before execution: the scripts will automatically generate the required directory tree during data processing. The final structure will be the following:

```text
.
├── data/
│   ├── raw/                 # Raw CSV files (e.g. CIG, awardees, etc.)
│   └── processed/           # Cleaned and optimized files (.parquet and .json formats)
├── models/                  # Saved model files:
│   │                        #  - PA_graph.pt (PyTorch Geometric heterogeneous graph)
│   │                        #  - level4_minibatch_outputs.pt (GNN model weights)
│   │                        #  - scaler_params.json (standardization parameters)
├── reports/                 # Anomaly Detection outputs
│   ├── level4_results/      # Tabular anomaly scores (.csv) and training history
│   └── interpretations/     # Text (.txt) and Excel (.xlsx) details for top anomalies
├── scripts/                 # Pipeline source code
│   ├── 1_DataPreprocessing.py
│   ├── 2_HeteroGraph.py
│   ├── 3_AnomalyDetection.py
│   ├── 4_Interpreta_Top_Anomalyes.py
│   ├── 5_Dashboard_Grafo.py
│   └── 6_Dashboard_Anomalie.py
├── docs/
│   ├── Bando.pdf
│   ├── Report.pdf                     # Original final project report (Italian)
│   ├── Report_appendice.pdf           # Original technical appendix (Italian)
│   ├── Report_ENG.pdf                 # English translation of the final report
│   └── Report_appendice_ENG.pdf       # English translation of the technical appendix
└── README.md
```

---

## Requirements

To run the entire project, make sure you have a Python 3.x environment with the following libraries installed:

```bash
pip install pandas numpy requests scipy openpyxl pyarrow fastparquet
pip install torch torchvision torchaudio  # Choose the correct version for your OS/CUDA
pip install torch_geometric
pip install streamlit
```

*(Note: the `pyarrow` or `fastparquet` engine is required for handling pandas `.parquet` files; `openpyxl` is required for exporting Excel interpretation files).*

---

## Execution Instructions

To ensure correct execution, it is necessary to:

- populate the `LabStatistica/data/raw` folder with the files contained in the Drive folder available at the following link:
  https://drive.google.com/drive/folders/1ZAqTPAGTKu3UOiHKoHtLwQ2FAvN394Un?usp=sharing

- execute the Python scripts **in sequential order**. In particular, open a terminal in the project root folder (`Lab_Statistica`) and run:

### 1. Base Pipeline Execution

```bash
python scripts/1_DataPreprocessing.py
python scripts/2_HeteroGraph.py
python scripts/3_AnomalyDetection.py
python scripts/4_Interpreta_Top_Anomalyes.py
```

### 2. Interactive Dashboard Execution (Streamlit)

```bash
# To explore the preprocessed data / graph:
streamlit run scripts/5_Dashboard_Grafo.py

# To interactively visualize the detected anomalies:
streamlit run scripts/6_Dashboard_Anomalie.py
```

---

## Script Description

Below is a detailed explanation of each file inside the `scripts` folder:

### `1_DataPreprocessing.py`

Loads the CSV files from `data/raw` and cleans them for processing. Operations include duplicate removal, data type casting, missing value imputation, and standardization of continuous features. The processed data is saved in compressed `.parquet` format, while key mapping dictionaries are stored as `.json` files inside the `data/processed` directory.

### `2_HeteroGraph.py`

Takes the preprocessed data and constructs a heterogeneous graph using the **PyTorch Geometric** library. It creates 4 node types (`SA`, `CIG`, `AGZ`, `AG`) and establishes directed edges based on procurement participation relationships. The resulting graph tensor is saved as `PA_graph.pt` inside `models`.

### `3_AnomalyDetection.py`

This is the algorithmic core of the project. It defines a GNN model composed of *HeteroEncoder*, *AttributeDecoder*, and *RelationDecoder*. The script performs mini-batch training using graph sampling techniques due to the large dataset size. It computes structural and attributive losses, applies early stopping, and evaluates the full graph during inference to generate an anomaly ranking (saved in `reports/level4_results/`).

### `4_Interpreta_Top_Anomalyes.py`

Reads the top-ranked anomalies and anonymized IDs from the model output and matches them with the original JSON dictionaries. This allows anomalous node IDs to be mapped back to readable identifiers such as Tax Codes, Company Names, or CIG Codes. It also provides interpretative summaries indicating which features (attributive or relational) contributed most to the anomaly. The textual and `.xlsx` outputs are saved in `reports/interpretations`.

### `5_Dashboard_Grafo.py`

A Streamlit dashboard designed to provide exploratory statistics about the dataset and the heterogeneous graph structure.

### `6_Dashboard_Anomalie.py`

Another Streamlit application dedicated to browsing and visualizing anomaly detection results. It provides a user-friendly interface to inspect entities and procurement procedures with the highest anomaly scores and the reconstructed logic behind them.

---

April, 2026
