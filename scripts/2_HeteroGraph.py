import pandas as pd
import numpy as np
import torch
import json
import gc
import os
from pathlib import Path
from torch_geometric.data import HeteroData

# --- DEFINIZIONE PERCORSI RELATIVI ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

# Assicura caricamento da processed e salvataggio in models
MODELS_DIR.mkdir(parents=True, exist_ok=True)

stazioni_appaltanti = pd.read_parquet(DATA_PROCESSED_DIR / "stazioni_appaltanti_v2.parquet")
print("CARICAMENTO [1/13]: file stazioni_appaltanti_v2.parquet caricato con successo")

cig = pd.read_parquet(DATA_PROCESSED_DIR / "CIG_v2.parquet")
print("CARICAMENTO [2/13]: file CIG_v2.parquet caricato con successo")

aggiudicazioni = pd.read_parquet(DATA_PROCESSED_DIR / "aggiudicazioni_v2.parquet")
print("CARICAMENTO [3/13]: file aggiudicazioni_v2.parquet caricato con successo")

aggiudicatari = pd.read_parquet(DATA_PROCESSED_DIR / "aggiudicatari_v2.parquet")
print("CARICAMENTO [4/13]: file aggiudicatari_v2.parquet caricato con successo")

aggiudicatari_links = pd.read_parquet(DATA_PROCESSED_DIR / "aggiudicatari_links.parquet")
print("CARICAMENTO [5/13]: file aggiudicatari_links.parquet caricato con successo")

with open(DATA_PROCESSED_DIR / "SA_id.json", "r") as f:
    SA_id = json.load(f)
SA_id = {int(k): v for k, v in SA_id.items()}
print("CARICAMENTO [6/13]: file SA_id.json caricato con successo")

with open(DATA_PROCESSED_DIR / "SA_ca.json", "r") as f:
    SA_ca = json.load(f)
print("CARICAMENTO [7/13]: file SA_ca.json caricato con successo")

with open(DATA_PROCESSED_DIR / "CIG_id.json", "r") as f:
    CIG_id = json.load(f)
CIG_id = {int(k): v for k, v in CIG_id.items()}
print("CARICAMENTO [8/13]: file CIG_id.json caricato con successo")

with open(DATA_PROCESSED_DIR / "CIG_cig.json", "r") as f:
    CIG_cig = json.load(f)
print("CARICAMENTO [9/13]: file CIG_cig.json caricato con successo")

with open(DATA_PROCESSED_DIR / "AGZ_id.json", "r") as f:
    AGZ_id = json.load(f)
AGZ_id = {int(k): v for k, v in AGZ_id.items()}
print("CARICAMENTO [10/13]: file AGZ_id.json caricato con successo")

with open(DATA_PROCESSED_DIR / "AGZ_idag.json", "r") as f:
    AGZ_idag = json.load(f)
print("CARICAMENTO [11/13]: file AGZ_idag.json caricato con successo")

with open(DATA_PROCESSED_DIR / "AG_id.json", "r") as f:
    AG_id = json.load(f)
AG_id = {int(k): v for k, v in AG_id.items()}
print("CARICAMENTO [12/13]: file AG_id.json caricato con successo")

with open(DATA_PROCESSED_DIR / "AG_cf.json", "r") as f:
    AG_cf = json.load(f)
print("CARICAMENTO [13/13]: file AG_cf.json caricato con successo")

del AG_id, SA_id
print("----- HETERO GRAPH -----")
#####################################################################################################################
##NODI         ######################################################################################################
#####################################################################################################################

# SA nodes
nodeA = stazioni_appaltanti.reset_index(drop=True).copy()
del stazioni_appaltanti
gc.collect()
nodeA_x_df = nodeA.drop(columns=["id"], errors="ignore")
nodeA_x = torch.tensor(nodeA_x_df.values, dtype=torch.float)
del nodeA_x_df
gc.collect()
print("--- Nodi di tipo SA [1/4] creati con successo")

# CIG nodes
nodeB = cig.reset_index(drop=True).copy()
del cig
gc.collect()
nodeB_x_df = nodeB.drop(columns=["id"], errors="ignore")
nodeB_x = torch.tensor(nodeB_x_df.values, dtype=torch.float)
del nodeB_x_df
gc.collect()
print("--- Nodi di tipo CIG [2/4] creati con successo")

# AGZ nodes
nodeC = aggiudicazioni.reset_index(drop=True).copy()
del aggiudicazioni
gc.collect()
nodeC_x_df = nodeC.drop(columns=["id"], errors="ignore")
nodeC_x = torch.tensor(nodeC_x_df.values, dtype=torch.float)
del nodeC_x_df
gc.collect()
print("--- Nodi di tipo AGZ [3/4] creati con successo")

# AG nodes
nodeD = aggiudicatari.reset_index(drop=True).copy()
del aggiudicatari
gc.collect()
nodeD_x_df = nodeD.drop(columns=["id"], errors="ignore")
nodeD_x = torch.tensor(nodeD_x_df.values, dtype=torch.float)
del nodeD_x_df
gc.collect()
print("--- Nodi di tipo AG [4/4] creati con successo")

#####################################################################################################################
##ARCHI         #####################################################################################################
#####################################################################################################################

def build_edge_index(src_list, dst_list, dtype=torch.long):
    return torch.tensor([src_list, dst_list], dtype=dtype)

# Archi SA_CIG : stazioni appaltanti -> bandi pubblici
src_sa_cig = []
dst_sa_cig = []

for cig_id, meta in CIG_id.items():
    codice_ausa = meta["codice_ausa"]
    if codice_ausa is not None and codice_ausa in SA_ca:
        src_sa_cig.append(int(SA_ca[codice_ausa]))
        dst_sa_cig.append(int(cig_id))

edge_sa_cig = build_edge_index(src_sa_cig, dst_sa_cig)
print("--- Archi [1/4]: SA -> CIG creati con successo")

del src_sa_cig, dst_sa_cig
del SA_ca
gc.collect()


# ArchiBC : bandi pubblici -> aggiudicazioni
src_cig_agz = []
dst_cig_agz = []

for agz_id, meta in AGZ_id.items():
    cig_code = meta["cig"]
    if cig_code is not None and cig_code in CIG_cig:
        src_cig_agz.append(int(CIG_cig[cig_code]))
        dst_cig_agz.append(int(agz_id))

edge_cig_agz = build_edge_index(src_cig_agz, dst_cig_agz)
print("--- Archi [2/4]: CIG -> AGZ creati con successo")

del agz_id
del src_cig_agz, dst_cig_agz
gc.collect()


# ArchiCD : aggiudicazioni -> aggiudicatari
src_agz_ag = []
dst_agz_ag = []

for row in aggiudicatari_links.itertuples(index=False):
    key = f"{row.id_aggiudicazione}_{row.cig}"

    if key in AGZ_idag and row.codice_fiscale in AG_cf:
        src_agz_ag.append(int(AGZ_idag[key]))
        dst_agz_ag.append(int(AG_cf[row.codice_fiscale]))

edge_pairs = sorted(set(zip(src_agz_ag, dst_agz_ag)))
src_agz_ag = [p[0] for p in edge_pairs]
dst_agz_ag = [p[1] for p in edge_pairs]

edge_agz_ag = build_edge_index(src_agz_ag, dst_agz_ag)
print("--- Archi [3/4]: AGZ -> AG creati con successo")

del src_agz_ag, dst_agz_ag
del edge_pairs
del aggiudicatari_links
del AGZ_idag
del AG_cf
gc.collect()


# ArchiBB : bandi pubblici -> bandi pubblici
src_cig_cig = []
dst_cig_cig = []

for cig_id, meta in CIG_id.items():
    linked_cig = meta["CIG_COLLEGAMENTO"]

    if linked_cig is not None and linked_cig in CIG_cig:
        linked_id = int(CIG_cig[linked_cig])

        if linked_id != int(cig_id):
            src_cig_cig.append(int(cig_id))
            dst_cig_cig.append(linked_id)
            src_cig_cig.append(linked_id)   # simmetrizzazione esplicita
            dst_cig_cig.append(int(cig_id))

edge_pairs = sorted(set(zip(src_cig_cig, dst_cig_cig)))
src_cig_cig = [p[0] for p in edge_pairs]
dst_cig_cig = [p[1] for p in edge_pairs]

edge_cig_cig = build_edge_index(src_cig_cig, dst_cig_cig)
print("--- Archi [4/4]: CIG -> CIG creati con successo")

del src_cig_cig, dst_cig_cig
del CIG_id
del CIG_cig
gc.collect()

PA = HeteroData()

PA["SA"].x = nodeA_x
PA["CIG"].x = nodeB_x
PA["AGZ"].x = nodeC_x
PA["AG"].x = nodeD_x

PA["SA", "publishes", "CIG"].edge_index = edge_sa_cig
PA["CIG", "rev_publishes", "SA"].edge_index = edge_sa_cig.flip(0)

PA["CIG", "has_award", "AGZ"].edge_index = edge_cig_agz
PA["AGZ", "rev_has_award", "CIG"].edge_index = edge_cig_agz.flip(0)

PA["AGZ", "awarded_to", "AG"].edge_index = edge_agz_ag
PA["AG", "rev_awarded_to", "AGZ"].edge_index = edge_agz_ag.flip(0)

PA["CIG", "linked_to", "CIG"].edge_index = edge_cig_cig

# --- VALIDAZIONE DELL'INTEGRITÀ DEL GRAFO ---
assert edge_sa_cig[0].max() < PA["SA"].x.size(0), "Indice SA fuori range in edge_sa_cig"
assert edge_sa_cig[1].max() < PA["CIG"].x.size(0), "Indice CIG fuori range in edge_sa_cig"
assert edge_cig_agz[0].max() < PA["CIG"].x.size(0), "Indice CIG fuori range in edge_cig_agz"
assert edge_cig_agz[1].max() < PA["AGZ"].x.size(0), "Indice AGZ fuori range in edge_cig_agz"
assert edge_agz_ag[0].max() < PA["AGZ"].x.size(0), "Indice AGZ fuori range in edge_agz_ag"
assert edge_agz_ag[1].max() < PA["AG"].x.size(0), "Indice AG fuori range in edge_agz_ag"

PA.validate(raise_on_error=True)

torch.save(PA, MODELS_DIR / "PA_graph.pt")
print(f"--- Il grafo è stato salvato in {MODELS_DIR / 'PA_graph.pt'}")
#PA = torch.load("PA_graph.pt", weights_only=False)