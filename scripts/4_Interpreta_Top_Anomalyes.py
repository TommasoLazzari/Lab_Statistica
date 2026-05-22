import pandas as pd
import numpy as np
import json
import os
import sys
import torch
import torch.nn.functional as F
from pathlib import Path

# ── Aggiunge la cartella scripts al path per importare 4_AnomalyDetection ──
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

# ── Percorsi base ──────────────────────────────────────────────────────────
BASE_DIR = SCRIPT_DIR.parent
MODELS_DIR       = BASE_DIR / "models"
REPORTS_DIR      = BASE_DIR / "reports"
DATA_PROCESSED   = BASE_DIR / "data" / "processed"
DATA_RAW         = BASE_DIR / "data" / "raw"

# Score CSV
FILE_SA_SCORES  = REPORTS_DIR / "level4_results" / "anomaly_scores_SA.csv"
FILE_AG_SCORES  = REPORTS_DIR / "level4_results" / "anomaly_scores_AG.csv"
FILE_CIG_SCORES = REPORTS_DIR / "level4_results" / "anomaly_scores_CIG.csv"
FILE_AGZ_SCORES = REPORTS_DIR / "level4_results" / "anomaly_scores_AGZ.csv"

# Mapping SA / AG / AGZ
FILE_SA_MAPPING = DATA_PROCESSED / "SA_id.json"   # node_id -> {codice_fiscale, denominazione}
FILE_AG_MAPPING = DATA_PROCESSED / "AG_id.json"   # node_id -> {codice_fiscale, denominazione}
FILE_AGZ_MAPPING = DATA_PROCESSED / "AGZ_id.json" # node_id -> {cig, id_aggiudicazione}
FILE_SA_RAW     = DATA_RAW / "stazioni-appaltanti_csv.csv"
FILE_AG_RAW     = DATA_RAW / "aggiudicatari_csv.csv"

# Mapping CIG
FILE_CIG_CIG   = DATA_PROCESSED / "CIG_cig.json"              # cig_code -> graph_node_id
FILE_AGZ_LINKS = DATA_PROCESSED / "aggiudicatari_links.parquet"  # id_agz, cig, codice_fiscale

# ── Numero di anomalie da estrarre ─────────────────────────────────────────
try:
    TOP_N_INPUT = input("Inserisci il numero di anomalie da estrarre (es. 20): ").strip()
    TOP_N = int(TOP_N_INPUT) if TOP_N_INPUT else 20
except (ValueError, EOFError):
    TOP_N = 20
    print(f"Utilizzo il valore di default: {TOP_N}")

# ── Numero di feature da mostrare per spiegabilità ────────────────────────
try:
    TOP_FEAT_INPUT = input("Feature più impattanti da mostrare per nodo (es. 10): ").strip()
    TOP_FEAT = int(TOP_FEAT_INPUT) if TOP_FEAT_INPUT else 10
except (ValueError, EOFError):
    TOP_FEAT = 10
    print(f"Utilizzo il valore di default per le feature: {TOP_FEAT}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. Caricamento nomi da file RAW  (SA e AG)
# ══════════════════════════════════════════════════════════════════════════════

def load_name_lookup(mapping_file: Path, raw_file: Path) -> tuple[dict, pd.DataFrame]:
    """Restituisce (mapping dict, df con codice_fiscale -> denominazione)."""
    with open(mapping_file, "r") as f:
        mapping = json.load(f)

    df_raw = pd.read_csv(raw_file, sep=";", quotechar='"', low_memory=False)
    df_raw["codice_fiscale"] = df_raw["codice_fiscale"].astype(str).str.strip()
    df_raw["denominazione"]  = df_raw["denominazione"].astype(str).str.strip()
    df_names = df_raw[["codice_fiscale", "denominazione"]].drop_duplicates(subset="codice_fiscale")
    return mapping, df_names


def get_top_anomalies_with_names(score_file: Path, mapping_file: Path,
                                  raw_file: Path, node_type: str, n: int) -> pd.DataFrame:
    print(f"\n--- Analisi Top {n} Anomalie per: {node_type} ---")
    df_scores = pd.read_csv(score_file)
    top_scores = df_scores.sort_values(by="final_score", ascending=False).head(n).copy()

    mapping, df_names = load_name_lookup(mapping_file, raw_file)

    top_scores["codice_fiscale"] = top_scores["node_id"].apply(
        lambda x: mapping.get(str(x), {}).get("codice_fiscale", "N/D")
    )
    top_with_names = top_scores.merge(df_names, on="codice_fiscale", how="left")
    top_with_names["denominazione"] = top_with_names["denominazione"].fillna(
        top_with_names["node_id"].apply(
            lambda x: mapping.get(str(x), {}).get("denominazione", "Sconosciuto")
        )
    )
    return top_with_names[
        ["node_id", "codice_fiscale", "denominazione",
         "final_score", "attribute_score", "structural_score"]
    ]


# ══════════════════════════════════════════════════════════════════════════════
# 1b. Lookup CIG: codice, stazione appaltante e aggiudicatari
# ══════════════════════════════════════════════════════════════════════════════

def build_cig_node_to_code(cig_cig_file: Path) -> dict:
    """
    Inverte CIG_cig.json (cig_code -> node_id) in {node_id (int) -> cig_code (str)}.
    """
    print("\nCaricamento CIG_cig.json per lookup codice CIG...")
    with open(cig_cig_file, "r") as f:
        cig_cig = json.load(f)
    return {int(v): k for k, v in cig_cig.items()}


def build_cig_sa_lookup(data, sa_mapping_file: Path) -> dict:
    """
    Usa gli archi (SA, publishes, CIG) del grafo per costruire
    {cig_node_id (int) -> SA denominazione (str)}.
    Richiede il grafo già caricato in memoria.
    """
    print("Caricamento SA_id.json per lookup CIG→SA...")
    with open(sa_mapping_file, "r") as f:
        sa_id_map = json.load(f)
    sa_id_to_denom = {
        int(k): v.get("denominazione", "Sconosciuto")
        for k, v in sa_id_map.items()
    }

    # edge_index shape: [2, num_edges]  →  riga 0 = SA node_id, riga 1 = CIG node_id
    edge = data[("SA", "publishes", "CIG")].edge_index
    cig_to_sa: dict = {}
    for sa_idx, cig_idx in zip(edge[0].tolist(), edge[1].tolist()):
        cig_to_sa[cig_idx] = sa_id_to_denom.get(sa_idx, "Sconosciuto")
    return cig_to_sa


def get_top_cig_anomalies(
    score_file: Path,
    cig_node_to_code: dict,
    cig_to_sa: dict,
    ag_mapping_file: Path,
    links_file: Path,
    n: int,
) -> pd.DataFrame:
    """
    Carica i top N CIG più anomali e li arricchisce con:
      - codice CIG
      - denominazione della stazione appaltante
      - lista di tutti gli aggiudicatari (da tutte le aggiudicazioni del CIG)
    """
    print(f"\n--- Analisi Top {n} Anomalie per: CIG ---")
    df_scores = pd.read_csv(score_file)
    top = df_scores.sort_values(by="final_score", ascending=False).head(n).copy()

    node_ids = top["node_id"].tolist()

    # Codice CIG e stazione appaltante
    top["cig_code"]            = [cig_node_to_code.get(int(nid), "N/D")  for nid in node_ids]
    top["stazione_appaltante"] = [cig_to_sa.get(int(nid), "N/D")         for nid in node_ids]

    # Aggiudicatari: carichiamo solo le righe dei CIG di interesse
    valid_codes = {c for c in top["cig_code"].tolist() if c != "N/D"}

    print("Caricamento AG_id.json per lookup codice_fiscale → denominazione...")
    with open(ag_mapping_file, "r") as f:
        ag_id_raw = json.load(f)
    ag_cf_to_denom: dict = {
        v.get("codice_fiscale", ""): v.get("denominazione", "N/D")
        for v in ag_id_raw.values()
        if v.get("codice_fiscale")
    }
    del ag_id_raw

    print("Caricamento aggiudicatari_links.parquet (filtrato ai top CIG)...")
    links = pd.read_parquet(links_file, columns=["cig", "codice_fiscale"])
    links_top = links[links["cig"].isin(valid_codes)].copy()
    del links

    # Per ogni CIG code, raccogliamo l'insieme univoco di denominazioni aggiudicatari
    cig_code_to_ags: dict = (
        links_top.groupby("cig")["codice_fiscale"]
        .apply(lambda cf_series: sorted({
            ag_cf_to_denom.get(cf, cf)
            for cf in cf_series
            if isinstance(cf, str) and cf
        }))
        .to_dict()
    )
    del links_top

    top["aggiudicatari"] = top["cig_code"].apply(
        lambda code: cig_code_to_ags.get(code, [])
    )
    top["aggiudicatari_str"] = top["aggiudicatari"].apply(
        lambda lst: " | ".join(lst) if lst else "N/D"
    )

    return top[[
        "node_id", "cig_code", "stazione_appaltante",
        "aggiudicatari", "aggiudicatari_str",
        "final_score", "attribute_score", "structural_score",
    ]]


def get_top_agz_anomalies(
    score_file: Path,
    agz_mapping_file: Path,
    ag_mapping_file: Path,
    links_file: Path,
    n: int,
    cig_code_to_sa: dict = None,
) -> pd.DataFrame:
    """
    Carica i top N AGZ (Aggiudicazioni) più anomali e li arricchisce con:
      - codice CIG
      - ID aggiudicazione
      - denominazione dell'aggiudicatario (da link cf)
    """
    print(f"\n--- Analisi Top {n} Anomalie per: Aggiudicazioni (AGZ) ---")
    df_scores = pd.read_csv(score_file)
    top = df_scores.sort_values(by="final_score", ascending=False).head(n).copy()

    print("Caricamento aggiudicazioni.parquet per lookup dati...")
    # Usiamo il parquet originale perché AGZ_id.json potrebbe essere incompleto o disallineato
    df_agz_raw = pd.read_parquet(DATA_PROCESSED / "aggiudicazioni.parquet")
    
    # Per sicurezza filtriamo gli indici (node_id) che sono nel range del parquet
    max_idx = len(df_agz_raw)
    valid_top = top[top["node_id"] < max_idx].copy()
    if len(valid_top) < len(top):
        print(f"  [ATTENZIONE] {len(top) - len(valid_top)} node_id fuori range per aggiudicazioni.parquet")
    
    # Estrazione cig e id_aggiudicazione usando positional index
    node_indices = valid_top["node_id"].tolist()
    chunk_data = df_agz_raw.iloc[node_indices][["cig", "id_aggiudicazione"]].copy()
    chunk_data["node_id"] = node_indices
    
    valid_top = valid_top.merge(chunk_data, on="node_id", how="left")
    valid_top["cig_code"] = valid_top["cig"].fillna("N/D")
    valid_top["id_agg"]   = valid_top["id_aggiudicazione"].fillna("N/D")

    print("Caricamento AG_id.json per lookup AGZ → Aggiudicatario...")
    with open(ag_mapping_file, "r") as f:
        ag_id_raw = json.load(f)
    ag_cf_to_denom = {
        v.get("codice_fiscale", ""): v.get("denominazione", "N/D")
        for v in ag_id_raw.values()
        if v.get("codice_fiscale")
    }

    print("Caricamento aggiudicatari_links.parquet per link AGZ...")
    links = pd.read_parquet(links_file)
    
    # Preparazione per merge
    valid_top["id_agg_str"] = valid_top["id_agg"].astype(str)
    links["id_agg_str"] = links["id_aggiudicazione"].astype(str)
    
    # Filtro links per velocizzare
    target_ids = set(valid_top["id_agg_str"].unique())
    links_filt = links[links["id_agg_str"].isin(target_ids)].drop_duplicates(subset="id_agg_str")
    
    valid_top = valid_top.merge(links_filt[["id_agg_str", "codice_fiscale"]], on="id_agg_str", how="left")
    valid_top["aggiudicatario"] = valid_top["codice_fiscale"].apply(lambda cf: ag_cf_to_denom.get(cf, cf) if isinstance(cf, str) and cf != "nan" else "N/D")

    if cig_code_to_sa:
        valid_top["stazione_appaltante"] = valid_top["cig_code"].apply(lambda c: cig_code_to_sa.get(c, "N/D"))
    else:
        valid_top["stazione_appaltante"] = "N/D"

    return valid_top[[
        "node_id", "cig_code", "id_agg", "aggiudicatario", "stazione_appaltante",
        "final_score", "attribute_score", "structural_score",
    ]]


# ══════════════════════════════════════════════════════════════════════════════
# 2. Caricamento modello e grafo per il calcolo dell'errore per-feature
# ══════════════════════════════════════════════════════════════════════════════

def load_model_and_graph():
    """Carica il grafo PA e il modello addestrato. Restituisce (model, data, ad_module)."""
    print("\nCaricamento grafo e modello...")

    # Importazione dell'architettura dal file 4_AnomalyDetection
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "anomaly_detection",
        str(SCRIPT_DIR / "4_AnomalyDetection.py")
    )
    ad = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ad)

    data = torch.load(MODELS_DIR / "PA_graph.pt", weights_only=False)

    checkpoint = torch.load(
        MODELS_DIR / "level4_minibatch_outputs.pt",
        weights_only=False,
        map_location="cpu",
    )

    hp = checkpoint["hyperparameters"]
    hidden_dim  = hp.get("hidden_dim", 32)
    latent_dim  = hp.get("latent_dim", 16)
    dropout     = hp.get("dropout", 0.1) if "dropout" in hp else 0.1

    metadata          = data.metadata()
    in_dims           = {nt: data[nt].x.size(1) for nt in data.node_types}
    decoder_relations = checkpoint["decoder_relations"]

    model = ad.BaselineMiniBatchModel(
        metadata=metadata,
        in_dims=in_dims,
        decoder_relations=decoder_relations,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        dropout=dropout,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Modello e grafo caricati con successo.")
    return model, data, ad


# ══════════════════════════════════════════════════════════════════════════════
# 3. Errore di ricostruzione per-feature per una lista di nodi
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_feature_reconstruction_details(
    model,
    data,
    ad_module,
    node_type: str,
    node_ids: list[int],
    num_neighbors: list[int] = [15, 10],
    batch_size: int = 512,
) -> dict:
    """
    Restituisce un dizionario con:
      - 'errors': matrice [len(node_ids), n_features] con (x_pred - x_true)^2
      - 'observed': matrice con x_true
      - 'predicted': matrice con x_pred
    """
    from torch_geometric.loader import NeighborLoader

    all_node_ids = torch.tensor(node_ids, dtype=torch.long)
    loader = NeighborLoader(
        data=data,
        num_neighbors=num_neighbors,
        input_nodes=(node_type, all_node_ids),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    n_features = data[node_type].x.size(1)
    id_to_row = {nid: i for i, nid in enumerate(node_ids)}

    errors = np.full((len(node_ids), n_features), np.nan, dtype=np.float32)
    obs    = np.full((len(node_ids), n_features), np.nan, dtype=np.float32)
    pred   = np.full((len(node_ids), n_features), np.nan, dtype=np.float32)

    for batch in loader:
        z_dict, x_hat_dict = model(batch.x_dict, batch.edge_index_dict)

        seed_size = batch[node_type].batch_size
        local_idx = torch.arange(seed_size)

        x_true_torch = batch[node_type].x[local_idx]
        x_pred_torch = x_hat_dict[node_type][local_idx]
        
        x_true = x_true_torch.cpu().numpy()
        x_pred = x_pred_torch.cpu().numpy()
        per_feat_err = (x_pred - x_true) ** 2

        global_ids = batch[node_type].n_id[local_idx].tolist()

        for local_i, gid in enumerate(global_ids):
            if gid in id_to_row:
                row_idx = id_to_row[gid]
                errors[row_idx] = per_feat_err[local_i]
                obs[row_idx]    = x_true[local_i]
                pred[row_idx]   = x_pred[local_i]

    return {"errors": errors, "observed": obs, "predicted": pred}


# ══════════════════════════════════════════════════════════════════════════════
# 4. Nomi delle feature dai file parquet
# ══════════════════════════════════════════════════════════════════════════════

def get_feature_names(node_type: str) -> list[str]:
    """
    Ritorna i nomi delle colonne features (escludendo 'id') usate per costruire
    il tensore x del nodo node_type.
    """
    parquet_map = {
        "SA":  DATA_PROCESSED / "stazioni_appaltanti_v2.parquet",
        "AG":  DATA_PROCESSED / "aggiudicatari_v2.parquet",
        "CIG": DATA_PROCESSED / "CIG_v2.parquet",
        "AGZ": DATA_PROCESSED / "aggiudicazioni_v2.parquet",
    }
    if node_type not in parquet_map:
        return [f"feature_{i}" for i in range(500)]  # fallback generico

    df = pd.read_parquet(parquet_map[node_type])
    cols = [c for c in df.columns if c != "id"]
    return cols


# ══════════════════════════════════════════════════════════════════════════════
# 5. Spiegabilità: top feature per anomalia
# ══════════════════════════════════════════════════════════════════════════════

def explain_top_anomalies(
    top_df: pd.DataFrame,
    node_type: str,
    model,
    data,
    ad_module,
    top_k_features: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Restituisce (top_df_arricchito, df_long_features).
    """
    node_ids = top_df["node_id"].tolist()
    feature_names = get_feature_names(node_type)

    # Statistiche globali per Z-score
    print(f"Calcolo statistiche globali per {node_type}...")
    x_all = data[node_type].x.cpu().numpy()
    means = np.mean(x_all, axis=0)
    stds  = np.std(x_all, axis=0)
    stds[stds < 1e-12] = 1.0

    print(f"\nCalcolo errori di ricostruzione per {len(node_ids)} nodi ({node_type})...")
    details = compute_feature_reconstruction_details(
        model=model,
        data=data,
        ad_module=ad_module,
        node_type=node_type,
        node_ids=node_ids,
    )
    errors = details["errors"]
    obs    = details["observed"]
    pred   = details["predicted"]

    top_feature_lists = []
    mean_attr_errors  = []
    long_data = []

    for i, nid in enumerate(node_ids):
        node_errors = errors[i]
        if np.all(np.isnan(node_errors)):
            top_feature_lists.append("N/D (nodo non trovato nel grafo)")
            mean_attr_errors.append(np.nan)
            continue

        ranked = np.argsort(node_errors)[::-1]
        top_feats_str = []
        
        for rank, fi in enumerate(ranked[:top_k_features]):
            fname = feature_names[fi] if fi < len(feature_names) else f"feat_{fi}"
            err_val  = float(node_errors[fi])
            obs_val  = float(obs[i, fi])
            pred_val = float(pred[i, fi])
            z_score  = float((obs_val - means[fi]) / stds[fi])

            info_str = f"{fname} (err={err_val:.4f}, obs={obs_val:.4f}, pred={pred_val:.4f}, z={z_score:.2f})"
            top_feats_str.append(info_str)

            long_data.append({
                "node_id": nid,
                "rank": rank + 1,
                "feature_name": fname,
                "error": err_val,
                "observed": obs_val,
                "predicted": pred_val,
                "z_score": z_score
            })
            
        top_feature_lists.append(" | ".join(top_feats_str))
        mean_attr_errors.append(float(np.mean(node_errors)))

    result = top_df.copy()
    result["mean_attr_reconstruction_err"] = mean_attr_errors
    result["top_anomalous_features"]        = top_feature_lists
    
    df_long = pd.DataFrame(long_data)
    return result, df_long


# ══════════════════════════════════════════════════════════════════════════════
# 6. Main
# ══════════════════════════════════════════════════════════════════════════════

# --- Top anomalie base SA e AG (con nomi) ---
top_sa = get_top_anomalies_with_names(FILE_SA_SCORES, FILE_SA_MAPPING, FILE_SA_RAW,
                                       "Stazioni Appaltanti (SA)", TOP_N)
top_ag = get_top_anomalies_with_names(FILE_AG_SCORES, FILE_AG_MAPPING, FILE_AG_RAW,
                                       "Aggiudicatari (AG)", TOP_N)

# --- Top CIG e AGZ anomalie (caricamento base senza grafo se possibile) ---
# Il lookup SA per CIG richiede il grafo, quindi lo gestiremo dopo.
# Il lookup base per CIG e AGZ invece può procedere.
try:
    cig_node_to_code = build_cig_node_to_code(FILE_CIG_CIG)
except:
    cig_node_to_code = {}

try:
    top_agz = get_top_agz_anomalies(
        score_file=FILE_AGZ_SCORES,
        agz_mapping_file=FILE_AGZ_MAPPING,
        ag_mapping_file=FILE_AG_MAPPING,
        links_file=FILE_AGZ_LINKS,
        n=TOP_N
    )
    agz_lookup_ok = True
except Exception as e:
    print(f"\n[ATTENZIONE] Impossibile caricare AGZ scores/lookup: {e}")
    top_agz = pd.DataFrame()
    agz_lookup_ok = False

# --- Caricamento modello e grafo (richiede PyTorch + torch_geometric) ---
try:
    model, data, ad_module = load_model_and_graph()
    model_loaded = True
except Exception as e:
    print(f"\n[ATTENZIONE] Impossibile caricare il modello: {e}")
    print("Verranno mostrati solo i punteggi senza spiegabilità per-feature.")
    model_loaded = False

# --- Top CIG anomalie ---
# Il lookup SA richiede il grafo; il lookup AG usa solo file su disco.
if model_loaded:
    try:
        cig_to_sa        = build_cig_sa_lookup(data, FILE_SA_MAPPING)
        cig_code_to_sa   = {cig_node_to_code[nid]: sa_denom for nid, sa_denom in cig_to_sa.items() if nid in cig_node_to_code}
        top_cig = get_top_cig_anomalies(
            score_file=FILE_CIG_SCORES,
            cig_node_to_code=cig_node_to_code,
            cig_to_sa=cig_to_sa,
            ag_mapping_file=FILE_AG_MAPPING,
            links_file=FILE_AGZ_LINKS,
            n=TOP_N,
        )
        cig_lookup_ok = True

        # Aggiorna top_agz con le stazioni appaltanti ora che abbiamo il grafo
        if agz_lookup_ok and not top_agz.empty:
            top_agz["stazione_appaltante"] = top_agz["cig_code"].apply(lambda c: cig_code_to_sa.get(c, "N/D"))
    except Exception as e:
        print(f"\n[ATTENZIONE] Impossibile costruire lookup CIG: {e}")
        cig_lookup_ok = False
else:
    # Fallback senza grafo: solo score e codice CIG, nessun SA/AG lookup
    try:
        cig_node_to_code = build_cig_node_to_code(FILE_CIG_CIG)
        df_cig_scores = pd.read_csv(FILE_CIG_SCORES)
        top_cig = df_cig_scores.sort_values("final_score", ascending=False).head(TOP_N).copy()
        top_cig["cig_code"]            = top_cig["node_id"].apply(
            lambda x: cig_node_to_code.get(int(x), "N/D"))
        top_cig["stazione_appaltante"] = "N/D (grafo non caricato)"
        top_cig["aggiudicatari"]       = [[] for _ in range(len(top_cig))]
        top_cig["aggiudicatari_str"]   = "N/D (grafo non caricato)"
        cig_lookup_ok = True
    except Exception as e:
        print(f"\n[ATTENZIONE] Impossibile caricare CIG scores: {e}")
        cig_lookup_ok = False

# --- Arricchimento con errori per-feature (SA, AG, CIG, AGZ) ---
if model_loaded:
    top_sa_explained, df_sa_feats = explain_top_anomalies(
        top_df=top_sa, node_type="SA",
        model=model, data=data, ad_module=ad_module,
        top_k_features=TOP_FEAT,
    )
    top_ag_explained, df_ag_feats = explain_top_anomalies(
        top_df=top_ag, node_type="AG",
        model=model, data=data, ad_module=ad_module,
        top_k_features=TOP_FEAT,
    )
    if cig_lookup_ok:
        top_cig_explained, df_cig_feats = explain_top_anomalies(
            top_df=top_cig, node_type="CIG",
            model=model, data=data, ad_module=ad_module,
            top_k_features=TOP_FEAT,
        )
    else:
        top_cig_explained, df_cig_feats = pd.DataFrame(), pd.DataFrame()

    if agz_lookup_ok:
        top_agz_explained, df_agz_feats = explain_top_anomalies(
            top_df=top_agz, node_type="AGZ",
            model=model, data=data, ad_module=ad_module,
            top_k_features=TOP_FEAT,
        )
    else:
        top_agz_explained, df_agz_feats = pd.DataFrame(), pd.DataFrame()

else:
    top_sa_explained = top_sa.copy()
    top_ag_explained = top_ag.copy()
    top_sa_explained["top_anomalous_features"] = "N/D"
    top_ag_explained["top_anomalous_features"] = "N/D"
    df_sa_feats = df_ag_feats = df_cig_feats = df_agz_feats = pd.DataFrame()
    
    if cig_lookup_ok:
        top_cig_explained = top_cig.copy()
        top_cig_explained["top_anomalous_features"] = "N/D"
    else:
        try:
            df_cig_scores = pd.read_csv(FILE_CIG_SCORES)
            top_cig_explained = df_cig_scores.sort_values("final_score", ascending=False).head(TOP_N).copy()
            top_cig_explained["cig_code"] = top_cig_explained["node_id"].apply(lambda x: cig_node_to_code.get(int(x), "N/D"))
            top_cig_explained["stazione_appaltante"] = "N/D"
            top_cig_explained["top_anomalous_features"] = "N/D"
        except:
            top_cig_explained = pd.DataFrame()
    
    if agz_lookup_ok:
        top_agz_explained = top_agz.copy()
        top_agz_explained["top_anomalous_features"] = "N/D"
    else:
        top_agz_explained = pd.DataFrame()


# ── Stampa SA / AG ────────────────────────────────────────────────────────────
def pretty_print(df: pd.DataFrame, title: str):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    cols_base = ["node_id", "codice_fiscale", "denominazione",
                 "final_score", "attribute_score", "structural_score"]
    if "mean_attr_reconstruction_err" in df.columns:
        cols_base.append("mean_attr_reconstruction_err")
    print(df[cols_base].to_string(index=False))

    if "top_anomalous_features" in df.columns:
        print(f"\n{'─'*80}")
        print("  DETTAGLIO FEATURE PIÙ ANOMALE PER NODO")
        print(f"{'─'*80}")
        for _, row in df.iterrows():
            print(f"\n  [{row['node_id']}] {row['denominazione']}"
                  f"  (final={row['final_score']:.4f} | attr={row['attribute_score']:.4f})")
            for feat_entry in str(row["top_anomalous_features"]).split(" | "):
                print(f"      • {feat_entry}")


# ── Stampa CIG ────────────────────────────────────────────────────────────────
def pretty_print_cig(df: pd.DataFrame, title: str):
    """Stampa formattata per i CIG: codice, SA, lista aggiudicatari, feature anomale."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    if df.empty:
        print("  [Nessun dato disponibile]")
        return

    cols_base = ["node_id", "cig_code", "stazione_appaltante",
                 "final_score", "attribute_score", "structural_score"]
    if "mean_attr_reconstruction_err" in df.columns:
        cols_base.append("mean_attr_reconstruction_err")
    print(df[cols_base].to_string(index=False))

    print(f"\n{'─'*80}")
    print("  DETTAGLIO CIG: AGGIUDICATARI E FEATURE PIÙ ANOMALE")
    print(f"{'─'*80}")
    for _, row in df.iterrows():
        print(f"\n  [{row['node_id']}] CIG: {row['cig_code']}"
              f"  (final={row['final_score']:.4f} | attr={row['attribute_score']:.4f})")
        print(f"    SA : {row.get('stazione_appaltante', 'N/D')}")
        aggiudicatari = row.get("aggiudicatari", [])
        if aggiudicatari:
            for ag in aggiudicatari:
                print(f"    AG : {ag}")
        else:
            print("    AG : N/D")
        if "top_anomalous_features" in df.columns:
            print("    Feature anomale:")
            for feat_entry in str(row.get("top_anomalous_features", "N/D")).split(" | "):
                print(f"      • {feat_entry}")


pretty_print(top_sa_explained,  f"TOP {TOP_N} ANOMALIE - STAZIONI APPALTANTI")
pretty_print(top_ag_explained,  f"TOP {TOP_N} ANOMALIE - AGGIUDICATARI")
pretty_print_cig(top_cig_explained, f"TOP {TOP_N} ANOMALIE - CIG")


# ── Stampa AGZ ────────────────────────────────────────────────────────────────
def pretty_print_agz(df: pd.DataFrame, title: str):
    """Stampa formattata per le Aggiudicazioni: CIG, ID Agg, Aggiudicatario, feature."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    if df.empty:
        print("  [Nessun dato disponibile]")
        return

    cols_base = ["node_id", "final_score", "attribute_score", "structural_score"]
    for c in ["cig_code", "id_agg", "aggiudicatario", "stazione_appaltante", "mean_attr_reconstruction_err"]:
        if c in df.columns:
            cols_base.append(c)
    print(df[cols_base].to_string(index=False))

    print(f"\n{'─'*80}")
    print("  DETTAGLIO AGGIUDICAZIONE: AGGIUDICATARIO E FEATURE PIÙ ANOMALE")
    print(f"{'─'*80}")
    for _, row in df.iterrows():
        print(f"\n  [{row['node_id']}] CIG: {row['cig_code']} | ID Agg: {row['id_agg']}"
              f"  (final={row['final_score']:.4f} | attr={row['attribute_score']:.4f})")
        print(f"    SA : {row.get('stazione_appaltante', 'N/D')}")
        print(f"    AG : {row.get('aggiudicatario', 'N/D')}")
        if "top_anomalous_features" in df.columns:
            print("    Feature anomale:")
            for feat_entry in str(row.get("top_anomalous_features", "N/D")).split(" | "):
                print(f"      • {feat_entry}")

pretty_print_agz(top_agz_explained, f"TOP {TOP_N} ANOMALIE - AGGIUDICAZIONI (AGZ)")


# ── Salvataggio testo ─────────────────────────────────────────────────────────
out_dir = REPORTS_DIR / "interpretazioni"
out_dir.mkdir(parents=True, exist_ok=True)

txt_path = out_dir / f"top_{TOP_N}_anomalies_names.txt"
with open(txt_path, "w", encoding="utf-8") as f:

    def write_section(df: pd.DataFrame, title: str):
        f.write(f"\n{'='*80}\n  {title}\n{'='*80}\n")
        cols_base = ["node_id", "codice_fiscale", "denominazione",
                     "final_score", "attribute_score", "structural_score"]
        if "mean_attr_reconstruction_err" in df.columns:
            cols_base.append("mean_attr_reconstruction_err")
        f.write(df[cols_base].to_string(index=False))
        f.write(f"\n\n{'─'*80}\n  FEATURE PIÙ ANOMALE\n{'─'*80}\n")
        for _, row in df.iterrows():
            f.write(f"\n  [{row['node_id']}] {row['denominazione']}"
                    f"  (final={row['final_score']:.4f} | attr={row['attribute_score']:.4f})\n")
            for feat_entry in str(row.get("top_anomalous_features", "N/D")).split(" | "):
                f.write(f"      • {feat_entry}\n")

    def write_section_cig(df: pd.DataFrame, title: str):
        f.write(f"\n{'='*80}\n  {title}\n{'='*80}\n")
        if df.empty:
            f.write("  [Nessun dato disponibile]\n")
            return
        cols_base = ["node_id", "cig_code", "stazione_appaltante",
                     "final_score", "attribute_score", "structural_score"]
        if "mean_attr_reconstruction_err" in df.columns:
            cols_base.append("mean_attr_reconstruction_err")
        f.write(df[cols_base].to_string(index=False))
        f.write(f"\n\n{'─'*80}\n  DETTAGLIO CIG: AGGIUDICATARI E FEATURE PIÙ ANOMALE\n{'─'*80}\n")
        for _, row in df.iterrows():
            f.write(f"\n  [{row['node_id']}] CIG: {row['cig_code']}"
                    f"  (final={row['final_score']:.4f} | attr={row['attribute_score']:.4f})\n")
            f.write(f"    SA : {row.get('stazione_appaltante', 'N/D')}\n")
            aggiudicatari = row.get("aggiudicatari", [])
            if aggiudicatari:
                for ag in aggiudicatari:
                    f.write(f"    AG : {ag}\n")
            else:
                f.write("    AG : N/D\n")
            if "top_anomalous_features" in df.columns:
                f.write("    Feature anomale:\n")
                for feat_entry in str(row.get("top_anomalous_features", "N/D")).split(" | "):
                    f.write(f"      • {feat_entry}\n")

    def write_section_agz(df: pd.DataFrame, title: str):
        f.write(f"\n{'='*80}\n  {title}\n{'='*80}\n")
        if df.empty:
            f.write("  [Nessun dato disponibile]\n")
            return
        cols_base = ["node_id", "final_score", "attribute_score", "structural_score"]
        for c in ["cig_code", "id_agg", "aggiudicatario", "stazione_appaltante", "mean_attr_reconstruction_err"]:
            if c in df.columns:
                cols_base.append(c)
        f.write(df[cols_base].to_string(index=False))
        f.write(f"\n\n{'─'*80}\n  DETTAGLIO AGGIUDICAZIONE: AGGIUDICATARIO E FEATURE PIÙ ANOMALE\n{'─'*80}\n")
        for _, row in df.iterrows():
            f.write(f"\n  [{row['node_id']}] CIG: {row['cig_code']} | ID Agg: {row['id_agg']}"
                    f"  (final={row['final_score']:.4f} | attr={row['attribute_score']:.4f})\n")
            f.write(f"    SA : {row.get('stazione_appaltante', 'N/D')}\n")
            f.write(f"    AG : {row.get('aggiudicatario', 'N/D')}\n")
            if "top_anomalous_features" in df.columns:
                f.write("    Feature anomale:\n")
                for feat_entry in str(row.get("top_anomalous_features", "N/D")).split(" | "):
                    f.write(f"      • {feat_entry}\n")

    write_section(top_sa_explained,  f"TOP {TOP_N} ANOMALIE - STAZIONI APPALTANTI")
    write_section(top_ag_explained,  f"TOP {TOP_N} ANOMALIE - AGGIUDICATARI")
    write_section_cig(top_cig_explained, f"TOP {TOP_N} ANOMALIE - CIG")
    write_section_agz(top_agz_explained, f"TOP {TOP_N} ANOMALIE - AGGIUDICAZIONI (AGZ)")

print(f"\nRisultati salvati in: {txt_path}")

# ── Salvataggio EXCEL ──────────────────────────────────────────────────────────
excel_path = out_dir / f"Analisi_Anomalie_Top_{TOP_N}.xlsx"

def prepare_for_excel(explained_df, feats_df):
    if explained_df.empty:
        return pd.DataFrame()
    # Rimuoviamo la colonna testuale cumulativa se esistono i dettagli per-feature
    df_meta = explained_df.copy()
    if not feats_df.empty and "top_anomalous_features" in df_meta.columns:
        df_meta = df_meta.drop(columns=["top_anomalous_features"])
    
    # Se abbiamo i dettagli per-feature, facciamo il merge per avere un formato "long" leggibile
    if not feats_df.empty:
        # Ordiniamo le colonne in modo che node_id e score siano all'inizio
        final_df = df_meta.merge(feats_df, on="node_id", how="left")
        # Riorganizzazione colonne: sposta feature_name, error, ecc. dopo i metadati base
        cols = list(final_df.columns)
        for c in ["rank", "feature_name", "error", "observed", "predicted", "z_score"]:
            if c in cols:
                cols.remove(c)
                cols.append(c)
        return final_df[cols]
    return df_meta

print(f"\nGenerazione file Excel: {excel_path}...")
try:
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        # SA
        df_sa_final = prepare_for_excel(top_sa_explained, df_sa_feats)
        if not df_sa_final.empty:
            df_sa_final.to_excel(writer, sheet_name="SA", index=False)
        
        # AG
        df_ag_final = prepare_for_excel(top_ag_explained, df_ag_feats)
        if not df_ag_final.empty:
            df_ag_final.to_excel(writer, sheet_name="AG", index=False)
            
        # CIG
        if cig_lookup_ok:
            df_cig_final = prepare_for_excel(top_cig_explained, df_cig_feats)
            if not df_cig_final.empty:
                # Rimuoviamo la lista oggetti (potrebbe essere troppo lunga per una cella excel)
                if "aggiudicatari" in df_cig_final.columns:
                    df_cig_final = df_cig_final.drop(columns=["aggiudicatari"])
                df_cig_final.to_excel(writer, sheet_name="CIG", index=False)
                
        # AGZ
        if agz_lookup_ok:
            df_agz_final = prepare_for_excel(top_agz_explained, df_agz_feats)
            if not df_agz_final.empty:
                df_agz_final.to_excel(writer, sheet_name="AGZ", index=False)

    print(f"File Excel salvato con successo: {excel_path}")
except Exception as e:
    print(f"[ERRORE] Salvataggio Excel fallito: {e}")

# Rimuoviamo il riferimento ai CSV nel print finale
