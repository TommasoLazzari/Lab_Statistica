import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

# --- DEFINIZIONE PERCORSI RELATIVI ---
# La directory base è la cartella superiore a quella dello script (assumendo che lo script sia in 'scripts/')
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# --- CONFIGURAZIONE PER ROBUSTEZZA ---
np.random.seed(42)
SCALER_PARAMS = {}

def standardize_col(df, col_name, train_ratio=0.8):
    """Calcola media e std solo sul train_ratio e applica a tutto il df."""
    # Creiamo un subset di train deterministico per questo dataframe
    mask = np.random.rand(len(df)) < train_ratio
    train_subset = df.loc[mask, col_name]
    
    mean_val = train_subset.mean(skipna=True)
    std_val  = train_subset.std(skipna=True)
    
    # Se std è quasi zero, evitiamo divisione per zero
    if pd.isna(std_val) or std_val < 1e-12:
        std_val = 1.0
        
    SCALER_PARAMS[col_name] = {"mean": float(mean_val), "std": float(std_val)}
    
    df[col_name] = ((df[col_name] - mean_val) / std_val).astype("float32")
    return df

# Crea le cartelle se non esistono
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", None)

def fix_codice_ausa(s: pd.Series) -> pd.Series:
    s = (
        s.astype("string")
         .str.strip()
         .str.replace(r"\.0+$", "", regex=True)
         .str.lstrip("0")
         .replace("", pd.NA)
    )
    return s.astype("object").where(s.notna(), None)

print("----- DATA PREPROCESSING -----")
#####################################################################################################################
##STAZIONI APPALTANTI   #############################################################################################
#####################################################################################################################
stazioni_appaltanti = pd.read_csv(DATA_RAW_DIR / "stazioni-appaltanti_csv.csv", na_values = ["N.A.", "NA", ""], sep=";", low_memory=False)
print("CARICAMENTO [1/4]: file stazioni-appaltanti_csv.csv caricato con successo")
stazioni_appaltanti.drop_duplicates(inplace=True)

#codice_fiscale
stazioni_appaltanti["codice_fiscale"] = stazioni_appaltanti["codice_fiscale"].astype("string")
#denominazione
stazioni_appaltanti["denominazione"] = stazioni_appaltanti["denominazione"].astype("string")
#codice_ausa
stazioni_appaltanti["codice_ausa"] = fix_codice_ausa(stazioni_appaltanti["codice_ausa"])
#provincia_codice
stazioni_appaltanti["provincia_codice"] = stazioni_appaltanti["provincia_codice"].astype("string")
stazioni_appaltanti["provincia_sigla"] = (
    stazioni_appaltanti["provincia_codice"]
        .str.replace("IT-", "", regex=False)
)
prov_to_reg = {
    "TO":"Piemonte","VC":"Piemonte","NO":"Piemonte","CN":"Piemonte","AT":"Piemonte","AL":"Piemonte","BI":"Piemonte","VB":"Piemonte",
    "AO":"Valle_Aosta",
    "MI":"Lombardia","BG":"Lombardia","BS":"Lombardia","CO":"Lombardia","CR":"Lombardia","LC":"Lombardia",
    "LO":"Lombardia","MN":"Lombardia","MB":"Lombardia","PV":"Lombardia","SO":"Lombardia","VA":"Lombardia",
    "TN":"Trentino_Alto_Adige","BZ":"Trentino_Alto_Adige",
    "VE":"Veneto","VR":"Veneto","VI":"Veneto","BL":"Veneto","TV":"Veneto","PD":"Veneto","RO":"Veneto",
    "UD":"Friuli_Venezia_Giulia","PN":"Friuli_Venezia_Giulia","GO":"Friuli_Venezia_Giulia","TS":"Friuli_Venezia_Giulia",
    "GE":"Liguria","IM":"Liguria","SP":"Liguria","SV":"Liguria",
    "BO":"Emilia_Romagna","FE":"Emilia_Romagna","FC":"Emilia_Romagna","MO":"Emilia_Romagna","PR":"Emilia_Romagna",
    "PC":"Emilia_Romagna","RA":"Emilia_Romagna","RE":"Emilia_Romagna","RN":"Emilia_Romagna","FO":"Emilia_Romagna",
    "FI":"Toscana","AR":"Toscana","GR":"Toscana","LI":"Toscana","LU":"Toscana","MS":"Toscana",
    "PI":"Toscana","PT":"Toscana","PO":"Toscana","SI":"Toscana",
    "PG":"Umbria","TR":"Umbria",
    "AN":"Marche","AP":"Marche","FM":"Marche","MC":"Marche","PU":"Marche",
    "RM":"Lazio","FR":"Lazio","LT":"Lazio","RI":"Lazio","VT":"Lazio",
    "AQ":"Abruzzo","CH":"Abruzzo","PE":"Abruzzo","TE":"Abruzzo",
    "CB":"Molise","IS":"Molise",
    "NA":"Campania","AV":"Campania","BN":"Campania","CE":"Campania","SA":"Campania","N.A.":"Campania",
    "BA":"Puglia","BR":"Puglia","BT":"Puglia","FG":"Puglia","LE":"Puglia","TA":"Puglia",
    "MT":"Basilicata","PZ":"Basilicata",
    "CZ":"Calabria","CS":"Calabria","KR":"Calabria","RC":"Calabria","VV":"Calabria",
    "PA":"Sicilia","AG":"Sicilia","CL":"Sicilia","CT":"Sicilia","EN":"Sicilia","ME":"Sicilia",
    "RG":"Sicilia","SR":"Sicilia","TP":"Sicilia",
    "CA":"Sardegna","NU":"Sardegna","OR":"Sardegna","SS":"Sardegna","SU":"Sardegna", "OT":"Sardegna", "VS":"Sardegna","OG":"Sardegna","CI":"Sardegna"
}
stazioni_appaltanti["regione"] = (
    stazioni_appaltanti["provincia_sigla"].map(prov_to_reg)
)
stazioni_appaltanti["soggetto_estero"] = stazioni_appaltanti["soggetto_estero"].fillna(0).astype("int8")
mask_estero = stazioni_appaltanti["provincia_sigla"].isin(["EE", "99"])
stazioni_appaltanti.loc[mask_estero, "soggetto_estero"] = 1
stazioni_appaltanti = pd.get_dummies(
    stazioni_appaltanti,
    columns=["regione"],
    prefix="reg_",
    dtype="int8"
)
del mask_estero
del prov_to_reg
stazioni_appaltanti.drop(columns=["provincia_sigla", "provincia_codice"], inplace=True)
#soggetto_estero
stazioni_appaltanti["soggetto_estero"] = stazioni_appaltanti.get("soggetto_estero", 0)
stazioni_appaltanti["soggetto_estero"] = stazioni_appaltanti["soggetto_estero"].fillna(0).astype("int8")
#flag_inHouse
stazioni_appaltanti["flag_inHouse"] = stazioni_appaltanti["flag_inHouse"].astype("int8")
#stato
stazioni_appaltanti = pd.get_dummies(
    stazioni_appaltanti,
    columns=["stato"],
    prefix="stato",
    dtype="int8"
)
#flag_aprtecipata
stazioni_appaltanti["flag_partecipata"] = stazioni_appaltanti["flag_partecipata"].astype("int8")
#natura_giuridica_codice
stazioni_appaltanti["natura_giuridica_codice"] = stazioni_appaltanti["natura_giuridica_codice"].fillna("UNKNOWN")
stazioni_appaltanti = pd.get_dummies(
    stazioni_appaltanti,
    columns=["natura_giuridica_codice"],
    prefix="ngc_",
    dtype="int8"
)

stazioni_appaltanti.drop(columns=[
    "data_inizio",
    "data_fine",
    "partita_iva",
    "natura_giuridica_descrizione",
    "citta_nome",
    "indirizzo_odonimo",
    "cap",
    "citta_codice",
    "provincia_nome"

], inplace=True)

stazioni_appaltanti.drop_duplicates(inplace=True)
print("--- Il database relativo alle stazioni appaltanti è stato ripulito [1/4]")
stazioni_appaltanti.to_parquet(DATA_PROCESSED_DIR / "stazioni_appaltanti.parquet")
print("--- Il file stazioni_appaltanti.parquet è stato creato [1/4]")
del stazioni_appaltanti

#####################################################################################################################
##AGGIUDICATARI         #############################################################################################
#####################################################################################################################
aggiudicatari = pd.read_csv(DATA_RAW_DIR / "aggiudicatari_csv.csv", na_values = ["N.A", "NA", ""], sep=";")
print("CARICAMENTO [2/4]: file aggiudicatari_csv.csv caricato con successo")
aggiudicatari.drop_duplicates(inplace=True)

#cig
aggiudicatari["cig"] = aggiudicatari["cig"].astype("string")
#codice_fiscale
aggiudicatari.loc[
    aggiudicatari["denominazione"] == "NOVARTIS FARMA SPA",
    "codice_fiscale"
] = "07195130153"
aggiudicatari.loc[
    aggiudicatari["denominazione"] == "SETTEN SRL",
    "codice_fiscale"
] = "03751150263"
aggiudicatari.loc[
    aggiudicatari["denominazione"] == "SOLETO SPA",
    "codice_fiscale"
] = "10640540158"
aggiudicatari.loc[
    aggiudicatari["denominazione"] == "BARATELLA F.LLI SRL",
    "codice_fiscale"
] = "00462790015"
aggiudicatari.loc[
    aggiudicatari["denominazione"] == "TURKISH AIRLINES",
    "codice_fiscale"
] = "00401030580"
aggiudicatari.loc[
    aggiudicatari["denominazione"] == "ACOUSTIQUE & CONSEILS SAS",
    "codice_fiscale"
] = "03427710540"
aggiudicatari.loc[
    aggiudicatari["denominazione"] == "NETMORE GROUP MB ",
    "codice_fiscale"
] = "03603320122"
aggiudicatari.loc[
    aggiudicatari["denominazione"] == "SCOLARO JAPAN LTD",
    "codice_fiscale"
] = "00101390441"
aggiudicatari["codice_fiscale"] = (
    aggiudicatari["codice_fiscale"]
      .astype("string")
      .str.strip()
      .replace("", pd.NA)
      .astype("object")
      .where(lambda s: s.notna(), None)
      )
aggiudicatari = aggiudicatari.dropna(subset=["codice_fiscale"])
#denominazione
aggiudicatari["denominazione"] = (
    aggiudicatari["denominazione"]
      .astype("string")
      .str.strip()
      .replace("", pd.NA)
      .astype("object")
      .where(lambda s: s.notna(), None)
)
#tipo_soggetto
aggiudicatari["tipo_soggetto"] = aggiudicatari["tipo_soggetto"].replace({
    "IMPRESA SINGOLA (IMPRENDITORI INDIVIDUALI, ANCHE ARTIGIANI, SOCIETA COMMERCIALI, SOCIETA COOPERATIVE)": "IMPRESA_SINGOLA",
    "Impresa singola (imprenditori individuali, anche artigiani, societa' commerciali, societa' cooperative)": "IMPRESA_SINGOLA",
    "Monosoggettivo": "MONOSOGGETTIVO",
    "Impresa":"IMPRESA",
    "Consorzio":"CONSORZIO",
    "Stazione appaltante":"STAZIONE_APPALTANTE",
    "STAZIONE APPALTANTE":"STAZIONE_APPALTANTE",
    "DITTA INDIVIDUALE":"DITTA_INDIVIDUALE",
    "Ditta individuale":"DITTA_INDIVIDUALE",
    "CONSORZIO (CONSORZI TRA SOCIETA'COOPERATIVE DI PRODUZIONE E LAVORO; CONSORZI TRA IMPRESE ARTIGIANE; CONSORZI STABILI)" : "CONSORZIO",
    "GEIE (SOGGETTI CHE HANNO STIPULATO IL CONTRATTO DI GRUPPO EUROPEO DI INTERESSE ECONOMICO)" : "GEIE",
    "ATI (RAGGRUPPAMENTI TEMPORANEI DI CONCORRENTI: CONSORZI ORDINARI DI CONCORRENTI)":"ATI",
    "IMPRESA SINGOLA (IMPRENDITORI INDIVIDUALI, ANCHE ARTIGIANI, SOCIETA' COMMERCIALI, SOCIETA' COOPERATIVE)":"IMPRESA_SINGOLA",
    "ASSOCIAZIONE DI CATEGORIA":"ASSOCIAZIONE_DI_CATEGORIA"


})
aggiudicatari["tipo_soggetto"] = aggiudicatari["tipo_soggetto"].fillna("UNKNOWN")
aggiudicatari = pd.get_dummies(
    aggiudicatari,
    columns=["tipo_soggetto"],
    prefix="tipo",
    dtype="int8"
)

aggiudicatari.drop(columns=["ruolo"], inplace=True)

aggiudicatari.drop_duplicates(inplace=True)
print("--- Il database relativo agli aggiudicatari è stato ripulito [2/4]")
aggiudicatari.to_parquet(DATA_PROCESSED_DIR / "aggiudicatari.parquet")
print("--- Il file aggiudicatari.parquet è stato creato [2/4]")
del aggiudicatari

#####################################################################################################################
##AGGIUDICAZIONI         ############################################################################################
#####################################################################################################################
aggiudicazioni = pd.read_csv(DATA_RAW_DIR / "aggiudicazioni_csv.csv", na_values=["N.A.", "NA", ""], sep=";")
print("CARICAMENTO [3/4]: file aggiudicazioni_csv.csv caricato con successo")
aggiudicazioni.drop_duplicates(inplace=True)

#cig
aggiudicazioni["cig"] = aggiudicazioni["cig"].astype("string")
#esito
aggiudicazioni["esito"] = aggiudicazioni["esito"].astype("category")
aggiudicazioni.loc[aggiudicazioni["esito"] == "Aggiudicata", "esito"] = "AGGIUDICATA"
aggiudicazioni = aggiudicazioni.loc[aggiudicazioni["esito"] == "AGGIUDICATA"]
aggiudicazioni = aggiudicazioni.drop(columns=["esito"])
#flag_subappalto
aggiudicazioni["flag_subappalto"] = aggiudicazioni["flag_subappalto"].astype("int8")
#data_comunicazione_esito
s = aggiudicazioni["data_comunicazione_esito"]
parsed = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
mask = parsed.isna() & s.notna()
aggiudicazioni.loc[mask, "data_comunicazione_esito"] = pd.NaT
aggiudicazioni["data_comunicazione_esito"] = pd.to_datetime(aggiudicazioni["data_comunicazione_esito"])
del s
del parsed
del mask
#importo_aggiudicazione
aggiudicazioni = standardize_col(aggiudicazioni, "importo_aggiudicazione")
aggiudicazioni["importo_aggiudicazione_NA"] = (
    aggiudicazioni["importo_aggiudicazione"].isna().astype("int8")
)
aggiudicazioni["importo_aggiudicazione"] = (
    aggiudicazioni["importo_aggiudicazione"].fillna(0)
)
#data_aggiudicazione_definitiva
s = aggiudicazioni["data_aggiudicazione_definitiva"]
parsed = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
mask = parsed.isna() & s.notna()
aggiudicazioni.loc[mask, "data_aggiudicazione_definitiva"] = pd.NaT
aggiudicazioni["data_aggiudicazione_definitiva"] = pd.to_datetime(aggiudicazioni["data_aggiudicazione_definitiva"])
del s
del parsed
del mask
#ribasso_aggiudicazione
aggiudicazioni["ribasso_aggiudicazione"] = abs(aggiudicazioni["ribasso_aggiudicazione"])
aggiudicazioni.loc[aggiudicazioni["ribasso_aggiudicazione"] > 100,"ribasso_aggiudicazione"] = pd.NA
aggiudicazioni = standardize_col(aggiudicazioni, "ribasso_aggiudicazione")
# ribasso_aggiudicazione_NA già gestito sotto? no, ecco:
aggiudicazioni["ribasso_aggiudicazione_NA"] = aggiudicazioni["ribasso_aggiudicazione"].isna().astype("int8")
aggiudicazioni["ribasso_aggiudicazione"] = aggiudicazioni["ribasso_aggiudicazione"].fillna(0)
#asta_elettronica
aggiudicazioni["asta_elettronica"] = aggiudicazioni["asta_elettronica"].fillna(0)
aggiudicazioni["asta_elettronica"] = aggiudicazioni["asta_elettronica"].astype("int8")
#FLAG_PROC_ACCELERATA
aggiudicazioni["FLAG_PROC_ACCELERATA"] = aggiudicazioni["FLAG_PROC_ACCELERATA"].fillna(0)
aggiudicazioni["FLAG_PROC_ACCELERATA"] = aggiudicazioni["FLAG_PROC_ACCELERATA"].astype("int8")
#FLAG_SCOMPUTO
aggiudicazioni["FLAG_SCOMPUTO"] = aggiudicazioni["FLAG_SCOMPUTO"].fillna(0)
aggiudicazioni["FLAG_SCOMPUTO"] = aggiudicazioni["FLAG_SCOMPUTO"].astype("int8")

aggiudicazioni = aggiudicazioni.drop(columns=["numero_offerte_ammesse",
                                              "num_imprese_offerenti",
                                              "num_imprese_richiedenti",
                                              "numero_offerte_escluse",
                                              "num_imprese_invitate",
                                              "N_MANIF_INTERESSE",
                                              "PRESTAZIONI_COMPRESE",
                                              "COD_PRESTAZIONI_COMPRESE",
                                              "criterio_aggiudicazione",
                                              "massimo_ribasso",
                                              "minimo_ribasso",
                                              "DATA_INCARICO_PROG",
                                              "DATA_CONS_PROG",
                                              "CIG_PROG_ESTERNA",
                                              "COD_MODO_RIAGGIUDICAZIONE",
                                              "MODO_RIAGGIUDICAZIONE"])

aggiudicazioni.drop_duplicates(inplace=True)
print("--- Il database relativo alle aggiudicazioni è stato ripulito [3/4]")
aggiudicazioni.to_parquet(DATA_PROCESSED_DIR / "aggiudicazioni.parquet")
print("--- Il file aggiudicazioni.parquet è stato creato [3/4]")
del aggiudicazioni

#####################################################################################################################
##CIG         #######################################################################################################
#####################################################################################################################
cig_2022 = pd.read_csv(DATA_RAW_DIR / "cig_csv_2022.csv", na_values = ["N.A.", "NA", ""], sep=";")
cig_2023 = pd.read_csv(DATA_RAW_DIR / "cig_csv_2023.csv", na_values = ["N.A.", "NA", ""], sep=";")
cig_2024 = pd.read_csv(DATA_RAW_DIR / "cig_csv_2024.csv", na_values = ["N.A.", "NA", ""], sep=";")
cig_2025 = pd.read_csv(DATA_RAW_DIR / "cig_csv_2025.csv", na_values = ["N.A.", "NA", ""], sep=";")
print("CARICAMENTO [4/4]: file cig_csv_2022.csv, cig_csv_2023.csv, cig_csv_2024.csv e cig_csv_2025.csv caricati correttamente")
dfs = [cig_2022, cig_2023, cig_2024, cig_2025]
del cig_2022
del cig_2023
del cig_2024
del cig_2025
cig = pd.concat(dfs, axis=0, ignore_index=True)
del dfs
cig.drop_duplicates(inplace=True)

#ESITO e COD_ESITO
esiti_da_escludere = [
    "DESERTA",
    "ANNULLATA/REVOCATA PRIMA DELL'APERTURA DELLE BUSTE AMMINISTRATIVE",
    "ANNULLATA/REVOCATA DOPO L'APERTURA DELLE BUSTE AMMINISTRATIVE",
    "NON AGGIUDICATA",
    "SENZA ESITO A SEGUITO DI OFFERTE IRREGOLARI O INAMMISSIBILI, NON CONGRUE O NON APPROPRIATE",
    "NON È STATO SELEZIONATO NESSUN VINCITORE E LA GARA È CHIUSA.",
    "PROPOSTA DI AGGIUDICAZIONE",
    "NON È ANCORA STATO SELEZIONATO UN VINCITORE MA LA GARA È ANCORA IN CORSO.",
    "ANNULLATA O REVOCATA SUCCESSIVAMENTE ALLA PUBBLICAZIONE"
]
cig = cig[~cig["ESITO"].isin(esiti_da_escludere)]
cig.drop(columns=["ESITO", "COD_ESITO"], inplace=True)
del esiti_da_escludere
#DENOMINAZIONE_SA_DELEGANTE, DATA_COMUNICAZIONE_ESITO, numero_gara, oggetto_gara, oggetto_lotto, 
#cod_tipo_scelta_contraente, tipo_scelta_contraente, modalita_realizzazione, cod_modalita_realizzazione,
#anno_pubblicazione, mese_pubblicazione, cod_cpv, descrizione_cpv, provincia, luogo_istat,
#COD_MOTIVO_CANCELLAZIONE, MOTIVO_CANCELLAZIONE, DATA_CANCELLAZIONE, DATA_ULTIMO_PERFEZIONAMENTO, 
#COD_MODALITA_INDIZIONE_SPECIALI, MODALITA_INDIZIONE_SPECIALI, COD_MODALITA_INDIZIONE_SERVIZI, MODALITA_INDIZIONE_SERVIZI
#DURATA_PREVISTA, COD_STRUMENTO_SVOLGIMENTO, STRUMENTO_SVOLGIMENTO, TIPO_APPALTO_RISERVATO, CUI_PROGRAMMA
#FLAG_PREV_RIPETIZIONI, DENOMINAZIONE_SA_DELEGATA, MOTIVO_URGENZA, COD_MOTIVO_URGENZA, COD_IPOTESI_COLLEGAMENTO
cig.drop(columns=["DENOMINAZIONE_SA_DELEGANTE",
                  "DATA_COMUNICAZIONE_ESITO",
                  "numero_gara", 
                  "oggetto_gara", 
                  "oggetto_lotto", 
                  "cod_tipo_scelta_contraente", 
                  "tipo_scelta_contraente",
                  "modalita_realizzazione", 
                  "cod_modalita_realizzazione",
                  "anno_pubblicazione", 
                  "mese_pubblicazione",
                  "cod_cpv", 
                  "descrizione_cpv",
                  "provincia", 
                  "luogo_istat",
                  "COD_MOTIVO_CANCELLAZIONE", 
                  "MOTIVO_CANCELLAZIONE", 
                  "DATA_CANCELLAZIONE", 
                  "DATA_ULTIMO_PERFEZIONAMENTO", 
                  "COD_MODALITA_INDIZIONE_SPECIALI", 
                  "MODALITA_INDIZIONE_SPECIALI", 
                  "COD_MODALITA_INDIZIONE_SERVIZI", 
                  "MODALITA_INDIZIONE_SERVIZI",
                  "DURATA_PREVISTA", 
                  "COD_STRUMENTO_SVOLGIMENTO", 
                  "STRUMENTO_SVOLGIMENTO",
                  "TIPO_APPALTO_RISERVATO", 
                  "CUI_PROGRAMMA",
                  "FLAG_PREV_RIPETIZIONI",
                  "DENOMINAZIONE_SA_DELEGATA",
                  "MOTIVO_URGENZA",
                  "COD_MOTIVO_URGENZA",
                  "COD_IPOTESI_COLLEGAMENTO"], inplace=True)
#cig
cig["cig"] = cig["cig"].astype("string")
#cig_accordo_quadro
cig["cig_aq"] = cig["cig_accordo_quadro"].notna().astype("int8")
cig.drop(columns = ["cig_accordo_quadro"], inplace=True)
#importo_complessivo_gara
cig = standardize_col(cig, "importo_complessivo_gara")
cig["importo_complessivo_gara_NA"] = (
    cig["importo_complessivo_gara"].isna().astype("int8")
)
cig["importo_complessivo_gara"] = (
    cig["importo_complessivo_gara"].fillna(0))
#n_lotti_componenti
cig = standardize_col(cig, "n_lotti_componenti")
cig["n_lotti_componenti_NA"] = (
    cig["n_lotti_componenti"].isna().astype("int8")
)
cig["n_lotti_componenti"] = (
    cig["n_lotti_componenti"].fillna(0))
#importo_lotto
cig = standardize_col(cig, "importo_lotto")
cig["importo_lotto_NA"] = (
    cig["importo_lotto"].isna().astype("int8")
)
cig["importo_lotto"] = (
    cig["importo_lotto"].fillna(0))
#oggetto_principale_contratto
cig = pd.get_dummies(
    cig,
    columns=["oggetto_principale_contratto"],
    prefix="opc",
    dtype="int8"
)
#stato
cig["stato"] = (cig["stato"] == "ATTIVO").astype("int8")
#settore
cig["settore"] = (cig["settore"] == "SETTORI ORDINARI").astype("int8")
#data_pubblicazione
s = cig["data_pubblicazione"]
parsed = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
mask = parsed.isna() & s.notna()
cig.loc[mask, "data_pubblicazione"] = pd.NaT
cig["data_pubblicazione"] = pd.to_datetime(cig["data_pubblicazione"])
del s
del parsed
del mask
#data_scadenza_offerta
s = cig["data_scadenza_offerta"]
parsed = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
mask = parsed.isna() & s.notna()
cig.loc[mask, "data_scadenza_offerta"] = pd.NaT
cig["data_scadenza_offerta"] = pd.to_datetime(cig["data_scadenza_offerta"])
#data_scadenza_offerta
cig["data_scadenza_offerta"] = (
    cig["data_scadenza_offerta"] - cig["data_pubblicazione"]
).dt.days
cig = standardize_col(cig, "data_scadenza_offerta")
cig["data_scadenza_offerta_NA"] = (
    cig["data_scadenza_offerta"].isna().astype("int8")
)
cig["data_scadenza_offerta"] = (
    cig["data_scadenza_offerta"].fillna(0))
#codice_ausa
cig["codice_ausa"] = fix_codice_ausa(cig["codice_ausa"])
#cf_amministrazione_appaltante
cig["cf_amministrazione_appaltante"] = (
    cig["cf_amministrazione_appaltante"]
      .astype("string")          
      .str.strip()
      .replace("", pd.NA)
)
#sezione_regionale
cig["sezione_regionale"] = cig["sezione_regionale"].astype("category")
cig["sezione_regionale"] = cig["sezione_regionale"].fillna("NON CLASSIFICATO")
cig = pd.get_dummies(
    cig,
    columns=["sezione_regionale"],
    prefix="sr",
    dtype="int8"
)
#flag_prevalente
cig["flag_prevalente"] = cig["flag_prevalente"].astype("int8")
#FLAG_URGENZA
cig["FLAG_URGENZA"] = cig["FLAG_URGENZA"].fillna(0)
cig["FLAG_URGENZA"] = cig["FLAG_URGENZA"].astype("int8")
#FLAG_DELEGA
cig["FLAG_DELEGA"] = cig["FLAG_DELEGA"].fillna(0)
cig["FLAG_DELEGA"] = cig["FLAG_DELEGA"].astype("int8")
#FUNZIONI_DELEGATE
cig["FUNZIONI_DELEGATE"] = cig["FUNZIONI_DELEGATE"].replace({
    "Solo aggiudicazione":"SOLO AGGIUDICAZIONE",
    "Proposta di aggiudicazione":"PROPOSTA DI AGGIUDICAZIONE",
    "Aggiudicazione, stipula ed esecuzione del contratto":"AGGIUDICAZIONE, STIPULA ED ESECUZIONE DEL CONTRATTO",
    "Aggiudicazione e stipula del contratto":"AGGIUDICAZIONE E STIPULA DEL CONTRATTO"
    })
cig.loc[
    (cig["FLAG_DELEGA"] == 1) & (cig["FUNZIONI_DELEGATE"].isna()),
    "FUNZIONI_DELEGATE"
] = "UNKNOWN"
cig.loc[
    cig["FLAG_DELEGA"] == 0,
    "FUNZIONI_DELEGATE"
] = "NON_DELEGATA"
cig = pd.get_dummies(
    cig,
    columns=["FUNZIONI_DELEGATE"],
    prefix="FD",
    dtype="int8"
)
#CF_SA_DELEGATA
cig["CF_SA_DELEGATA"] = (
    cig["CF_SA_DELEGATA"]
      .astype("string")
      .str.strip()
      .replace("", pd.NA)
      .astype("object")
      .where(lambda s: s.notna(), None)
)
#CF_SA_DELEGANTE
cig["CF_SA_DELEGANTE"] = (
    cig["CF_SA_DELEGANTE"]
      .astype("string")
      .str.strip()
      .replace("", pd.NA)
      .astype("object")
      .where(lambda s: s.notna(), None)
)
#IMPORTO_SICUREZZA
cig = standardize_col(cig, "IMPORTO_SICUREZZA")
cig["IMPORTO_SICUREZZA_NA"] = (
    cig["IMPORTO_SICUREZZA"].isna().astype("int8")
)
cig["IMPORTO_SICUREZZA"] = cig["IMPORTO_SICUREZZA"].fillna(0)
#IPOTESI_COLLEGAMENTO
cig["IPOTESI_COLLEGAMENTO"] = cig["IPOTESI_COLLEGAMENTO"].replace({
    "Ripetizione di lavori o servizi analoghi":"RIPETIZIONE DI LAVORI O SERVIZI ANALOGHI",
    "No, nessuna ipotesi di collegamento":"NO, NESSUNA IPOTESI DI COLLEGAMENTO",
    "Procedura a seguito di precedente gara annullata o deserta (senza esito)":"PROCEDURA A SEGUITO DI PRECEDENTE REVOCA DI AGGIUDICAZIONE O DI RESCISSIONE CONTRATTUALE, SENZA SCORRIMENTO GRADUATORIA",
    "Affidamento a seguito di appalto pre-commerciale":"AFFIDAMENTO A SEGUITO DI APPALTO PRE-COMMERCIALE",
    "Affidamento dell’esecuzione o della progettazione dei lavori successiva a progettazione o livello inferiore di progettazione":"AFFIDAMENTO DELL’ESECUZIONE O DELLA PROGETTAZIONE DEI LAVORI SUCCESSIVA A PROGETTAZIONE O LIVELLO INFERIORE DI PROGETTAZIONE",
    "Affidamento da parte di aggiudicatari di contratti di concessione o di project financing":"AFFIDAMENTO DA PARTE DI AGGIUDICATARI DI CONTRATTI DI CONCESSIONE O DI PROJECT FINANCING"
    })
cig["IPOTESI_COLLEGAMENTO"] = cig["IPOTESI_COLLEGAMENTO"].fillna("NO, NESSUNA IPOTESI DI COLLEGAMENTO")
cig = pd.get_dummies(
    cig,
    columns=["IPOTESI_COLLEGAMENTO"],
    prefix="IC",
    dtype="int8"
)
#CIG_COLLEGAMENTO
cig["CIG_COLLEGAMENTO"] = (
    cig["CIG_COLLEGAMENTO"]
      .astype("string")
      .str.strip()
      .str.replace(r"\.0+$", "", regex=True)
      .replace("", pd.NA)
      .astype("object")
      .where(lambda s: s.notna(), None)
)
#FLAG_PNRR_PNC
cig["FLAG_PNRR_PNC"] = cig["FLAG_PNRR_PNC"].fillna(0)
cig["FLAG_PNRR_PNC"] = cig["FLAG_PNRR_PNC"].astype("int8")

cig.drop_duplicates(inplace=True)
print("--- Il database relativo ai bandi pubblici è stato ripulito [4/4]")
cig.to_parquet(DATA_PROCESSED_DIR / "CIG.parquet")
print("--- Il file CIG.parquet è stato creato [4/4]")
del cig

#stazioni_appaltanti, oltre alle variabili numeriche, contiene codice_fiscale, denominazione e codice_ausa
#dopo aver eliminato le unità di cui non possiamo creare archi, creare dizionari
#id -> codice_fiscale, denominazione, codice_ausa
#codice_ausa -> id

#aggiudicatari, oltre alle variabili numeriche, contiene cig, codice_fiscale, denominazione e id_aggiudicazione
#dopo aver eliminato le unità di cui non possiamo creare archi, creare dizionari
#id -> codice_fiscale, cig, denominazione e id_aggiudicazione
#id_aggiudicazione -> id

#aggiudicazioni, oltre alle variabili numeriche, contiene id_aggiudicazione, cig, data_aggiudicazione_definitiva e data_comunicazione_esito
#dopo aver eliminato le unità di cui non possiamo creare archi, creare dizionari
#id -> id_aggiudicazione, cig
#id_aggiudicazione -> id
#inoltre trasformare le variabili data_aggiudicazione_definitiva e data_comunicazione_esito da date a numeriche
#come differenza (in giorni) tra l'evento che rappresentano e la data_pubblicazione del corrispondente bando contenuto nel dataframe cig

#cig, oltre alle variabili numeriche, contiene cig, codice_ausa, id_centro_costo, CIG_COLLEGAMENTO, CF_SA_DELEGATA, CF_SA_DELEGANTE e data_pubblicazione
#dopo aver eliminato le unità di cui non possiamo creare archi, creare dizionari
#id -> cig, codice_ausa, id_centro_costo, CIG_COLLEGAMENTO, CF_SA_DELGATA, CF_SA_DELEGANTE
#cig -> id
#utilizzare la variabile data_pubblicazione per trasformare in intero le due variabili date nel database aggiudicazioni, poi eliminarla

with open(MODELS_DIR / "scaler_params.json", "w") as f:
    json.dump(SCALER_PARAMS, f)
print("--- Il file scaler_params.json è stato creato")

print("PREPROCESSING COMPLETATO CON SUCCESSO")

print("----- MERGING DATABASES -----")
#####################################################################################################################
##            #######################################################################################################
#####################################################################################################################
aggiudicazioni = pd.read_parquet(DATA_PROCESSED_DIR / "aggiudicazioni.parquet")
print("--- file aggiudicazioni.parquet caricato correttamente [1/4]")
cig = pd.read_parquet(DATA_PROCESSED_DIR / "CIG.parquet")
print("--- file cig.parquet caricato correttamente [2/4]")
aggiudicatari = pd.read_parquet(DATA_PROCESSED_DIR / "aggiudicatari.parquet")
print("--- file aggiudicatari.parquet caricato correttamente [3/4]")
stazioni_appaltanti = pd.read_parquet(DATA_PROCESSED_DIR / "stazioni_appaltanti.parquet")
print("--- file stazioni_appaltanti.parquet caricato correttamente [4/4]")


# =========================
# 1. (id_aggiudicazione, cig) validi:
#    devono esistere sia in aggiudicazioni sia in aggiudicatari
# =========================

print("JOIN [1/5]: aggiudicatari --(cig, id_aggiudicazione)--> aggiudicazioni")

valid_id_aggiudicazione = set(
    map(tuple, aggiudicazioni[["id_aggiudicazione", "cig"]].values)
).intersection(
    set(map(tuple, aggiudicatari[["id_aggiudicazione", "cig"]].values))
)
aggiudicazioni = aggiudicazioni[
    aggiudicazioni[["id_aggiudicazione", "cig"]].apply(tuple, axis=1).isin(valid_id_aggiudicazione)
]
aggiudicatari = aggiudicatari[
    aggiudicatari[["id_aggiudicazione", "cig"]].apply(tuple, axis=1).isin(valid_id_aggiudicazione)
]
del valid_id_aggiudicazione

#all'interno di aggiudicazioni ci sono 76 coppie di unità che non risultano essere duplicati, 
#solo perchè alcune delle loro modalità sono stati inserite in maniera errata
#tuttavia la coppia di identificazione univoca (id_aggiudicazione, cig) è uguale
#le ispezioniamo una ad una per eliminare quelle corrotte e mantenere quelle migliori
#sistemiamo tutti i duplicati all'interno di aggiudicazioni
dup_keys = (
    aggiudicazioni
    .groupby(["id_aggiudicazione", "cig"])
    .size()
    .reset_index(name="n")
    .query("n > 1")
)
dup_rows = aggiudicazioni.merge(
    dup_keys[["id_aggiudicazione", "cig"]],
    on=["id_aggiudicazione", "cig"],
    how="inner"
)
del dup_keys
rows_to_remove = dup_rows.loc[[87, 138, 41, 94, 95, 15, 28, 45, 70, 71, 123, 82, 55, 99, 149, 132, 8, 143, 104, 53,
                           30, 120, 144, 61, 102, 117, 76, 110, 47, 125, 24, 66, 38, 137, 98, 75, 77, 119, 13,
                           115, 73, 86, 20, 130, 9, 2, 109, 116, 25, 118, 100, 103, 84, 44, 14, 114, 85, 121,
                           122, 46, 129, 81, 79, 57, 146, 11, 34, 1, 39, 127, 136, 131, 68, 96, 63, 4]]
del dup_rows
mask_update = (
    (aggiudicazioni["cig"] == "B06B4DAC10") &
    (aggiudicazioni["id_aggiudicazione"] == 18581558) 
)
aggiudicazioni.loc[mask_update, "data_aggiudicazione_definitiva"] = pd.Timestamp("2024-09-28")
del mask_update
key_cols = aggiudicazioni.columns.tolist()
rows_to_remove = rows_to_remove[key_cols].drop_duplicates()
aggiudicazioni = aggiudicazioni.merge(
    rows_to_remove.assign(_remove=1),
    on=key_cols,
    how="left"
)
del key_cols
del rows_to_remove
aggiudicazioni = aggiudicazioni[aggiudicazioni["_remove"].isna()].drop(columns="_remove")

# =========================
# 2. cig validi:
#    devono esistere sia in cig sia nelle aggiudicazioni filtrate
# =========================

print("JOIN [2/5]:  aggiudicazioni --(cig)--> bandi")
cig = cig.drop_duplicates(subset="cig", keep="first")

valid_cig = set(cig["cig"]).intersection(set(aggiudicazioni["cig"]))

cig = cig[cig["cig"].isin(valid_cig)]
aggiudicazioni = aggiudicazioni[aggiudicazioni["cig"].isin(valid_cig)]
aggiudicatari = aggiudicatari[aggiudicatari["cig"].isin(valid_cig)]
del valid_cig

# =========================
# 3. codice_ausa validi:
#    devono esistere sia in stazioni_appaltanti sia in cig filtrato
# =========================
print("JOIN [3/5]:  bandi --(codice_ausa)--> stazioni appaltanti")
valid_codice_ausa = set(stazioni_appaltanti["codice_ausa"]).intersection(
    set(cig["codice_ausa"])
)

stazioni_appaltanti = stazioni_appaltanti[
    stazioni_appaltanti["codice_ausa"].isin(valid_codice_ausa)
]

cig = cig[cig["codice_ausa"].isin(valid_codice_ausa)]
del valid_codice_ausa

# =========================
# 4. Rifinitura finale:
#    dopo il filtro su codice_ausa, i cig validi potrebbero ridursi ancora
#    e, di conseguenza, anche gli aggiudicatari e aggiudicazioni
# =========================
print("JOIN [4/5]:  bandi --(codice_ausa)--> aggiudicazioni")
valid_cig_final = set(cig["cig"]).intersection(set(aggiudicazioni["cig"]))

cig = cig[cig["cig"].isin(valid_cig_final)]
aggiudicazioni = aggiudicazioni[aggiudicazioni["cig"].isin(valid_cig_final)]

print("JOIN [5/5]:  aggiudicazioni --(codice_ausa)--> aggiudicatari")
aggiudicatari = aggiudicatari[aggiudicatari["cig"].isin(valid_cig_final)]
del valid_cig_final

#####################################################################################################################
##DATE         ######################################################################################################
#####################################################################################################################
aggiudicazioni = aggiudicazioni.merge(
    cig[["cig", "data_pubblicazione"]],
    on="cig",
    how="left"
)
s = aggiudicazioni["data_pubblicazione"]
parsed = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
mask = parsed.isna() & s.notna()
aggiudicazioni.loc[mask, "data_pubblicazione"] = pd.NaT
aggiudicazioni["data_pubblicazione"] = pd.to_datetime(aggiudicazioni["data_pubblicazione"])
del s
del parsed
del mask
#data_comunicazione_esito
s = aggiudicazioni["data_comunicazione_esito"]
parsed = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
mask = parsed.isna() & s.notna()
aggiudicazioni.loc[mask, "data_comunicazione_esito"] = pd.NaT
aggiudicazioni["data_comunicazione_esito"] = pd.to_datetime(aggiudicazioni["data_comunicazione_esito"])
del s
del parsed
del mask
aggiudicazioni.loc[
    aggiudicazioni["data_comunicazione_esito"] < pd.Timestamp("2000-01-01"),
    "data_comunicazione_esito"
] = pd.NaT
#data_comunicazione_esito
aggiudicazioni["data_comunicazione_esito"] = (
    aggiudicazioni["data_comunicazione_esito"] - aggiudicazioni["data_pubblicazione"]
).dt.days
aggiudicazioni = standardize_col(aggiudicazioni, "data_comunicazione_esito")
aggiudicazioni["data_comunicazione_esito_NA"] = (
    aggiudicazioni["data_comunicazione_esito"].isna().astype("int8")
)
aggiudicazioni["data_comunicazione_esito"] = (
    aggiudicazioni["data_comunicazione_esito"].fillna(0))
#data_aggiudicazione_definitiva
s = aggiudicazioni["data_aggiudicazione_definitiva"]
parsed = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
mask = parsed.isna() & s.notna()
aggiudicazioni.loc[mask, "data_aggiudicazione_definitiva"] = pd.NaT
aggiudicazioni["data_aggiudicazione_definitiva"] = pd.to_datetime(aggiudicazioni["data_aggiudicazione_definitiva"])
del s
del parsed
del mask
aggiudicazioni.loc[
    aggiudicazioni["data_aggiudicazione_definitiva"] < pd.Timestamp("2000-01-01"),
    "data_aggiudicazione_definitiva"
] = pd.NaT
#data_aggiudicazione_definitiva
aggiudicazioni["data_aggiudicazione_definitiva"] = (
    aggiudicazioni["data_aggiudicazione_definitiva"] - aggiudicazioni["data_pubblicazione"]
).dt.days
aggiudicazioni = standardize_col(aggiudicazioni, "data_aggiudicazione_definitiva")
aggiudicazioni["data_aggiudicazione_definitiva_NA"] = (
    aggiudicazioni["data_aggiudicazione_definitiva"].isna().astype("int8")
)
aggiudicazioni["data_aggiudicazione_definitiva"] = (
    aggiudicazioni["data_aggiudicazione_definitiva"].fillna(0))
aggiudicazioni.drop(columns=["data_pubblicazione"], inplace=True)
cig.drop(columns=["data_pubblicazione"], inplace=True)

print("----- BUILDING IDENTIFIERS -----")
#####################################################################################################################
##DIZIONARI         #################################################################################################
#####################################################################################################################
print("DIZIONARIO [1/8] STAZIONI APPALTANTI: id -> codice_fiscale, denominazione, codice_ausa")
#creiamo un id numerico per ciascuna stazione appaltante
stazioni_appaltanti = stazioni_appaltanti.reset_index(drop=True)
stazioni_appaltanti["id"] = stazioni_appaltanti.index
#creaimo un dizionario per poter sempre risalire al codice_fiscale e alla denominazione 
#dei nodi relativi alle stazioni appaltanti
SA_id = {
    row.id: {
        "codice_fiscale": row.codice_fiscale,
        "denominazione": row.denominazione,
        "codice_ausa": row.codice_ausa
    }
    for row in stazioni_appaltanti.itertuples(index=False)
}
print("DIZIONARIO [2/8] STAZIONI APPALTANTI: codice_ausa -> id")
SA_ca = {
    row.codice_ausa: row.id
    for row in stazioni_appaltanti.itertuples(index=False)
}
stazioni_appaltanti = stazioni_appaltanti.drop(
    columns=["codice_fiscale", "denominazione", "codice_ausa"]
)

stazioni_appaltanti.to_parquet(DATA_PROCESSED_DIR / "stazioni_appaltanti_v2.parquet")
print("--- Il file stazioni_appaltanti_v2.parquet è stato creato")

with open(DATA_PROCESSED_DIR / "SA_id.json", "w") as f:
    json.dump(SA_id, f)
print("--- Il file SA_id.json è stato creato")

with open(DATA_PROCESSED_DIR / "SA_ca.json", "w") as f:
    json.dump(SA_ca, f)
print("--- Il file SA_ca.json è stato creato")
del stazioni_appaltanti
del SA_id
del SA_ca


aggiudicatari_links = aggiudicatari[["codice_fiscale", "cig", "id_aggiudicazione"]]
aggiudicatari_links = aggiudicatari_links.drop_duplicates()

tipo_cols = [c for c in aggiudicatari.columns if c.startswith("tipo_")]
aggiudicatari = aggiudicatari[["codice_fiscale", "denominazione"] + tipo_cols].copy()
aggiudicatari = (
    aggiudicatari
    .groupby("codice_fiscale", dropna=True, as_index=False)
    .agg({"denominazione": "first", **{c: "max" for c in tipo_cols}})
)

# creiamo un id numerico per ciascun aggiudicatario
aggiudicatari = aggiudicatari.reset_index(drop=True)
aggiudicatari["id"] = aggiudicatari.index
#creaimo un dizionario per poter sempre risalire al cig, al codice_fiscale, alla denominazione e all'id_aggiudicazione
#dei nodi relativi agli aggiudicatari
print("DIZIONARIO [3/8] AGGIUDICATARI: id -> codice_fiscale, denominazione")
AG_id = {
    row.id: {
        "codice_fiscale": row.codice_fiscale,
        "denominazione": row.denominazione
    }
    for row in aggiudicatari.itertuples(index=False)
}

print("DIZIONARIO [4/8] AGGIUDICATARI: codice_fiscale -> id")
AG_cf = {
    row.codice_fiscale : row.id
    for row in aggiudicatari.itertuples(index=False)
}

aggiudicatari = aggiudicatari.drop(
    columns=["codice_fiscale", "denominazione"]
)

aggiudicatari.to_parquet(DATA_PROCESSED_DIR / "aggiudicatari_v2.parquet")
print("--- Il file aggiudicatari_v2.parquet è stato creato")

with open(DATA_PROCESSED_DIR / "AG_id.json", "w") as f:
    json.dump(AG_id, f)
print("--- Il file AG_id.json è stato creato")

with open(DATA_PROCESSED_DIR / "AG_cf.json", "w") as f:
    json.dump(AG_cf, f)
print("--- Il file AG_cf.json è stato creato")

del aggiudicatari
del AG_id
del AG_cf

# creiamo un id numerico per ciascuna aggiudicazione
aggiudicazioni = aggiudicazioni.reset_index(drop=True)
aggiudicazioni["id"] = aggiudicazioni.index
#creaimo un dizionario per poter sempre risalire al cig e all'id_aggiudicazione delle aggiudicazioni
print("DIZIONARIO [5/8] AGGIUDICAZIONI: id -> cig, id_aggiudicazione")
AGZ_id = {
    row.id: {
        "cig": row.cig,
        "id_aggiudicazione": row.id_aggiudicazione
    }
    for row in aggiudicazioni.itertuples(index=False)
}
print("DIZIONARIO [6/8] AGGIUDICAZIONI: id_aggiudicazione_cig -> id")
AGZ_idag = {
    f"{row.id_aggiudicazione}_{row.cig}": row.id
    for row in aggiudicazioni.itertuples(index=False)
}

aggiudicazioni = aggiudicazioni.drop(
    columns=["cig", "id_aggiudicazione"]
)

aggiudicazioni.to_parquet(DATA_PROCESSED_DIR / "aggiudicazioni_v2.parquet")
print("--- Il file aggiudicazioni_v2.parquet è stato creato")

with open(DATA_PROCESSED_DIR / "AGZ_id.json", "w") as f:
    json.dump(AGZ_id, f)
print("--- Il file AGZ_id.json è stato creato")

with open(DATA_PROCESSED_DIR / "AGZ_idag.json", "w") as f:
    json.dump(AGZ_idag, f)
print("--- Il file AGZ_idag.json è stato creato")
del aggiudicazioni
del AGZ_id
del AGZ_idag



cig = cig.reset_index(drop=True)
cig["id"] = cig.index
print("DIZIONARIO [7/8] CIG: id -> cig, codice_ausa, id_centro_costo, CIG_COLLEGAMENTO, CF_SA_DELEGATA, CF_SA_DELEGANTE")
CIG_id = {
    row.id:{
        "cig" : row.cig,
        "codice_ausa" : row.codice_ausa,
        "id_centro_costo" : row.id_centro_costo,
        "CIG_COLLEGAMENTO" : row.CIG_COLLEGAMENTO,
        "CF_SA_DELEGATA" : row.CF_SA_DELEGATA,
        "CF_SA_DELEGANTE" : row.CF_SA_DELEGANTE
    }
    for row in cig.itertuples(index=False)
}
print("DIZIONARIO [8/8] CIG: cig -> id")           
CIG_cig = {
    row.cig : row.id
    for row in cig.itertuples(index=False)
}

cig = cig.drop(
    columns=["cig", "CIG_COLLEGAMENTO", "CF_SA_DELEGATA", "CF_SA_DELEGANTE", "codice_ausa",
             "cf_amministrazione_appaltante", "denominazione_amministrazione_appaltante",
             "id_centro_costo", "denominazione_centro_costo"]
)
cig.to_parquet(DATA_PROCESSED_DIR / "CIG_v2.parquet")
print("--- Il file CIG_v2.parquet è stato creato")

with open(DATA_PROCESSED_DIR / "CIG_id.json", "w") as f:
    json.dump(CIG_id, f)
print("--- Il file CIG_id.json è stato creato")

with open(DATA_PROCESSED_DIR / "CIG_cig.json", "w") as f:
    json.dump(CIG_cig, f)
print("--- Il file CIG_cig.json è stato creato")
del cig
del CIG_id
del CIG_cig
del f


aggiudicatari_links.to_parquet(DATA_PROCESSED_DIR / "aggiudicatari_links.parquet")
print("--- Il file aggiudicatari_links.parquet è stato creato")