import torch
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import networkx as nx
from collections import Counter
import os

from dash import Dash, html, dcc, Input, Output, callback, State, no_update
import dash_bootstrap_components as dbc

# --- UTILS ---
def load_json_int_keys(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return {int(k): v for k, v in d.items()}

from pathlib import Path

# --- DEFINIZIONE PERCORSI RELATIVI ---
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_PATH = BASE_DIR / "models"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed"

# --- DATA LOADING ---
print("Caricamento dati in corso...")
PA = torch.load(f"{MODELS_PATH}/PA_graph.pt", weights_only=False)
SA_id = load_json_int_keys(f"{PROCESSED_DATA_PATH}/SA_id.json")
CIG_id = load_json_int_keys(f"{PROCESSED_DATA_PATH}/CIG_id.json")
AGZ_id = load_json_int_keys(f"{PROCESSED_DATA_PATH}/AGZ_id.json")
AG_id = load_json_int_keys(f"{PROCESSED_DATA_PATH}/AG_id.json")

# Caricamento nomi colonne feature per visualizzazione completa
def get_columns(path):
    if not os.path.exists(path): return []
    cols = pd.read_parquet(path, engine='pyarrow').columns.tolist()
    # RIMUOVIAMO 'id' perché non è presente nei tensori x del grafo
    return [c for c in cols if c != 'id']

print("Caricamento metadati feature...")
SA_cols = get_columns(f"{PROCESSED_DATA_PATH}/stazioni_appaltanti_v2.parquet")
CIG_cols = get_columns(f"{PROCESSED_DATA_PATH}/CIG_v2.parquet")
AGZ_cols = get_columns(f"{PROCESSED_DATA_PATH}/aggiudicazioni_v2.parquet")
AG_cols = get_columns(f"{PROCESSED_DATA_PATH}/aggiudicatari_v2.parquet")

# Aggregazione per Macro-View
edge_sa_cig = PA["SA", "publishes", "CIG"].edge_index.numpy()
edge_cig_agz = PA["CIG", "has_award", "AGZ"].edge_index.numpy()
edge_agz_ag = PA["AGZ", "awarded_to", "AG"].edge_index.numpy()
sa_cig_counts = Counter(edge_sa_cig[0])

# Prepariamo i dati delle SA per la mappa macro (campionamento per fluidità iniziale)
MIN_CIGS = 5
active_sa_ids = [sa for sa, count in sa_cig_counts.items() if count >= MIN_CIGS]
display_sa_ids = active_sa_ids[:3000]

df_macro = pd.DataFrame([
    {
        "id": sa_id,
        "denom": SA_id.get(sa_id, {}).get("denominazione", f"ID {sa_id}"),
        "count": sa_cig_counts[sa_id]
    }
    for sa_id in display_sa_ids
])

# Layout Macro
angles = np.linspace(0, 2*np.pi, len(df_macro), endpoint=False)
radii = 10 + np.sqrt(df_macro["count"]) * 1.5
df_macro["x"] = radii * np.cos(angles)
df_macro["y"] = radii * np.sin(angles)

# --- DASH APP ---
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    # DASHBOARD PRINCIPALE
    dbc.Row([
        # Grafici a sinistra (Colonna 9)
        dbc.Col([
            # LIVELLO 1: SA
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5("1. Mappa Stazioni Appaltanti (SA)", className="mb-0 d-inline-block"),
                            html.Div([
                                dcc.Input(
                                    id='sa-search-input', type='text', placeholder='Cerca SA...',
                                    className="form-control form-control-sm", debounce=True
                                )
                            ], className="float-end", style={'width': '250px', 'marginTop': '-5px'})
                        ]),
                        dbc.CardBody([
                            dcc.Loading(children=dcc.Graph(id='macro-graph', style={'height': '55vh'}))
                        ])
                    ], className="mb-4 shadow-sm")
                ], width=12)
            ]),
            
            # LIVELLO 2: CIG
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H5("2. Tutti i Bandi (CIG) della SA selezionata", className="mb-0")),
                        dbc.CardBody([
                            html.Div(id='cig-info', className="mb-2 text-primary font-weight-bold"),
                            dcc.Loading(children=dcc.Graph(id='cig-graph', style={'height': '45vh'}))
                        ])
                    ], className="mb-4 shadow-sm")
                ], width=12)
            ], id='level-2-row', style={'display': 'none'}),
            
            # LIVELLO 3: VINCITORI
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H5("3. Vincitori e Aggiudicatari del Bando", className="mb-0")),
                        dbc.CardBody([
                            html.Div(id='awards-info', className="mb-2 text-success font-weight-bold"),
                            dcc.Loading(children=dcc.Graph(id='awards-graph', style={'height': '35vh'}))
                        ])
                    ], className="mb-5 shadow-sm")
                ], width=12)
            ], id='level-3-row', style={'display': 'none'}),
        ], width=9),

        # Pannello Informativo a destra (Colonna 3) - Sticky
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("🔍 Dettagli Nodo", className="mb-0")),
                dbc.CardBody([
                    html.Div(id='detail-panel-content', children=[
                        html.P("Clicca su un nodo per vedere i dettagli.", className="text-muted italic")
                    ])
                ], style={'maxHeight': '80vh', 'overflowY': 'auto'})
            ], className="shadow-sm border-primary sticky-top", style={'top': '20px'})
        ], width=3)
    ]),

    dcc.Store(id='sa-store'),
    dcc.Store(id='cig-store')
], fluid=True, style={'backgroundColor': '#f8f9fa', 'padding': '20px'})

# CALLBACKS

@app.callback(
    Output('macro-graph', 'figure'),
    Input('sa-search-input', 'value')
)
def update_macro(search_value):
    filtered_df = df_macro.copy()
    
    if search_value:
        filtered_df = filtered_df[filtered_df['denom'].str.contains(search_value, case=False, na=False)]
        
    if filtered_df.empty:
        fig = go.Figure()
        fig.update_layout(template="plotly_white", title="Nessuna Stazione Appaltante trovata")
        return fig

    # Usiamo il logaritmo per il colore per dare risalto alle differenze tra piccoli valori
    log_counts = np.log10(filtered_df["count"])
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=filtered_df["x"], y=filtered_df["y"],
        mode='markers',
        marker=dict(
            size=10, 
            color=log_counts, 
            colorscale='Viridis',
            reversescale=True,
            showscale=True,
            colorbar=dict(
                title="CIG (Log)",
                tickvals=[1, 2, 3, 4, 5],
                ticktext=["10", "100", "1k", "10k", "100k"]
            ),
            line=dict(width=0.5, color='white')
        ),
        text=filtered_df["denom"]
    ))
    
    # Per mostrare il valore reale nel tooltip mentre il colore è logaritmico:
    fig.data[0].marker.color = log_counts
    # Usiamo customdata per passare SA_ID e Conteggio
    fig.data[0].customdata = np.stack((filtered_df["id"], filtered_df["count"]), axis=-1).tolist()
    fig.data[0].hovertemplate = (
            "<b>%{text}</b><br>" +
            "ID SA: %{customdata[0]}<br>" +
            "Contratti (CIG): %{customdata[1]:,}<br>" +
            "<extra></extra>"
    )

    fig.update_layout(
        template="plotly_white", 
        margin=dict(l=0, r=0, t=40, b=0), 
        clickmode='event+select',
        hovermode='closest',
        xaxis=dict(title="Distribuzione Spaziale", showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(title="Distanza dal centro = Volume Contratti", showgrid=False, zeroline=False, showticklabels=False),
        annotations=[
            dict(
                text="PIÙ LONTANO DAL CENTRO = PIÙ CONTRATTI (CIG)",
                xref="paper", yref="paper", x=0.5, y=0.02, showarrow=False,
                font=dict(size=10, color="gray")
            )
        ]
    )
    return fig

@app.callback(
    [Output('cig-graph', 'figure'),
     Output('cig-info', 'children'),
     Output('level-2-row', 'style'),
     Output('sa-store', 'data')],
    Input('macro-graph', 'clickData')
)
def update_cigs(clickData):
    if not clickData: return no_update, "", {'display': 'none'}, None
    
    sa_id = clickData['points'][0]['customdata']
    # Se customdata è una lista [id, count], prendiamo il primo elemento
    if isinstance(sa_id, list): 
        sa_id = sa_id[0]
    
    # Debug print per sicurezza
    print(f"DEBUG: Selezionata SA ID: {sa_id}")
    
    mask = edge_sa_cig[0] == sa_id
    c_ids = edge_sa_cig[1, mask].tolist()
    
    # Layout a NUBE per migliaia di CIG
    n = len(c_ids)
    # Generiamo coordinate a spirale o random per visualizzarne molti
    indices = np.arange(n)
    phi = indices * np.pi * (3 - np.sqrt(5))  # golden angle
    r = np.sqrt(indices) / np.sqrt(n)
    cx = r * np.cos(phi)
    cy = r * np.sin(phi)
    
    labels = [CIG_id.get(int(cid), {}).get('cig', f"ID {cid}") for cid in c_ids]
    
    fig = go.Figure()
    # Punto centrale SA
    fig.add_trace(go.Scatter(x=[0], y=[0], mode='markers', marker=dict(size=20, color='red'), name="SA"))
    
    # Tutti i CIG come Scattergl per velocità
    fig.add_trace(go.Scattergl(
        x=cx, y=cy,
        mode='markers',
        marker=dict(size=6, color='#1f77b4', opacity=0.7),
        text=labels,
        customdata=c_ids,
        hovertemplate="<b>CIG: %{text}</b><br>ID: %{customdata}<extra></extra>"
    ))
    
    fig.update_layout(
        template="plotly_white", 
        title=f"Visualizzazione di {n} CIG (Clicca un punto per i vincitori)",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=40, b=20),
        clickmode='event+select'
    )
    
    sa_name = SA_id.get(sa_id, {}).get('denominazione', f'SA {sa_id}')
    return fig, f"Stazione: {sa_name}", {'display': 'block'}, sa_id

@app.callback(
    [Output('awards-graph', 'figure'),
     Output('awards-info', 'children'),
     Output('level-3-row', 'style')],
    Input('cig-graph', 'clickData')
)
def update_awards(clickData):
    if not clickData or 'customdata' not in clickData['points'][0]:
        return no_update, "", {'display': 'none'}
    
    cig_id = clickData['points'][0]['customdata']
    if isinstance(cig_id, list): cig_id = cig_id[0]
    cig_id = int(cig_id)
    
    cig_code = CIG_id.get(cig_id, {}).get('cig', f"ID {cig_id}")
    
    # Get AGZ and AG
    mask_z = edge_cig_agz[0] == cig_id
    agz_ids = edge_cig_agz[1, mask_z].tolist()
    
    fig = go.Figure()
    nodes_x, nodes_y, texts, colors = [0], [2], [f"CIG: {cig_code}"], ['#1f77b4']
    
    for i, az in enumerate(agz_ids):
        ax = (i - (len(agz_ids)-1)/2) * 1.5
        nodes_x.append(ax); nodes_y.append(1); colors.append('#ff7f0e'); texts.append(f"Aggiudicazione {az}")
        fig.add_shape(type="line", x0=0, y0=2, x1=ax, y1=1, line=dict(color="#ccc", dash="dot"))
        
        mask_g = edge_agz_ag[0] == az
        for j, gid in enumerate(edge_agz_ag[1, mask_g]):
            gx = ax + (j - (len(edge_agz_ag[1, mask_g])-1)/2) * 0.4
            nodes_x.append(gx); nodes_y.append(0); colors.append('#2ca02c')
            w_name = AG_id.get(int(gid), {}).get('denominazione', 'N/A')
            texts.append(f"VINCITORE: {w_name}")
            fig.add_shape(type="line", x0=ax, y0=1, x1=gx, y1=0, line=dict(color="#99ff99"))

    fig.add_trace(go.Scatter(x=nodes_x, y=nodes_y, mode='markers+text', marker=dict(size=15, color=colors), text=texts, textposition="top center"))
    fig.update_layout(template="plotly_white", xaxis=dict(visible=False), yaxis=dict(visible=False), margin=dict(l=0, r=0, t=30, b=0))
    
    return fig, f"Dettagli per CIG {cig_code}", {'display': 'block'}

# --- NUOVA CALLBACK PER IL PANNELLO LATERALE ---
@app.callback(
    Output('detail-panel-content', 'children'),
    [Input('macro-graph', 'clickData'),
     Input('cig-graph', 'clickData'),
     Input('awards-graph', 'clickData')],
    prevent_initial_call=True
)
def display_node_details(macro_click, cig_click, awards_click):
    # Dobbiamo capire quale input è stato attivato
    from dash import callback_context
    if not callback_context.triggered:
        return html.P("Clicca su un nodo per vedere i dettagli.", className="text-muted")
    
    triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]
    click_data = callback_context.triggered[0]['value']
    
    if not click_data or 'points' not in click_data:
        return html.P("Seleziona un nodo.", className="text-muted")
    
    point = click_data['points'][0]
    
    # Inizializziamo le variabili per il recupero delle feature analitiche
    idx = None
    cols_to_use = []
    tensor_data = None

    # 1. Caso Stazione Appaltante (Macro Graph)
    if triggered_id == 'macro-graph':
        sa_id = point['customdata'][0] if isinstance(point['customdata'], list) else point['customdata']
        idx = int(sa_id)
        info = SA_id.get(idx, {})
        title = "Stazione Appaltante (SA)"
        color = "primary"
        items = [(k.replace('_', ' ').capitalize(), v) for k, v in info.items()]
        items.append(("Volume Appalti (CIG)", point['customdata'][1] if isinstance(point['customdata'], list) else "N/A"))
        cols_to_use, tensor_data = SA_cols, PA['SA'].x

    # 2. Caso CIG (CIG Graph)
    elif triggered_id == 'cig-graph':
        if point.get('curveNumber') == 0: 
            return html.Div([
                html.H6("Nodo Centrale (SA)", className="text-danger"),
                html.P("Torna alla mappa macro per i dettagli di questa SA.")
            ])
        
        cig_id = point['customdata']
        if isinstance(cig_id, list): cig_id = cig_id[0]
        idx = int(cig_id)
        info = CIG_id.get(idx, {})
        title = "Dettaglio Appalto (CIG)"
        color = "info"
        items = [(k.replace('_', ' ').capitalize(), v) for k, v in info.items()]
        cols_to_use, tensor_data = CIG_cols, PA['CIG'].x

    # 3. Caso Vincitore o CIG Centrale (Awards Graph)
    elif triggered_id == 'awards-graph':
        text = point.get('text', '')
        if "CIG:" in text:
            cig_code = text.split("CIG: ")[-1]
            # Troviamo l'ID numerico tramite il dizionario inverso o iterazione
            idx = next((int(k) for k, v in CIG_id.items() if v.get('cig') == cig_code), None)
            info = CIG_id.get(idx, {}) if idx is not None else {}
            title = "Dettaglio Appalto (CIG)"
            color = "info"
            items = [(k.replace('_', ' ').capitalize(), v) for k, v in info.items()]
            cols_to_use, tensor_data = CIG_cols, PA['CIG'].x
        elif "Aggiudicazione" in text:
             az_id = text.split("Aggiudicazione ")[-1]
             idx = int(az_id)
             info = AGZ_id.get(idx, {})
             title = "Dettaglio Aggiudicazione (AGZ)"
             color = "warning"
             items = [(k.replace('_', ' ').capitalize(), v) for k, v in info.items()]
             cols_to_use, tensor_data = AGZ_cols, PA['AGZ'].x
        elif "VINCITORE:" in text:
            nom_ag = text.split("VINCITORE: ")[-1]
            idx = next((int(k) for k, v in AG_id.items() if v.get('denominazione') == nom_ag), None)
            info = AG_id.get(idx, {}) if idx is not None else {}
            title = "Aggiudicatario (AG)"
            color = "success"
            items = [(k.replace('_', ' ').capitalize(), v) for k, v in info.items()]
            cols_to_use, tensor_data = AG_cols, PA['AG'].x
        else:
            return html.P("Dettagli non disponibili.")

    # Generazione HTML common
    rows = []
    
    # 1. Aggiungiamo i dati identificativi (JSON)
    for label, val in items:
        rows.append(html.Div([
            html.B(f"{label}: ", className="text-secondary small"),
            html.Span(str(val), className="d-block mb-2 font-monospace", style={'wordBreak': 'break-all'})
        ]))

    # 2. Aggiungiamo le feature numeriche (Tensor)
    try:
        if idx is not None and tensor_data is not None and idx < len(tensor_data):
            rows.append(html.Hr(className="my-3"))
            rows.append(html.H6(f"Feature Analitiche (Totale: {len(cols_to_use)})", className="text-muted small uppercase mb-3"))
            
            node_features = tensor_data[idx].numpy()
            
            # Mapping prefissi -> Titoli leggibili
            prefix_map = {
                'reg_': 'REGIONE',
                'tipo_': 'TIPO SOGGETTO',
                'ruolo_': 'RUOLO',
                'ngc_': 'NATURA GIURIDICA',
                'sr_': 'SEZIONE REG.',
                'FD_': 'FUNZ. DELEGATA',
                'IC_': 'IPOTESI COLL.',
                'opc_': 'OGGETTO PREV.'
            }

            active_badges = []
            inactive_features = []

            for i, col_name in enumerate(cols_to_use):
                if i >= len(node_features): break
                val = node_features[i]
                
                # Identificazione Dummy
                is_dummy = False
                for prefix, group_name in prefix_map.items():
                    if col_name.startswith(prefix):
                        is_dummy = True
                        label = col_name[len(prefix):].replace('_', ' ').strip().upper()
                        if label.startswith('_'): label = label[1:]
                        
                        if val > 0.5:
                            # Badge ATTIVO con categoria inclusa
                            active_badges.append(dbc.Badge(f"{group_name}: {label}", color="primary", className="me-1 mb-1 shadow-sm"))
                        else:
                            # Se a zero, mostriamo piccolo con etichetta
                            inactive_features.append(html.Div([
                                html.Small(f"{group_name}({label}): 0", className="text-muted", style={'fontSize': '0.65rem', 'marginRight': '8px'})
                            ], style={'display': 'inline-block'}))
                        break
                
                if not is_dummy:
                    # Feature numerica continua
                    is_active = abs(val) > 0.0001
                    clean_name = col_name.replace('_', ' ').capitalize()
                    inactive_features.append(html.Div([
                        html.Small(clean_name, 
                                 className="text-muted d-block" if is_active else "text-muted d-inline", 
                                 style={'fontSize': '0.7rem', 'marginRight': '5px'} if not is_active else {}),
                        html.B(f"{val:.4f}", className="d-block mb-1" if is_active else "me-3")
                    ], className="mb-2" if is_active else "d-inline-block"))

            # Mostriamo prima i badge attivi
            if active_badges:
                rows.append(html.Div(active_badges, className="mb-3 d-flex flex-wrap"))
            
            # Poi tutto il resto (continue e dummy a zero)
            rows.append(html.Div(inactive_features, className="mt-2 border-top pt-2", style={'opacity': '0.8'}))
    except Exception as e:
        print(f"Errore nel recupero feature: {idx}, {triggered_id}, {e}")
        rows.append(html.P(f"Errore tecnico: {str(e)}", className="text-danger small"))

    return html.Div([
        html.H5(title, className=f"text-{color} mb-3 border-bottom pb-2"),
        *rows
    ])

if __name__ == '__main__':
    # Disabilitiamo il reloader per evitare doppi caricamenti in memoria del grafo
    app.run(debug=False, port=8050)
