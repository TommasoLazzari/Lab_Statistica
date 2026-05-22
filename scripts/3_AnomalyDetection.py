import os
import gc
import copy
import math
import random
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from pathlib import Path

# --- DEFINIZIONE PERCORSI RELATIVI ---
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# Crea le cartelle se non esistono
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import softmax
from torch_geometric.loader import LinkNeighborLoader, NeighborLoader

#seed per riproducibilità
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

#GPU se disponibili, CPU altrimenti
def get_device():
    return torch.device("cpu")

#separa le relazioni "canoniche", reali (SA -> CIG; CIG -> AGZ; AGZ -> AG; CIG -> CIG); da quelle inverse, inserite nel grafo
#esclusivamente ai fini di migliorare le rappresentazioni latenti risultanti in output all'encoder
def canonical_relations_for_decoder(edge_types):
    preferred = [
        ("SA", "publishes", "CIG"),
        ("CIG", "has_award", "AGZ"),
        ("AGZ", "awarded_to", "AG"),
        ("CIG", "linked_to", "CIG"),
    ]
    return [rel for rel in preferred if rel in edge_types]

#trasforma le relazioni in stringa 
def relation_key(rel: Tuple[str, str, str]) -> str:
    return "__".join(rel)

#serve a trovare il nome della relazione inversa di quella data in input
def get_reverse_relation(rel, edge_types):
    #rel = relazione del grafo
    #edge_type = lista di tutte le relazioni presenti nel grafo    
    src, rel_name, dst = rel

    if rel_name == "linked_to" and rel in edge_types:
        return rel

    candidate = (dst, f"rev_{rel_name}", src)
    if candidate in edge_types:
        return candidate

    return None

#standardizzazione dei tensori
def zscore_safe(x: torch.Tensor):
    mean = x.mean()
    std = x.std(unbiased=False)
    if std.item() < 1e-12:
        return torch.zeros_like(x)
    return (x - mean) / std


########################################################################
## Encoder  ############################################################
########################################################################

#per una singola relazione del grafo, costruisce messaggi dai nodi sorgente, 
#calcola per ogni arco un coefficiente di attenzione che dipende sia dal source sia dal target, 
#normalizza tali coefficienti tra i vicini dello stesso target, 
#aggrega i messaggi pesandoli con questi coefficienti e 
#infine aggiunge una componente self del nodo target.

#Il risultato è che, dentro una stessa relazione, 
#il nodo target può imparare che alcuni vicini sono più importanti di altri.

class RelationAttentionConv(nn.Module):
    #classe PyTorch che rappresenta un layer che prende in input le feature dei nodi sorgenti e target di una 
    #certa relazione e produce nuove rappresentazioni per i nodi target

    def __init__(self, in_channels, out_channels):
        super().__init__()

        if isinstance(in_channels, (tuple, list)):
        #se i nodi uniti dalla relazione sono diversi
            in_src, in_dst = in_channels
            #in_src = dimensione feature del nodo sorgente
            #in_dst = dimensione features del nodo target
        else:
        #altrimenti le due dimensioni coincidono
            in_src = in_channels
            in_dst = in_channels

        #dimensione dell'embedding in uscita
        self.out_channels = out_channels

        #trasformazione lineare: prende le features del nodo sorgente h_j e e proietta nello spazio latente del layer: W_r h_j per la relazione r
        self.lin_src_msg = Linear(in_src, out_channels, bias=False)
        #trasformazione lineare: prende le features del nodo target e le proietta nello stesso spazio latente: K_r h_i per la relazione r
        self.lin_dst_att = Linear(in_dst, out_channels, bias=False)
        #trasformazione lineare: W_self h_i
        self.lin_self = Linear(in_dst, out_channels, bias=True)

        #vettore di parametri allenabili usato per calcolare il punteggio di attenzione sugli archi
        self.att = nn.Parameter(torch.empty(1, 2 * out_channels)) #2*channels perchè l'attenzione usa una concatenazione tra informazione del nodo target e source
        self.reset_parameters()

    def reset_parameters(self):
    #nel metodo reset_parameters
        #vengono inizializzati i pesi delle tre trasformazioni lineari 
        #e il vettore dell'attenzione con Xavier uniform
        self.lin_src_msg.reset_parameters()
        self.lin_dst_att.reset_parameters()
        self.lin_self.reset_parameters()
        nn.init.xavier_uniform_(self.att)

    def forward(self, x, edge_index):
        x_src, x_dst = x #features dei noi sorgenti e target
        src_idx, dst_idx = edge_index #indici dei nodi sorgenti e target
        #quindi se l'arco k-esimo unisce il nodo i con il nodo j, avremo
        #src_idx[k] = i e dst_idx[k] = j

        #trasformiamo le features dei due nodi nello spazio dei messaggi
        msg_src = self.lin_src_msg(x_src)
        dst_ctx = self.lin_dst_att(x_dst)

        #costruiamo il vettore su cui verrà calcolata l'attenzione, concatenando le features trasfromate dei nodi target e sorgente coinvolti
        #quindi per ogni arco (j -> i), otteniamo [K_r h_i || W_r h_j ]
        att_input = torch.cat([dst_ctx[dst_idx], msg_src[src_idx]], dim=-1)
        #otteniamo degli scalari di attenzione e_ij, applicando una leakyRelu per la non linearità
        #sul prodotto scalare di att_input e il parametro att
        e_ij = F.leaky_relu((att_input * self.att).sum(dim=-1), negative_slope=0.2)

        #softmax per normalizzare l'attenzione saparatamente per ciascun nod target
        #quindi, per un nodo target i, tutti i vicini j \in N_{r}(i) collegati con la relazione r 
        #ottengono coefficienti che sommano a 1
        #a_{ij}^{(r)} = exp(e_{ij})   / sum_{k \in N_{r}(i)} exp(e_{ik})
        alpha_ij = softmax(e_ij, dst_idx, num_nodes=x_dst.size(0))

        #tensore di output con numero di righe = numero di nodi target e numero di colonne = out_channels
        out = msg_src.new_zeros((x_dst.size(0), self.out_channels))
        #per ogni arco prendiamo le features del nodo source, j, trasformato, W_r h_j =  msg_src[src_idx], 
        #lo moltiplichiamo per il coefficiente di attenzione alpha_ij
        #e lo sommiamo nella riga corrispondente al nodo target
        out.index_add_(0, dst_idx, msg_src[src_idx] * alpha_ij.unsqueeze(-1))

        #dopo aver aggregato i vicini, aggiungiamo il contributo del nodo stesso
        #quindi per un nodo target i, per una relazione r, avremo una rappresentazione:
        #h_i = sum_{j \in N_{r}(i)} a_{ij}^{(r)} W_{r}h_{j} + W_{self} h_{i}
        out = out + self.lin_self(x_dst)
        return out


class RelationAwareHeteroLayer(nn.Module):
    #è un layer PyTorch che aggiorna contemporaneamente tutti i tipi di nodo del grafo eterogeneo

    def __init__(self, metadata, in_channels, out_channels):
        super().__init__()

        node_types, edge_types = metadata
        self.node_types = list(node_types)
        self.edge_types = list(edge_types)
        self.out_channels = out_channels

        #per ogni relazione crea un modulo RelationAttentionConv
        #la chiave del ModuleDict è la stringa ottenuta con relation_key
        #dunque per ogni relazione r, esisterà una convoluzione dedicata
        self.convs = nn.ModuleDict({
            relation_key(edge_type): RelationAttentionConv(in_channels, out_channels)
            for edge_type in self.edge_types
        })

        #dizionario delle relazioni entranti in ciascun tipo di nodo
        self.incoming_by_dst = {node_type: [] for node_type in self.node_types}
        for edge_type in self.edge_types:
            self.incoming_by_dst[edge_type[2]].append(edge_type)

        self.relation_logits = nn.ParameterDict()
        self.relation_index_by_dst = {}
        for dst_type, rels in self.incoming_by_dst.items():
            self.relation_index_by_dst[dst_type] = {
                rel: idx for idx, rel in enumerate(rels)
            }
            #inizializziamo un vettore di parametri 0 per ogni tipo di nodo della lunghezza del numero di relazioni entranti in quel tipo di nodo
            #che durante ik training verranno modificati
            if len(rels) > 0:
                self.relation_logits[dst_type] = nn.Parameter(torch.zeros(len(rels)))

    def forward(self, x_dict, edge_index_dict):
        #per_relation_outputs raccoglie, per ogni tipo di nodo target
        #gli output separati delle varie relazioni
        per_relation_outputs = {node_type: {} for node_type in self.node_types}

        for edge_type in self.edge_types:
        #per ogni relazione edge_type
            #controlliamo se esiste nel batch
            if edge_type not in edge_index_dict:
                continue

            edge_index = edge_index_dict[edge_type]
            #se ha archi nel batch
            if edge_index.numel() == 0:
                continue

            #estra i nodi sorgenti e destinazione degli archi edge_type
            src_type, _, dst_type = edge_type
            # e applica la convoluzione RelationAttentionConv corrispondente a edge_type
            try:
                out = self.convs[relation_key(edge_type)](
                    (x_dict[src_type], x_dict[dst_type]),
                    edge_index,
                )
                per_relation_outputs[dst_type][edge_type] = out
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Conv fallita per {edge_type}: {e}", exc_info=True)
                raise

        out_dict = {}
        for dst_type in self.node_types:
            #per ogni tipo di nodo dst_type
            #otteniamo tutti i corrispettivi output di RelationAttentionConv disponibili            
            available = per_relation_outputs[dst_type]

            if len(available) == 0:
                #se non ci sono contributi, restituiamo una matrice di zeri
                num_nodes = x_dict[dst_type].size(0)
                out_dict[dst_type] = x_dict[dst_type].new_zeros((num_nodes, self.out_channels))
                continue

            rels = list(available.keys())
            #impiliamo gli output ottenuti dalla SAGEconv corrispondenti alle varie relazioni entranti in dst_type
            stacked = torch.stack([available[rel] for rel in rels], dim=0)

            #se c'è solo una relazione, non serve calcolare i pesi; l'output coincide con quello dell'unico contributo
            if len(rels) == 1:
                out_dict[dst_type] = stacked[0]
                continue

            #se invece ci sono più relazioni:
            #per ciascuna relazione disponibile nel batch, recuperiamo l'indice corrispondente nel vettore dei logits            
            rel_indices = [self.relation_index_by_dst[dst_type][rel] for rel in rels]
            #selezioniamo i logits relativi alle relazioni effettivamente disponibili in quel batch
            logits = self.relation_logits[dst_type][rel_indices]
            #trasformiamo i logits in pesi normalizzati che sommano a 1
            weights = torch.softmax(logits, dim=0).view(-1, 1, 1)
            #facciamo la combinazione pesate dei contributi ottenuti dalle diverse relazioni 
            out_dict[dst_type] = (weights * stacked).sum(dim=0)

        #restituisce un dizionario con le nuove embedding, una matrice per ciascun tipo di nodo
        return out_dict


class HeteroEncoder(nn.Module):

    def __init__(self, metadata, hidden_dim: int = 32, latent_dim: int = 16, dropout: float = 0.25):
        #metadata = data.metadata() è una coppia (node_types, edge_types)
        #hidden_dim = dimensione dello spazio nascosto delle rappresentazioni intermedie, ottenute dopo il primo layer
        #latent_dim = dimensione finale delle embedding latenti; l'encoder restituisce per ogni nodo un vettore finale di lunghezza latent_dim          
        super().__init__()
        
        self.dropout = dropout

        #primo layer RelationAttentionConv
        self.conv1 = RelationAwareHeteroLayer(
            metadata=metadata,
            in_channels=(-1, -1),
            out_channels=hidden_dim,
        )

        #secondo layer RelationAttentionConv
        self.conv2 = RelationAwareHeteroLayer(
            metadata=metadata,
            in_channels=(hidden_dim, hidden_dim),
            out_channels=latent_dim,
        )

    def forward(self, x_dict, edge_index_dict):
        #x_dict è il dizionario delle feature dei nodi
        
        #{
        #    'SA': tensor [num_SA_nodes, 129],
        #    'CIG': tensor [num_CIG_nodes, 63],
        #    'AGZ': tensor [num_AGZ_nodes, 13],
        #    'AG': tensor [num_AG_nodes, 11]
        #}
        
        #edge_index_dict è un dizionario degli archi per tipo di relazione
        
        #{
        #('SA', 'publishes', 'CIG'): edge_index,
        #('CIG', 'rev_publishes', 'SA'): edge_index,
        #...
        #}  
        
        h_dict = self.conv1(x_dict, edge_index_dict)
        #prima convoluzione eterogenea. h_dict è un dizionario con la stessa struttura di x_dict, 
        #ma ogni tipo di nodo ora ha embedding di dimensione hidden_dim
        
        #{
        #'SA': tensor [n_SA_batch, hidden_dim],
        #'CIG': tensor [n_CIG_batch, hidden_dim],
        #'AGZ': tensor [n_AGZ_batch, hidden_dim],
        #'AG': tensor [n_AG_batch, hidden_dim]
        #}    
        
        h_dict = {k: F.dropout(F.relu(v), p=self.dropout, training=self.training) for k, v in h_dict.items()}
        #RELU ad ogni tensore del dizionario h_dict
        z_dict = self.conv2(h_dict, edge_index_dict)
        #seconda convoluzione eterogenea
        
        #{
        #'SA': tensor [n_SA_batch, latent_dim],
        #'CIG': tensor [n_CIG_batch, latent_dim],
        #'AGZ': tensor [n_AGZ_batch, latent_dim],
        #'AG': tensor [n_AG_batch, latent_dim]
        #}
        
        return z_dict #nuove rappresentazioni: embedding




########################################################################
## Decoder  attributivo ################################################
########################################################################
#AttributeDecoder è un insieme di MLP specifici per tipo di nodo 
#che ricostruiscono le feature originali a partire dalle embedding latenti
class AttributeDecoder(nn.Module):
    
    def __init__(self, in_dims, latent_dim: int = 16, hidden_dim: int = 32, dropout: float = 0.25):
        #in_dims = dizionario che dice le dimensioni delle feature originali per ogni tipo di nodo
        #latent_dim = dimensione delle embedding in input
        #hidden_dim = dimensione della reppresentazione intermedia del decoder
        super().__init__()

        self.decoders = nn.ModuleDict() #dizionario Pytorch di moduli, che 
        #- registra automaticamente i parametri per l'optimizer
        #- è compatibile con .to(device)
        #- comunica a Pytorch che i parametri sono allenabili
        for node_type, out_dim in in_dims.items(): #per ogni tipo di nodo costruiamo un decoder separato
            self.decoders[node_type] = nn.Sequential( #che è un piccolo MLP
                nn.Linear(latent_dim, hidden_dim), #latent_dim -> Linear -> hidden_dim
                nn.ReLU(),                         #Relu
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, out_dim),    #hidden_dim -> Linear -> dimensioni originali
            )

    def forward(self, z_dict):
        #z_dict è l'output dell'encoder che contiene le embedding per ogni nodo nel batch
        x_hat_dict = {}
        for node_type, z in z_dict.items(): #per ogni tipo di nodo, prendine gli embeddings
            x_hat_dict[node_type] = self.decoders[node_type](z) #applicaci il corrispettivo decoder
        #l'output finale corrisponde alla ricostruzione degli attributi dei nodi dati come input, in un dizionario dalla forma:
        
        #{
        #'SA': tensor [n_SA_batch, 129],
        #'CIG': tensor [n_CIG_batch, 63],
        #'AGZ': tensor [n_AGZ_batch, 13],
        #'AG': tensor [n_AG_batch, 11]
        #}
        
        return x_hat_dict

########################################################################
## Decoder  strutturale ################################################
########################################################################
#RelationDecoder assegna uno score a ogni edge usando una funzione bilineare relation-specific tra le embedding dei nodi
class RelationDecoder(nn.Module):
    def __init__(self, relations: List[Tuple[str, str, str]], latent_dim: int = 16):
        #relations = lista delle relazioni canoniche (quelle reali; nel grafo sono state inserite anche le relazioni inverse)
        #latent_dim = dimensione delle embedding
        super().__init__()

        self.relations = relations
        self.rel_to_key = {rel: relation_key(rel) for rel in relations} #trasforma ogni relazione in una stringa e serve perchè ParameterDict usa stringhe come chiavi, non tuple

        self.weights = nn.ParameterDict() 
        #parametri allenabili
        #{
        #"SA__publishes__CIG": W1,
        #"CIG__has_award__AGZ": W2,
        #...
        #}
        for rel in relations: 
            #per ogni relazione r, 
            #creiamo una matrice W_r \in R^{latent_dim \times latent_dim}
            key = self.rel_to_key[rel]
            W = torch.empty(latent_dim, latent_dim)
            nn.init.xavier_uniform_(W)
            self.weights[key] = nn.Parameter(W)
            
  

    def score_edges(self, z_src, z_dst, edge_index, rel): #funzione di scoring
        #z_src = embedding dei nodi sorgente
        #z_dst = embedding dei nodi di destinazione
        #edge_index = prima riga con gli indici dei nodi sorgente, seconda riga con gli indici dei nodi destinazione
        #rel = una relazione specifica, ad esempio ("CIG", "has_award", "AGZ")
        key = self.rel_to_key[rel]
        W = self.weights[key] #recupero della matrice della relazione rel

        src_idx = edge_index[0] #indici dei nodi interessati dagli archi della relazione rel
        dst_idx = edge_index[1]

        z_u = z_src[src_idx] #selezione degli embedding corrispondenti
        z_v = z_dst[dst_idx]

        z_uW = z_u @ W 
        score = (z_uW * z_v).sum(dim=-1) #score(u,v|rel) = z_u^{T} W_rel z_v --> quanto è compatabile il nodo u con v sotto la relazione rel?
        return score
    

########################################################################
## Modello #############################################################
########################################################################
#BaselineMiniBatchModel è un modulo Pytorch che assemble l'encoder eterogeneo,
#il decoder attributivo e il decoder strutturale in un unico modello. Restituisce nel forward gli embedding 
#latenti risultanti dall'encoder e la ricostruzione risultanti dal decoder attributivo
#mentre il decoder strutturale viene usato separatamente per dare un punteggio agli archi.
class BaselineMiniBatchModel(nn.Module):
    
    def __init__(
        self,
        metadata,
        in_dims: Dict[str, int],
        decoder_relations: List[Tuple[str, str, str]],
        hidden_dim: int = 32,
        latent_dim: int = 16, 
        dropout: float = 0.3
    ):
        super().__init__()

        #encoder -> z_dict (embeddings)
        self.encoder = HeteroEncoder(
            metadata=metadata,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout
        )
        #decoder attributivo -> x_hat_dict (ricostruzione delle features originali)
        self.attr_decoder = AttributeDecoder(
            in_dims=in_dims,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        #decoder relazione -> score
        self.rel_decoder = RelationDecoder(
            relations=decoder_relations,
            latent_dim=latent_dim,
        )

    def forward(self, x_dict, edge_index_dict):
        z_dict = self.encoder(x_dict, edge_index_dict)
        x_hat_dict = self.attr_decoder(z_dict)
        return z_dict, x_hat_dict #restituisce sia gli embeddings prodotti dall'encoder sia le feature ricostruite dal decoder attributivo
    
#novità rispetto alla baseline
# -nessuna
    

########################################################################
## Loss attributiva ####################################################
########################################################################

# Novità: media pesata per tipo di nodo, con peso proporzionale al numero di nodi presenti nel batch.
def attribute_reconstruction_loss(x_dict, x_hat_dict):
    #x_dict = dizionario contenente le feature originali dei nodi separate per tipo di nodo
    #x_hat_dict = dizionario contenente le feature ricostruite dal decoder separate per tipo di nodo
    total_weighted_loss = 0.0
    total_nodes = 0
    details = {}

    for node_type in x_dict.keys():
    #per ogni tipo di nodo
        if x_dict[node_type].numel() == 0:
        #se c'è almeno un elemento di quel tipo di nodo
            continue

        #MSE tra feature ricostruite e originali dei nodi node_type
        rec = F.mse_loss(x_hat_dict[node_type], x_dict[node_type], reduction="mean")
        #otteniamo il numero di nodi node_type
        num_nodes = x_dict[node_type].size(0)

        #aggiungiamo alla loss totale la loss attributiva derivante da node_types pesata per il numero di nodi
        total_weighted_loss = total_weighted_loss + rec * num_nodes
        total_nodes += num_nodes
        #salvataggio delle informazioni diagnostiche
        details[node_type] = {
            "loss": rec.detach().item(),
            "num_nodes": num_nodes,
        }

    #caso anomalo, nel caso in cui non venga trovato alcun caso valido
    if total_nodes == 0:
        device = next(iter(x_dict.values())).device
        return torch.tensor(0.0, device=device), details, 0

    #altrimenti: loss attributiva totale / numero di nodi
    loss = total_weighted_loss / total_nodes
    return loss, details, total_nodes


########################################################################
## Loss strutturale ####################################################
########################################################################

#Questa funzione prende un batch di archi di una relazione, 
#calcola quanto bene il decoder distingue archi veri e archi negativi, 
#e restituisce una BCE strutturale in cui positivi e negativi possono contribuire 
#con pesi diversi invece di essere lasciati al puro sbilanciamento del batch
def structural_loss_from_batch(
    z_dict,
    batch,
    rel,
    rel_decoder,
    pos_weight: float = 0.5,
    neg_weight: float = 0.5,
):
    #z_dict = embedding latenti dei nodi, separati per tipo di nodo
    #batch = mini-batch prodotto dal loader per una certa relazione rel
    #rel = una relazione canonica
    #rel_decoder = decoder strutturale per rel
    #pos_weights = peso da assegnare alla loss sugli archi positivi
    #neg_weights = peso da assegnare alla loss sugli archi negativi
    
    #estrazione di nodi sorgenti e di destinazione per le relazioni rel
    src_type, _, dst_type = rel

    #indici dei nodi coinvolti nelle relazioni rel nel batch
    edge_label_index = batch[rel].edge_label_index
    #etichette corrispondenti: 1 arco positivo, 0 altrimenti
    edge_label = batch[rel].edge_label.float()

    #score logit per ogni arco prodotti dal decoder strutturale
    scores = rel_decoder.score_edges(
        z_src=z_dict[src_type],
        z_dst=z_dict[dst_type],
        edge_index=edge_label_index,
        rel=rel,
    )

    #vettore con binary cross entropy per ciascun arco (reduction = None)
    bce = F.binary_cross_entropy_with_logits(scores, edge_label, reduction="none")
    
    #maschere booleane per selezionare archi positivi e negativi
    pos_mask = edge_label == 1
    neg_mask = edge_label == 0
    #loss media su gli archi positivi
    pos_loss = bce[pos_mask].mean() if pos_mask.any() else None
    #loss media su gli archi negativi
    neg_loss = bce[neg_mask].mean() if neg_mask.any() else None

    weighted_sum = 0.0
    weight_sum = 0.0

    if pos_loss is not None:
    #se esiste pos_loss, la si aggiunge a weighted_sum con peso pos_weight
        weighted_sum = weighted_sum + pos_weight * pos_loss
        weight_sum += pos_weight
    if neg_loss is not None:
    #se esiste neg_loss, la si aggiunge a weighted_sum con peso neg_weight
        weighted_sum = weighted_sum + neg_weight * neg_loss
        weight_sum += neg_weight
        
   
    if weight_sum == 0.0:
        loss = bce.mean()
    else:
        #loss = ( w_pos * L_pos + w_neg * L_neg) / (w_pos + w_neg)
        loss = weighted_sum / weight_sum

    details = {
        "num_edges": int(edge_label.numel()),
        "num_pos": int(pos_mask.sum().item()),
        "num_neg": int(neg_mask.sum().item()),
        "pos_loss": None if pos_loss is None else float(pos_loss.detach().item()),
        "neg_loss": None if neg_loss is None else float(neg_loss.detach().item()),
    }
    #restituisce loss strutturale del batch, statistiche dettagliate sul batch, numero totale di archi nel batch
    return loss, details, int(edge_label.numel())


########################################################################
## Train/Val split #####################################################
########################################################################
def split_edges_train_val(
    data,
    decoder_relations,
    val_ratio: float = 0.10,
    seed: int = 2,
):
    #data = grafo
    #decoder_relations = lista delle relazioni su cui bisogna fare reconstruction
    #val_ratio = percentuale di archi da mettere in validation
    #seed = per rendere lo split riproducibile
    generator = torch.Generator()
    generator.manual_seed(seed)

    edge_splits = {}

    for rel in decoder_relations:
    #per ogni relazione rel nel decoder_relations
        #estra gli indici dei nodi sorgenti e di destinazione coinvolti in rel
        edge_index = data[rel].edge_index
        #numero totale di archi rel
        num_edges = edge_index.size(1)

        if num_edges <= 1 or val_ratio <= 0.0:
        #se ci sono troppi pochi archi per fare lo split o se val_ratio è zero
            #inseriamo tutto nel train set
            train_edge_index = edge_index
            val_edge_index = edge_index[:, :0]
        else:
            val_size = max(1, int(num_edges * val_ratio))
            val_size = min(val_size, num_edges - 1)
            #permutazione casuale sugli archi
            perm = torch.randperm(num_edges, generator=generator)
            val_idx = perm[:val_size]
            train_idx = perm[val_size:]
            #selezioniamo le colonne di edge_index corrispondenti agli archi selezionai per il val e il train set
            train_edge_index = edge_index[:, train_idx]
            val_edge_index = edge_index[:, val_idx]

        #salvataggio nella struttura finale
        edge_splits[rel] = {
            "train": train_edge_index,
            "val": val_edge_index,
            "num_total": num_edges,
            "num_train": train_edge_index.size(1),
            "num_val": val_edge_index.size(1),
        }

    #restituisce un dizionario completo con tutti gli split
    return edge_splits


#Questa funzione costruisce il grafo usato nel training, 
#mantenendo solo gli archi di train nelle relazioni del decoder 
#e aggiornando anche le reverse relations corrispondenti
def build_train_graph_from_splits(data, edge_splits):
    #data = grafo
    #edge_splits = dizionario prodotto da split_edges_train_val
    
    #copia del grafo
    train_data = copy.deepcopy(data)
    #relazioni presenti nel grafo
    edge_types = set(train_data.edge_types)

    for rel, split in edge_splits.items():
    #per ogni relazione rel di cui abbiamo costruito uno split
        #estraiamo gli archi per il train
        train_edge_index = split["train"]
        train_data[rel].edge_index = train_edge_index.clone()

        #otteniamo l'eventuale relazione contraria
        rev_rel = get_reverse_relation(rel, edge_types)
        if rev_rel is not None and rev_rel != rel:
            train_data[rev_rel].edge_index = torch.stack(
                [train_edge_index[1], train_edge_index[0]],
                dim=0,
            )

    return train_data



########################################################################
## Loaders #############################################################
########################################################################

#Questa funzione costruisce, per ogni relazione, 
#un loader che genera mini-batch di archi positivi e negativi con neighborhood sampling, 
#pronti per essere usati nel training della structural loss.
def build_link_loaders(
    data,
    edge_index_by_relation,
    num_neighbors: List[int],
    edge_batch_size: int,
    neg_sampling_ratio: float,
    shuffle: bool,
    num_workers: int = 0,
):
    #data = grafo (train_data oppure val_data)
    #edge_index_by_relation = dizionario del tipo
    #{
    #rel1: edge_index_train,
    #rel2: edge_index_train,
    #...
    #}
    #num_neighbors = numero di vicini da campionare per ogni hop
    #edge_batch_size = numero di archi per batch
    #neg_sampling_ratio = numero di archi negativi ogni arco positivo
    #shuffle = randomizzazione degli archi tra epoche
    loaders = {}

    for rel, edge_index in edge_index_by_relation.items():
    #per ogni relazione rel per cui vogliamo costruire un loader
        if edge_index.size(1) == 0:
            continue

        #assegniamo etichetta 1 a tutti gli archi
        edge_label = torch.ones(edge_index.size(1), dtype=torch.float)

        #creiamo un loader che
        #- prenda un insieme di archi positivi
        #- li divida in mini-batch
        #- per ogni batch
        #     - campioni i vicini dei nodi coinvolti
        #     - genera archi negativi
        #     - costruisce un sotto-grafo
        loader = LinkNeighborLoader(
            data=data,
            num_neighbors=num_neighbors,
            edge_label_index=(rel, edge_index),
            edge_label=edge_label,
            neg_sampling_ratio=neg_sampling_ratio,
            batch_size=edge_batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
        loaders[rel] = loader

    #restituisce un dizionario di loaders:
    #{
    #("SA", "publishes", "CIG"): loader1,
    #("CIG", "rev_publishes", "SA"): loader2,
    #...
    #}
    return loaders


#build_node_inference_loader costruisce un NeighborLoader che scorre tutti i nodi di un certo tipo in mini-batch, 
#campionando il loro intorno nel grafo per permettere il calcolo scalabile delle embedding e 
#della ricostruzione delle feature durante la fase di inferenza.
def build_node_inference_loader(
    data,
    node_type: str,
    num_neighbors: List[int],
    batch_size: int,
    num_workers: int = 0,
):
    #data = grafo eterogeneo completo
    #node_type = tipo di nodo su cui si vuole fare inferenza
    #batch_size = numero di nodi per batch
    #num_neighbors = lista di interi che controlla il neighbor sampling multi-hop
    
    input_nodes = (node_type, torch.arange(data[node_type].num_nodes)) #tutti i nodi del tipo node_type

    #creazione del NieghborLoader che prende un batch di nodi target,
    #compiona i loro vicini (multi-hop),
    #costruisce un sottografo locale,
    #restituisce quel sottografo
    loader = NeighborLoader(
        data=data,
        num_neighbors=num_neighbors,
        input_nodes=input_nodes,
        batch_size=batch_size,
        shuffle=False, #False perchè vogliamo coprire tutto il dataset in maniera ordinata
        num_workers=num_workers,
    )
    #restituisce il loader 
    return loader


#build_edge_inference_loader costruisce un LinkNeighborLoader 
#che scorre tutti gli archi reali di una relazione 
#senza negative sampling, permettendo di calcolare in mini-batch gli score strutturali 
#del modello su tutto il grafo durante la fase di inferenza.
def build_edge_inference_loader(
    data,
    rel,
    num_neighbors: List[int],
    edge_batch_size: int,
    num_workers: int = 0,
):
    #data = grafo
    #rel = relazione canonica
    #num_neighbors = il solito
    #edge_batch_size = numero di archi target per batch
    
    edge_index = data[rel].edge_index #tutti gli indici dei nodi sorgenti e destinazione relativi agli archi rel
    edge_label = torch.ones(edge_index.size(1), dtype=torch.float) #vettore di 1 della lunghezza di edge_index

    #creazione del loader
    loader = LinkNeighborLoader(
        data=data,
        num_neighbors=num_neighbors,
        edge_label_index=(rel, edge_index),
        edge_label=edge_label,
        neg_sampling_ratio=0.0,
        batch_size=edge_batch_size,
        shuffle=False, #vogliamo scorrere il grafo in maniera deterministica
        num_workers=num_workers,
    )
    return loader



########################################################################
## Training ############################################################
########################################################################

#Questa funzione valuta il modello sui validation mini-batch, 
#calcolando loss attributiva, strutturale e totale, e le aggrega con medie pesate
@torch.no_grad() #non consideriamo i gradienti
def evaluate_model_minibatch(
    model,
    val_loaders,
    decoder_relations,
    alpha_attr: float,
    alpha_struct: float,
    pos_edge_weight: float,
    neg_edge_weight: float,
):
    #model = modello completo
    #val_loaders = dizionario dei loader di validation (uno per relazione)
    #decoder_relations = lista delle relazioni da valutare
    #alpha_attr = peso della componente attributiva nella loss totale
    #alpha_struct = peso della componente strutturale nella loss totale
    #pos_edge_weight = peso della parte positiva nella loss strutturale
    #neg_edge_weight = peso della parte negativa nella loss strutturale
    device = get_device()
    model.eval()

    total_loss_sum = 0.0
    attr_loss_sum = 0.0
    struct_loss_sum = 0.0

    total_loss_weight = 0.0
    attr_loss_weight = 0.0
    struct_loss_weight = 0.0

    for rel in decoder_relations:
    #per ogni relazione rel in decoder_relations
        if rel not in val_loaders:
            continue

        #seleziona il loader
        loader = val_loaders[rel]

        for batch in loader:
        #per ogi batch nel loader
            batch = batch.to(device)

            #ottieni embedding dei nodi e ricostruzione delle features dei nodi nel batch
            z_dict, x_hat_dict = model(batch.x_dict, batch.edge_index_dict)

            #loss attributiva del batch e numero di nodi nel batch
            loss_attr, _, attr_weight = attribute_reconstruction_loss(batch.x_dict, x_hat_dict)
            #loss strutturale del batch e numero di archi nel batch
            loss_struct, _, struct_weight = structural_loss_from_batch(
                z_dict=z_dict,
                batch=batch,
                rel=rel,
                rel_decoder=model.rel_decoder,
                pos_weight=pos_edge_weight,
                neg_weight=neg_edge_weight,
            )

            #loss totale del batch = alpha_attr * loss_attr + alpha_struct * loss_struct
            loss = alpha_attr * loss_attr + alpha_struct * loss_struct
            total_weight = max(1, struct_weight)

            #aggiungiamo alla loss totale la loss del batch pesata per numero di archi
            total_loss_sum += loss.item() * total_weight
            #aggiungiamo alla loss attributiva la loss del batch pesata per numero di archi
            attr_loss_sum += loss_attr.item() * max(1, attr_weight)
            #aggiungiamo alla loss strutturale la loss del batch pesata per numero di nodi
            struct_loss_sum += loss_struct.item() * max(1, struct_weight)

            total_loss_weight += total_weight
            attr_loss_weight += max(1, attr_weight)
            struct_loss_weight += max(1, struct_weight)

            #rimozione variabili inutili
            del batch, z_dict, x_hat_dict, loss, loss_attr, loss_struct

    # Fine valutazione: liberiamo la memoria accumulata
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    #calcolo e salvataggio delle metriche finali: loss totale, loss attributiva e loss strutturale
    metrics = {
        "total": total_loss_sum / max(1.0, total_loss_weight),
        "attr": attr_loss_sum / max(1.0, attr_loss_weight),
        "struct": struct_loss_sum / max(1.0, struct_loss_weight),
    }
    return metrics


#Questa funzione allena il modello mini-batch della v2 su un grafo eterogeneo con split train/validation degli archi, 
#randomizzazione delle relazioni, loss pesate, evaluation in validation ed early stopping, 
#e alla fine restituisce il miglior modello osservato durante il training.
def train_model_minibatch(
    data,
    hidden_dim: int = 64,
    latent_dim: int = 32,
    epochs: int = 50,
    lr: float = 2e-4,
    weight_decay: float = 5e-3,
    alpha_attr: float = 1.0,
    alpha_struct: float = 0.5,
    dropout: float = 0.3,
    num_neighbors: List[int] = [15, 10],
    edge_batch_size: int = 4096,
    neg_sampling_ratio: float = 1.0,
    num_workers: int = 0,
    val_ratio: float = 0.10,
    early_stopping_patience: int = 10,
    early_stopping_min_delta: float = 1e-4,
    pos_edge_weight: float = 0.5,
    neg_edge_weight: float = 0.5,
    seed: int = 2,
):
    if neg_sampling_ratio == 0.0:
        import warnings
        warnings.warn("neg_sampling_ratio=0.0: nessun arco negativo, structural loss inutile!")
    #hidden_dim = dimensione della rappresentazione intermedia dei nodi nell'encoder e nel decoder attributivo
    #latent_dim = dimensione delle embedding
    #epochs = numero massimo di epoche
    #lr = learning rate
    #weight decay = regolarizzazione L2 dell'optimizer
    #alpha_attr = peso della loss attributiva
    #alpha_struct = peso della loss strutturale
    #num_neighbors = numero di vicini campionati per ogni hop
    #edge_batch_size = numero di archi positivi per batch
    #neg_sampling_ratio = numero di archi negativi ogni positivo
    
    #val_ratio = quota di archi da mettere in validation
    #early_stopping_patience = numero di epoche senza miglioramente tollerate
    #early_stopping_min_delta = miglioramento minimo richiesto per considerare valida una nuova best validation
    
    device = get_device()
    metadata = data.metadata() #lista dei tipi di nodo, lista dei tipi di arco
    in_dims = {nt: data[nt].x.size(1) for nt in data.node_types}
    decoder_relations = canonical_relations_for_decoder(data.edge_types) #relazioni canoniche

    #split train/validation sugli archi
    edge_splits = split_edges_train_val(
        data=data,
        decoder_relations=decoder_relations,
        val_ratio=val_ratio,
        seed=seed,
    )

    #costruzione del grafo per il training: parte dal grafo completo, 
    #sostituisce gli archi completi con i soli archi di train e 
    #aggiorna le rispettive relazioni inverse
    train_graph = build_train_graph_from_splits(data, edge_splits)

    #costruzione del modello completo:
    # -HeteroEncoder
    # -AttributeDecoder
    # -RelationDecoder    
    model = BaselineMiniBatchModel(
        metadata=metadata,
        in_dims=in_dims,
        decoder_relations=decoder_relations,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        dropout=dropout
    ).to(device)

    #costruzione dell'optimizer che aggiornerà i pesi dell'intero modello dopo ogni mini-batch
    optimizer = torch.optim.Adam(
        model.parameters(), #include pesi dell'encoder, del decoder attributivo e del decoder strutturale
        lr=lr,
        weight_decay=weight_decay,
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=3,
        factor=0.5,
    )    

    #dividiamo edge_splits in due dizionari più comodi per costruire i loader
    train_edge_index_by_relation = {
        rel: edge_splits[rel]["train"] for rel in decoder_relations
    }
    val_edge_index_by_relation = {
        rel: edge_splits[rel]["val"] for rel in decoder_relations
    }

    #costruiamo un dizionario d loader per il train, uno per relazione
    #ogni loader farà: 
    #sampling di archi positivi,
    #negative sampling
    #neighbor sampling
    #costruzione del sottografo locale del batch    
    train_loaders = build_link_loaders(
        data=train_graph,
        edge_index_by_relation=train_edge_index_by_relation,
        num_neighbors=num_neighbors,
        edge_batch_size=edge_batch_size,
        neg_sampling_ratio=neg_sampling_ratio,
        shuffle=True,
        num_workers=num_workers,
    )

    #stessa cosa per il validation set senza shuffle
    val_loaders = build_link_loaders(
        data=train_graph,
        edge_index_by_relation=val_edge_index_by_relation,
        num_neighbors=num_neighbors,
        edge_batch_size=edge_batch_size,
        neg_sampling_ratio=neg_sampling_ratio,
        shuffle=False, 
        num_workers=num_workers,
    )

    print("\n===== TRAINING MINI-BATCH LEVEL 4 =====")
    print(f"Device: {device}")
    print(f"Relazioni decoder: {decoder_relations}")
    print(f"Num neighbors: {num_neighbors}")
    print(f"Edge batch size: {edge_batch_size}")
    print(f"Negative sampling ratio: {neg_sampling_ratio}")
    print(f"Validation ratio: {val_ratio}")
    for rel in decoder_relations:
        split = edge_splits[rel]
        print(
            f"  {relation_key(rel)} | total={split['num_total']} | "
            f"train={split['num_train']} | val={split['num_val']}"
        )

    best_val_loss = math.inf
    best_state_dict = copy.deepcopy(model.state_dict())
    best_epoch = 0
    patience_counter = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()

        epoch_total_loss_sum = 0.0
        epoch_attr_loss_sum = 0.0
        epoch_struct_loss_sum = 0.0

        epoch_total_loss_weight = 0.0
        epoch_attr_loss_weight = 0.0
        epoch_struct_loss_weight = 0.0

        #randomizzazione dell'ordine delle relazioni, in modo che il modello non iteri sempre le relazioni nello stesso ordine
        #per evitare forme di bias o dipendenza dall'ordine delle relazioni
        relations_this_epoch = [rel for rel in decoder_relations if rel in train_loaders]
        random.shuffle(relations_this_epoch)

        for rel in relations_this_epoch:
        #per ogni relazione rel nel loader 
            #prendi il nome
            rel_name = relation_key(rel)
            #prendiamo il loader
            loader = train_loaders[rel]

            print(f"\n[Epoch {epoch:02d}] Training relation: {rel_name}")

            for step, batch in enumerate(loader, start=1):
            #per ogni batch, che contiene sottografo campionato, nodi necessari e gli archi con etichette 0 e 1
                batch = batch.to(device)

                optimizer.zero_grad() #azzeriamo i gradienti

                #forward del modello
                z_dict, x_hat_dict = model(batch.x_dict, batch.edge_index_dict)

                #loss attributiva del batch e numero di nodi nel batch
                loss_attr, _, attr_weight = attribute_reconstruction_loss(batch.x_dict, x_hat_dict)
                #loss strutturale del batch e numero di nodi nel batch
                loss_struct, struct_details, struct_weight = structural_loss_from_batch(
                    z_dict=z_dict,
                    batch=batch,
                    rel=rel,
                    rel_decoder=model.rel_decoder,
                    pos_weight=pos_edge_weight,
                    neg_weight=neg_edge_weight,
                )

                #loss totale del batch = alpha_attr * loss_attr + alpha_struct * loss_struct
                loss = alpha_attr * loss_attr + alpha_struct * loss_struct
                loss.backward() #calcolo dei gradienti per ogni parametro
                optimizer.step() #aggiornamento dei parametri

                total_weight = max(1, struct_weight) #definiamo il peso del batch come numero di archi presenti nel batch

                #aggiorniamo le metriche pesate dell'epoca
                epoch_total_loss_sum += loss.item() * total_weight
                epoch_attr_loss_sum += loss_attr.item() * max(1, attr_weight)
                epoch_struct_loss_sum += loss_struct.item() * max(1, struct_weight)

                epoch_total_loss_weight += total_weight
                epoch_attr_loss_weight += max(1, attr_weight)
                epoch_struct_loss_weight += max(1, struct_weight)

                if step % 50 == 0:
                    print(
                        f"  step {step:04d} | "
                        f"total={loss.item():.4f} | "
                        f"attr={loss_attr.item():.4f} | "
                        f"struct={loss_struct.item():.4f} | "
                        f"pos={struct_details['num_pos']} | "
                        f"neg={struct_details['num_neg']}"
                    )

                #pulizia delle variabili
                del batch, z_dict, x_hat_dict, loss, loss_attr, loss_struct

        #otteniamo la loss media (loss dei batch pesata per numero di archi nel batch) dell'epoca
        mean_total = epoch_total_loss_sum / max(1.0, epoch_total_loss_weight)
        mean_attr = epoch_attr_loss_sum / max(1.0, epoch_attr_loss_weight)
        mean_struct = epoch_struct_loss_sum / max(1.0, epoch_struct_loss_weight)

        # Fine epoca: liberiamo la memoria accumulata
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        #validation
        val_metrics = evaluate_model_minibatch(
            model=model,
            val_loaders=val_loaders,
            decoder_relations=decoder_relations,
            alpha_attr=alpha_attr,
            alpha_struct=alpha_struct,
            pos_edge_weight=pos_edge_weight,
            neg_edge_weight=neg_edge_weight,
        )
        
        scheduler.step(val_metrics["total"])

        #salvataggio nello storico
        history.append(
            {
                "epoch": epoch,
                "train_total": mean_total,
                "train_attr": mean_attr,
                "train_struct": mean_struct,
                "val_total": val_metrics["total"],
                "val_attr": val_metrics["attr"],
                "val_struct": val_metrics["struct"],
            }
        )

        print(
            f"\nEpoch {epoch:02d} COMPLETED | "
            f"train_total={mean_total:.4f} | "
            f"train_attr={mean_attr:.4f} | "
            f"train_struct={mean_struct:.4f} | "
            f"val_total={val_metrics['total']:.4f} | "
            f"val_attr={val_metrics['attr']:.4f} | "
            f"val_struct={val_metrics['struct']:.4f}"
        )

        #loss totale dell'ultima epoca
        current_val = val_metrics["total"]
        #miglioramento rispetto all'epoca migliore (= con loss totale più bassa)
        improvement = best_val_loss - current_val

        if improvement > early_stopping_min_delta:
        #se il miglioramento > early_stopping_min_delta
            #impostiamo le nuove migliori metriche ottenute dall'ultima epoca come migliori
            best_val_loss = current_val
            best_state_dict = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            #resettiamo il contatore dlela patience a 0
            patience_counter = 0
            print(f"Nuovo best model alla epoca {epoch:02d} | val_total={current_val:.4f}")
        else:
        #altrimenti auentiamo di 1 il contatore della patience
            patience_counter += 1
            print(
                f"Nessun miglioramento sufficiente in validation | "
                f"patience={patience_counter}/{early_stopping_patience}"
            )

        if patience_counter >= early_stopping_patience:
        #se il contatore della patience supera la soglia, fermiamo il training
            print(f"Early stopping attivato alla epoca {epoch:02d}")
            break

    #impostiamo il modello con i pesi che hanno ottenuto loss minore sul validation set
    model.load_state_dict(best_state_dict)
    print(f"\nBest model ripristinato dalla epoca {best_epoch:02d} | best_val_total={best_val_loss:.4f}")

    #restituisce 
    #model = modello migliore
    #decoder_relations = lista delle relazioni canoniche
    #history = storico completo delle metriche per epoca
    #edge_splits = i train/val degli archi
    #best_epoch = il numero dell'epoca che ha ottenuto il modello migliore
    return model, decoder_relations, history, edge_splits, best_epoch



########################################################################
## Inferenza ###########################################################
########################################################################

#compute_attribute_scores_minibatch esegue l’inferenza attributiva su tutto il grafo in mini-batch, 
#scorrendo ogni tipo di nodo con un NeighborLoader, ricostruendo le feature dei seed nodes e 
#salvando per ciascuno l’errore medio quadratico di ricostruzione come score di anomalia attributiva.
@torch.no_grad() #non consideriamo i gradienti: stiamo facendo inferenza e non training
def compute_attribute_scores_minibatch(
    model,
    data,
    num_neighbors: List[int],
    node_batch_size: int = 4096,
    num_workers: int = 0,
):
    #model = modello già allenato
    #data = grafo
    #num_neighbors = solito
    #node_batch_size = numero di nodi target per batch durante l'inferenza
    device = get_device()
    model.eval()

    #inizializzazione di un dizionario che conterrà uno score per ogni
    #nodo del grao, separato per tipo di nodo
    attr_scores = {
        nt: torch.zeros(data[nt].num_nodes, dtype=torch.float)
        for nt in data.node_types
    }

    print("\n===== ATTRIBUTE INFERENCE =====")

    for node_type in data.node_types:
    #per ogni tipo di nodo
        print(f"Node type: {node_type}")

        #loader che scorre tutti i nodi di quel tipo in mini-batch
        #campiona il loro intorno
        #costruisce un sottografo locale
        #restituisce il batch
        loader = build_node_inference_loader(
            data=data,
            node_type=node_type,
            num_neighbors=num_neighbors,
            batch_size=node_batch_size,
            num_workers=num_workers,
        )

        for step, batch in enumerate(loader, start=1):
        #per ogni batch
            batch = batch.to(device)

            #forward del modello
            z_dict, x_hat_dict = model(batch.x_dict, batch.edge_index_dict)

            #numero di nodi node_type totali nel minibatch e nei sottografi locali campionati
            seed_size = batch[node_type].batch_size
            #indici di questi nodi
            local_seed_idx = torch.arange(seed_size, device=device)

            #feature vere di questi nodi
            x_true = batch[node_type].x[local_seed_idx]
            #feature stimate di questi nodi
            x_pred = x_hat_dict[node_type][local_seed_idx]

            #score attributivo solo sui seed nodes
            err = ((x_pred - x_true) ** 2).mean(dim=1)

            global_ids = batch[node_type].n_id[local_seed_idx].cpu()

            attr_scores[node_type][global_ids] = err.detach().cpu()

            if step % 100 == 0:
                print(f"  step {step:04d}")

            #pulizia della memoria
            del batch, z_dict, x_hat_dict, x_true, x_pred, err

    # Fine inferenza attributiva: liberiamo la memoria
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return attr_scores


#compute_structural_scores_minibatch esegue l’inferenza strutturale su tutto il grafo in mini-batch, 
#valutando la plausibilità di ogni edge reale con il decoder relazionale, 
#trasformando tale plausibilità in un punteggio di anomalia dell’edge e 
#aggregandolo per media sui nodi incidenti per ottenere lo score strutturale finale di ogni nodo.
@torch.no_grad() #non consideriamo i gradienti: stiamo facendo inferenza e non training
def compute_structural_scores_minibatch(
    model,
    data,
    decoder_relations,
    num_neighbors: List[int],
    edge_batch_size: int = 4096,
    num_workers: int = 0,
):
    #model = modello già addestrato
    #data = grafo
    #decoder_relations = lista delle relazioni canoniche
    #num_neighbors = solito
    #edge_batch_size = numero di edge target per batch durante l'inferenza
    
    device = get_device()
    model.eval()

    #somma degli score strutturali per nodo: per ogni tipo di nodo creiamo un vettore che accumuli
    #la somma dei contributi di anomalia degli archi incidenti
    node_scores_sum = {
        nt: torch.zeros(data[nt].num_nodes, dtype=torch.float)
        for nt in data.node_types
    }
    #conteggio del numero di archi che contribuiscono a ogni nodo
    node_counts = {
        nt: torch.zeros(data[nt].num_nodes, dtype=torch.float)
        for nt in data.node_types
    }    

    print("\n===== STRUCTURAL INFERENCE =====")

    for rel in decoder_relations:
    #per ogni relazione canonica rel
        src_type, _, dst_type = rel
        rel_name = relation_key(rel)
        print(f"Relation: {rel_name}")

        #costruzione del loader che scorre tutti gli archi della relazione rel
        loader = build_edge_inference_loader(
            data=data,
            rel=rel,
            num_neighbors=num_neighbors,
            edge_batch_size=edge_batch_size,
            num_workers=num_workers,
        )

        for step, batch in enumerate(loader, start=1):
        #per ogni batch
            batch = batch.to(device)

            #forward del modello
            z_dict, _ = model(batch.x_dict, batch.edge_index_dict)

            #archi target
            edge_label_index = batch[rel].edge_label_index

            #facciamo calcolare al decoder relazionale gli score di plausibilità
            #per ciascun arco del batch            
            pos_scores = model.rel_decoder.score_edges(
                z_src=z_dict[src_type],
                z_dst=z_dict[dst_type],
                edge_index=edge_label_index,
                rel=rel,
            )

            #trasformiamo il punteggio :
            # -se score alto -> sigmoid(score) vicino a 1 -> log(sigmoid(score)) vicino a 0 -> -log(sigmoid(score)) piccolo
            # -se score basso o molto negativo -> -log(sigmoid(score)) è grande            
            edge_anom = -F.logsigmoid(pos_scores)

            local_src = edge_label_index[0]
            local_dst = edge_label_index[1]

            global_src = batch[src_type].n_id[local_src].cpu()
            global_dst = batch[dst_type].n_id[local_dst].cpu()
            edge_anom_cpu = edge_anom.detach().cpu()

            #accumulo dei contributi di anomalia sui nodi sorgente
            node_scores_sum[src_type].index_add_(0, global_src, edge_anom_cpu)
            #accumulo dei contributi di anomalia sui nodi destinazione
            node_scores_sum[dst_type].index_add_(0, global_dst, edge_anom_cpu)

            #contiamo quante volte un nodo ha ricevuto un contributo
            ones = torch.ones_like(edge_anom_cpu)
            node_counts[src_type].index_add_(0, global_src, ones)
            node_counts[dst_type].index_add_(0, global_dst, ones)

            if step % 100 == 0:
                print(f"  step {step:04d}")

            #pulizia della memoria
            del batch, z_dict, edge_label_index, pos_scores, edge_anom

    # Fine inferenza strutturale: liberiamo la memoria
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    structural_scores = {}
    for nt in data.node_types:
    #per ogni tipo di nodo
        #creiamo il vettore finale degli score
        structural_scores[nt] = torch.zeros_like(node_scores_sum[nt])
        #per distinguere nodi che hanno almeno un arco incidente considerato
        mask = node_counts[nt] > 0
        #media degli score per nodi con contributi
        structural_scores[nt][mask] = node_scores_sum[nt][mask] / node_counts[nt][mask]
        #se un nodo non ha archi incidenti, il suo score viene posto a 0 (non abbiamo indicazioni di anomalia)
        structural_scores[nt][~mask] = 0.0

    return structural_scores



########################################################################
## Loss ################################################################
########################################################################

def normalize_scores_ecdf(scores_dict):
    from scipy.stats import rankdata

    def ecdf_normalize(t: torch.Tensor):
        if t.numel() == 0:
            return t
        ranks = rankdata(t.numpy(), method="average")
        return torch.tensor(ranks / (len(ranks) + 1), dtype=torch.float)

    return {
        nt: ecdf_normalize(scores)
        for nt, scores in scores_dict.items()
    }

def compute_final_scores(
    attr_scores_norm: Dict[str, torch.Tensor],
    structural_scores_norm: Dict[str, torch.Tensor],
    lambda_attr_by_type: Dict[str, float],
    lambda_struct_by_type: Dict[str, float],
):
    #attr_scores = un dizionario di tensori (uno per tipo di nodo) contenenti lo score attributivo di tutti i nodi di quel tipo
    #structural_scores = un dizionario di tensori (uno per tipo di nodo) contenenti lo score strutturale di tutti i nodi di quel tipo
    #lambda_attr = un dizionario di pesi della componente attributiva (uno per tipo di nodo)
    #lambda_struct = un dizionario di pesi della componente strutturale (uno per tipo di nodo)
    final_scores = {}   #inizializzazione del dizionario finale

    for nt in attr_scores_norm.keys():
    #per ogni tipo di nodo
        #usa normalizzazione rank-based (ECDF) invece di z-score
        #combinali con una somma pesata con i rispettivi pesi
        lambda_attr = lambda_attr_by_type[nt]
        lambda_struct = lambda_struct_by_type[nt]

        final_scores[nt] = (
            lambda_attr * attr_scores_norm[nt]
            + lambda_struct * structural_scores_norm[nt]
        )

    return final_scores



########################################################################
## Export ##############################################################
########################################################################

#save_scores_to_csv costruisce, per ogni tipo di nodo, un DataFrame con score finale, 
#attributivo e strutturale, lo ordina per anomalia decrescente e lo salva in formato CSV, 
#rendendo i risultati del modello direttamente analizzabili.
def save_scores_to_csv(
    final_scores,
    attr_scores,
    structural_scores,
    out_dir=None,
):
    if out_dir is None:
        out_dir = REPORTS_DIR / "level4_results"

    #creazione della directory (se non esiste già)
    os.makedirs(out_dir, exist_ok=True)

    for node_type in final_scores.keys():
    #per ogni tipo di nodo
        #creiamo un dataframe che contiene:
        # -id nel grafo dei nodi di tipo node_type
        # -score finale dei nodi node_type
        # -score attributivo, non ancora standardizzato
        # -score strutturale non ancora standardizzato
        df = pd.DataFrame({
            "node_id": range(final_scores[node_type].size(0)),
            "final_score": final_scores[node_type].numpy(),
            "attribute_score": attr_scores[node_type].numpy(),
            "structural_score": structural_scores[node_type].numpy(),
        })

        #ordiniamo il dataframe dal più anomalo al meno anomalo
        df = df.sort_values("final_score", ascending=False).reset_index(drop=True)
        #costruzione del path dove verranno salvati i csv
        path = os.path.join(out_dir, f"anomaly_scores_{node_type}.csv")
        #trasformazioone in .csv eliminando gli indici pandas
        df.to_csv(path, index=False)
        print(f"Salvato: {path}")

#export della training history con metriche train/validation per ogni epoca.
def save_training_history(history, out_path):
    df = pd.DataFrame(history)
    df.to_csv(out_path, index=False)
    print(f"Salvato: {out_path}")


def print_top_anomalies(final_scores, k=10):
    print("\n===== TOP ANOMALIES =====")
    for node_type, scores in final_scores.items():
        values, indices = torch.topk(scores, k=min(k, scores.numel()))
        print(f"\nNode type: {node_type}")
        for rank, (idx, val) in enumerate(zip(indices.tolist(), values.tolist()), start=1):
            print(f"{rank:2d}. node_id={idx:<10d} score={val:.4f}")


def main():
    set_seed(42)

    print(f"Caricamento grafo da {MODELS_DIR / 'PA_graph.pt'} ...")
    data = torch.load(MODELS_DIR / "PA_graph.pt", weights_only=False)

    print("\n===== GRAFO CARICATO =====")
    print(data)
    print("\nNode types:", data.node_types)
    print("Edge types:", data.edge_types)

    hidden_dim = 32
    latent_dim = 16

    epochs = 30
    lr = 2e-4
    weight_decay = 5e-3

    alpha_attr = 1.0
    alpha_struct = 0.5
    
    lambda_attr_by_type = {
        "SA": 0.1,
        "CIG": 0.5,
        "AGZ": 0.8,
        "AG": 0.1,
    }
    
    lambda_struct_by_type = {
        "SA": 0.9,
        "CIG": 0.5,
        "AGZ": 0.2,
        "AG": 0.9,
    }    
    
    dropout = 0.3

    num_neighbors = [15, 10]

    edge_batch_size = 4096
    neg_sampling_ratio = 1.0

    node_infer_batch_size = 4096
    edge_infer_batch_size = 4096

    num_workers = 4

    val_ratio = 0.10
    early_stopping_patience = 10
    early_stopping_min_delta = 1e-4
    pos_edge_weight = 0.5
    neg_edge_weight = 0.5

    model, decoder_relations, history, edge_splits, best_epoch = train_model_minibatch(
        data=data,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        epochs=epochs,
        lr=lr,
        dropout=dropout,
        weight_decay=weight_decay,
        alpha_attr=alpha_attr,
        alpha_struct=alpha_struct,
        num_neighbors=num_neighbors,
        edge_batch_size=edge_batch_size,
        neg_sampling_ratio=neg_sampling_ratio,
        num_workers=num_workers,
        val_ratio=val_ratio,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        pos_edge_weight=pos_edge_weight,
        neg_edge_weight=neg_edge_weight,
        seed=42,
    )

    attr_scores = compute_attribute_scores_minibatch(
        model=model,
        data=data,
        num_neighbors=num_neighbors,
        node_batch_size=node_infer_batch_size,
        num_workers=num_workers,
    )

    structural_scores = compute_structural_scores_minibatch(
        model=model,
        data=data,
        decoder_relations=decoder_relations,
        num_neighbors=num_neighbors,
        edge_batch_size=edge_infer_batch_size,
        num_workers=num_workers,
    )
    
    attr_scores_norm = normalize_scores_ecdf(attr_scores)
    structural_scores_norm = normalize_scores_ecdf(structural_scores) 
    
    final_scores = compute_final_scores(
        attr_scores_norm=attr_scores_norm,
        structural_scores_norm=structural_scores_norm,
        lambda_attr_by_type=lambda_attr_by_type,
        lambda_struct_by_type=lambda_struct_by_type,
    )    

    (REPORTS_DIR / "level4_results").mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "final_scores": final_scores,
            "attribute_scores_ecdf": attr_scores_norm,
            "structural_scores_ecdf": structural_scores_norm,
            "decoder_relations": decoder_relations,
            "training_history": history,
            "edge_splits": {
                relation_key(rel): {
                    "num_total": split["num_total"],
                    "num_train": split["num_train"],
                    "num_val": split["num_val"],
                }
                for rel, split in edge_splits.items()
            },
            "best_epoch": best_epoch,
            "hyperparameters": {
                "hidden_dim": hidden_dim,
                "latent_dim": latent_dim,
                "epochs": epochs,
                "lr": lr,
                "weight_decay": weight_decay,
                "alpha_attr": alpha_attr,
                "alpha_struct": alpha_struct,
                "lambda_attr_by_type": lambda_attr_by_type,
                "lambda_struct_by_type": lambda_struct_by_type,  
                "num_neighbors": num_neighbors,
                "edge_batch_size": edge_batch_size,
                "neg_sampling_ratio": neg_sampling_ratio,
                "val_ratio": val_ratio,
                "early_stopping_patience": early_stopping_patience,
                "early_stopping_min_delta": early_stopping_min_delta,
                "pos_edge_weight": pos_edge_weight,
                "neg_edge_weight": neg_edge_weight,
            }
        },
        MODELS_DIR / "level4_minibatch_outputs.pt"
    )

    print(f"\nSalvato: {MODELS_DIR / 'level4_minibatch_outputs.pt'}")

    save_scores_to_csv(
        final_scores=final_scores,
        attr_scores=attr_scores_norm,
        structural_scores=structural_scores_norm,
        out_dir=REPORTS_DIR / "level4_results",
    )

    save_training_history(
        history=history,
        out_path=REPORTS_DIR / "level4_results" / "training_history.csv",
    )

    print_top_anomalies(final_scores, k=10)


if __name__ == "__main__":
    main()
