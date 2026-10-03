"""Eval avec les poids fournis par les auteurs du papier STGAN.

Différences vs eval_trace.py original :
- charge depuis trained_weights/trace_paper/
- adaptateur state_dict pour les vieux noms PyG (lin_src/lin_dst -> lin)
- échantillonnage complet du papier : num_neighbors=[-1,-1], batch_size=26000
- utilise notre word2vec (le papier ne fournit pas le sien) — caveat documenté
"""

import os, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm
from gensim.models import Word2Vec
from torch_geometric import utils

from utils.helper import helper
from utils.graphutils_trace import add_attributes, prepare_graph
from model.MultiHeadSelfAttention import MultiHeadSelfAttentionModel
from model.GAT import GAT
from model.PositionalEncoder import PositionalEncoder
from model.TGN import TemporalGraphNetwork


w2vmodel = Word2Vec.load("../train/word2vec_trace.model")
encoder = PositionalEncoder(30)


def infer_word2vec_embedding(document):
    word_embeddings = [w2vmodel.wv[word] for word in document if word in w2vmodel.wv]
    if not word_embeddings:
        return np.zeros(30)
    out = np.array(word_embeddings)
    out = torch.tensor(out, dtype=torch.float)
    out = encoder.embed(out).detach().cpu().numpy()
    return np.mean(out, axis=0)


def adapt_paper_state_dict(sd, verbose=False):
    """Rename lin_src -> lin, drop lin_dst (legacy PyG GATConv naming)."""
    new_sd, renamed, dropped = {}, [], []
    for k, v in sd.items():
        if 'lin_src.weight' in k:
            new_k = k.replace('lin_src.weight', 'lin.weight')
            new_sd[new_k] = v
            renamed.append(k)
        elif 'lin_dst' in k:
            dropped.append(k)
        else:
            new_sd[k] = v
    if verbose and (renamed or dropped):
        print(f"  state_dict: renamed {len(renamed)} lin_src->lin, dropped {len(dropped)} lin_dst")
    return new_sd


def load_with_adapter(model, path, device, verbose=False):
    sd = torch.load(path, map_location=device)
    sd_new = adapt_paper_state_dict(sd, verbose=verbose)
    missing, unexpected = model.load_state_dict(sd_new, strict=False)
    if verbose:
        if missing:    print(f"  WARN missing: {missing[:3]}{'...' if len(missing)>3 else ''}")
        if unexpected: print(f"  WARN unexpected: {unexpected[:3]}{'...' if len(unexpected)>3 else ''}")
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open("../data/trace/trace_test.txt") as f:
        data = f.read().split('\n')
    data = [line.split('\t') for line in data if line]
    df = pd.DataFrame(data, columns=['actorID', 'actor_type', 'objectID', 'object', 'action', 'timestamp'])
    df = df.dropna()
    df.sort_values(by='timestamp', ascending=True, inplace=True)
    df = add_attributes(df, "../data/trace/ta1-trace-e3-official-1.json.4")
    with open("../data/trace/trace.json", "r") as f:
        GT_mal = set(json.load(f))
    phrases, labels, edges, mapp, node_types, edges_attr = prepare_graph(df)
    all_ids = set(list(df['actorID']) + list(df['objectID']))

    gat_model = GAT(in_channel=12, out_channel=30).to(device)
    tgn_model = TemporalGraphNetwork(in_channels=60, out_channels=32).to(device)
    num_classes = 12
    sa_model = MultiHeadSelfAttentionModel(in_channels=32, out_channels=12,
                                            num_classes=num_classes, num_heads=4).to(device)

    graph = Data(
        x=torch.tensor(np.array(node_types), dtype=torch.float).to(device),
        y=torch.tensor(labels, dtype=torch.long).to(device),
        edge_index=torch.tensor(edges, dtype=torch.long).to(device),
        edge_time=torch.tensor(edges_attr, dtype=torch.float).to(device),
    )
    graph.n_id = torch.arange(graph.num_nodes).to(device)
    flag = torch.tensor([True] * graph.num_nodes, dtype=torch.bool).to(device)
    print("Graph Created")

    weights_dir = "../trained_weights/trace_paper"
    print(f"Weights from : {weights_dir}")

    start = time.time()
    for m_n in range(50):
        verbose = (m_n == 0)
        if verbose:
            print(f"\n--- Loading paper weights for epoch {m_n} (verbose) ---")
        load_with_adapter(gat_model, f'{weights_dir}/gat{m_n}.pth', device, verbose=verbose)
        gat_model.eval()
        load_with_adapter(tgn_model, f'{weights_dir}/tgn{m_n}.pth', device, verbose=verbose)
        tgn_model.eval()
        load_with_adapter(sa_model, f'{weights_dir}/selfattention{m_n}.pth', device, verbose=verbose)
        sa_model.eval()

        print(f"Epoch: {m_n}")
        loader = NeighborLoader(graph, num_neighbors=[-1, -1], batch_size=26000)
        for subg in tqdm(loader, desc="Eval", leave=False):
            subg.x = subg.x.to(device)
            subg.edge_index = subg.edge_index.to(device)
            subg.y = subg.y.to(device)
            subg.edge_time = subg.edge_time.to(device)
            with torch.no_grad():
                cps = [phrases[idx] for idx in subg.n_id.tolist()]
                w2v_emb = [infer_word2vec_embedding(p) for p in cps]
                w2v_out = torch.tensor(np.array(w2v_emb), dtype=torch.float).to(device)

                gat_out = gat_model(subg.x, subg.edge_index)
                combined = torch.cat((gat_out, w2v_out), dim=1)
                tgn_out = tgn_model(combined, subg.edge_index, subg.edge_time)
                tgn_out = tgn_out.unsqueeze(1)
                final_out = sa_model(tgn_out)

                sorted_, indices = final_out.view(-1, num_classes).sort(dim=1, descending=True)
                conf = (sorted_[:, 0] - sorted_[:, 1]) / sorted_[:, 0]
                conf = (conf - conf.min()) / conf.max()
                pred = indices[:, 0]
                cond = (pred == subg.y)
                flag[subg.n_id[cond]] = torch.logical_and(
                    flag[subg.n_id[cond]],
                    torch.tensor([False] * len(flag[subg.n_id[cond]]), dtype=torch.bool).to(device),
                )

    print(f"\n[TIMING] Eval done in {(time.time()-start)/60:.1f} min")
    index = utils.mask_to_index(flag).tolist()
    ids = set([mapp[x] for x in index])
    alerts = helper(set(ids), set(all_ids), GT_mal, edges, mapp)


if __name__ == "__main__":
    main()
