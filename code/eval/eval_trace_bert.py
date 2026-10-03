"""Eval STGAN avec embedding BERT (cache embeddings + projection chargée)."""
import os, json, time, sys
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm
from torch_geometric import utils

from utils.det_utils import set_deterministic
set_deterministic(42)

from utils.helper import helper
from utils.graphutils_trace import add_attributes, prepare_graph
from utils.bert_utils import load_bert, encode_phrases
from model.GAT import GAT
from model.TGN import TemporalGraphNetwork
from model.MultiHeadSelfAttention import MultiHeadSelfAttentionModel

# import projection class
sys.path.insert(0, "../train")
from train_trace_bert import BertProjection


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== C1 BERT EVAL ===")

    # ---------- Data ----------
    with open("../data/trace/trace_test.txt") as f:
        data = f.read().split('\n')
    data = [line.split('\t') for line in data if line]
    df = pd.DataFrame(data, columns=['actorID','actor_type','objectID','object','action','timestamp'])
    df = df.dropna(); df.sort_values(by='timestamp', ascending=True, inplace=True)
    df = add_attributes(df, "../data/trace/ta1-trace-e3-official-1.json.4")
    with open("../data/trace/trace.json") as f:
        GT_mal = set(json.load(f))
    phrases, labels, edges, mapp, node_types, edges_attr = prepare_graph(df)
    all_ids = set(list(df['actorID']) + list(df['objectID']))

    # ---------- BERT cache (test) ----------
    cache = "bert_trace_test_embeddings.pt"
    if os.path.exists(cache):
        bert_emb = torch.load(cache, map_location="cpu")
        print(f"BERT test embeddings loaded from cache. Shape: {bert_emb.shape}")
    else:
        bert = load_bert(device=device)
        bert_emb = encode_phrases(bert, phrases, batch_size=256).cpu()
        torch.save(bert_emb, cache)
        print(f"BERT test embeddings cached. Shape: {bert_emb.shape}")
        del bert; torch.cuda.empty_cache()
    bert_emb = bert_emb.to(device)

    # ---------- Models ----------
    num_classes = 12
    gat  = GAT(in_channel=12, out_channel=30).to(device)
    tgn  = TemporalGraphNetwork(in_channels=60, out_channels=32).to(device)
    mha  = MultiHeadSelfAttentionModel(in_channels=32, out_channels=12,
                                        num_classes=num_classes, num_heads=4).to(device)
    proj = BertProjection(in_dim=bert_emb.shape[1], out_dim=30).to(device)

    graph = Data(
        x=torch.tensor(np.array(node_types), dtype=torch.float).to(device),
        y=torch.tensor(labels, dtype=torch.long).to(device),
        edge_index=torch.tensor(edges, dtype=torch.long).to(device),
        edge_time=torch.tensor(edges_attr, dtype=torch.float).to(device))
    graph.n_id = torch.arange(graph.num_nodes).to(device)
    flag = torch.tensor([True]*graph.num_nodes, dtype=torch.bool).to(device)

    weights_dir = "../trained_weights/trace/c1_bert"

    t0 = time.time()
    for m_n in range(50):
        print(f"Epoch: {m_n}")
        gat.load_state_dict(torch.load(f'{weights_dir}/gnn{m_n}.pth', map_location=device))
        tgn.load_state_dict(torch.load(f'{weights_dir}/tgn{m_n}.pth', map_location=device))
        mha.load_state_dict(torch.load(f'{weights_dir}/fusion{m_n}.pth', map_location=device))
        proj.load_state_dict(torch.load(f'{weights_dir}/proj{m_n}.pth', map_location=device))
        for m in (gat, tgn, mha, proj): m.eval()

        loader = NeighborLoader(graph, num_neighbors=[-1,-1], batch_size=26000)
        for subg in tqdm(loader, desc="Eval", leave=False):
            subg.x = subg.x.to(device); subg.edge_index = subg.edge_index.to(device)
            subg.y = subg.y.to(device); subg.edge_time = subg.edge_time.to(device)
            with torch.no_grad():
                sem = proj(bert_emb[subg.n_id.cpu()].to(device))
                gat_out = gat(subg.x, subg.edge_index)
                combined = torch.cat((gat_out, sem), dim=1)
                tgn_out = tgn(combined, subg.edge_index, subg.edge_time).unsqueeze(1)
                final = mha(tgn_out)
                s, idx = final.view(-1, num_classes).sort(dim=1, descending=True)
                conf = (s[:,0]-s[:,1])/s[:,0]; conf = (conf-conf.min())/conf.max()
                pred = idx[:,0]
                cond = (pred == subg.y)
                flag[subg.n_id[cond]] = torch.logical_and(
                    flag[subg.n_id[cond]],
                    torch.tensor([False]*len(flag[subg.n_id[cond]]), dtype=torch.bool).to(device))

    print(f"[TIMING] infer_seconds={time.time()-t0:.1f}")
    index = utils.mask_to_index(flag).tolist()
    ids = set([mapp[x] for x in index])
    alerts = helper(set(ids), set(all_ids), GT_mal, edges, mapp)
    print(f"[RESUME] emb=bert gnn=gat fusion=mha")


if __name__ == "__main__":
    main()
