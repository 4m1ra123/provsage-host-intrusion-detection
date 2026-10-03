"""Train STGAN avec embedding BERT (frozen + projection apprise)."""
import os, time
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch.nn import CrossEntropyLoss
from sklearn.utils import class_weight
from tqdm import tqdm

from utils.det_utils import set_deterministic
set_deterministic(42)

from utils.bert_utils import load_bert, encode_phrases
from model.GAT import GAT
from model.TGN import TemporalGraphNetwork
from model.MultiHeadSelfAttention import MultiHeadSelfAttentionModel
from utils.graphutils_trace import add_attributes, prepare_graph


class BertProjection(nn.Module):
    def __init__(self, in_dim=384, out_dim=30):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        return self.linear(x)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== C1 BERT TRAIN ===")
    print(f"Device: {device}")

    # ---------- Data ----------
    with open("../data/trace/trace_train.txt") as f:
        data = f.read().split('\n')
    data = [line.split('\t') for line in data if line]
    df = pd.DataFrame(data, columns=['actorID','actor_type','objectID','object','action','timestamp'])
    df = df.dropna()
    df.sort_values(by='timestamp', ascending=True, inplace=True)
    df = add_attributes(df, "../data/trace/ta1-trace-e3-official-1.json")
    phrases, labels, edges, mapp, node_types, edges_attr = prepare_graph(df)
    print(f"Phrases: {len(phrases)}")

    # ---------- BERT cache ----------
    cache = "bert_trace_train_embeddings.pt"
    if os.path.exists(cache):
        bert_emb = torch.load(cache, map_location="cpu")
        print(f"BERT embeddings loaded from cache. Shape: {bert_emb.shape}")
    else:
        print("Computing BERT embeddings (one-time)...")
        bert = load_bert(device=device)
        bert_emb = encode_phrases(bert, phrases, batch_size=256).cpu()
        torch.save(bert_emb, cache)
        print(f"BERT embeddings cached. Shape: {bert_emb.shape}")
        del bert; torch.cuda.empty_cache()
    bert_emb = bert_emb.to(device)
    bert_dim = bert_emb.shape[1]

    # ---------- Models ----------
    num_classes = 12
    gat  = GAT(in_channel=12, out_channel=30).to(device)
    tgn  = TemporalGraphNetwork(in_channels=60, out_channels=32).to(device)
    mha  = MultiHeadSelfAttentionModel(in_channels=32, out_channels=12,
                                        num_classes=num_classes, num_heads=4).to(device)
    proj = BertProjection(in_dim=bert_dim, out_dim=30).to(device)

    p_total = sum(p.numel() for m in [gat, tgn, mha, proj] for p in m.parameters() if p.requires_grad)
    print(f"[PARAMS] TOTAL={p_total:,}")

    optimizer = optim.Adam(
        list(mha.parameters())+list(tgn.parameters())+list(gat.parameters())+list(proj.parameters()),
        lr=0.0005, weight_decay=5e-4)

    cw = class_weight.compute_class_weight(class_weight=None, classes=np.arange(num_classes), y=labels)
    cw = torch.tensor(cw, dtype=torch.float).to(device)
    criterion = CrossEntropyLoss(weight=cw, reduction='mean')

    graph = Data(
        x=torch.tensor(np.array(node_types), dtype=torch.float).to(device),
        y=torch.tensor(labels, dtype=torch.long).to(device),
        edge_index=torch.tensor(edges, dtype=torch.long).to(device),
        edge_time=torch.tensor(edges_attr, dtype=torch.float).to(device))
    graph.n_id = torch.arange(graph.num_nodes)
    mask = torch.tensor([True]*graph.num_nodes, dtype=torch.bool)

    weights_dir = "../trained_weights/trace/c1_bert"
    os.makedirs(weights_dir, exist_ok=True)

    t0 = time.time()
    for epoch in range(50):
        print(f"Epoch: {epoch}")
        loader = NeighborLoader(graph, num_neighbors=[15,10], batch_size=26000, input_nodes=mask)
        for m in (gat, tgn, mha, proj): m.train()
        for subg in tqdm(loader, desc="Training", leave=False):
            subg = subg.to(device)
            optimizer.zero_grad()
            sem = proj(bert_emb[subg.n_id])
            gat_out = gat(subg.x, subg.edge_index)
            combined = torch.cat((gat_out, sem), dim=1)
            tgn_out = tgn(combined, subg.edge_index, subg.edge_time).unsqueeze(1)
            final = mha(tgn_out)
            loss = criterion(final.view(-1, num_classes), subg.y)
            loss.backward(); optimizer.step()
            del subg; torch.cuda.empty_cache()

        # eval-mask step (self-distillation)
        loader = NeighborLoader(graph, num_neighbors=[15,10], batch_size=26000, input_nodes=mask)
        for m in (gat, tgn, mha, proj): m.eval()
        for subg in tqdm(loader, desc="Eval-mask", leave=False):
            subg = subg.to(device)
            with torch.no_grad():
                sem = proj(bert_emb[subg.n_id])
                gat_out = gat(subg.x, subg.edge_index)
                combined = torch.cat((gat_out, sem), dim=1)
                tgn_out = tgn(combined, subg.edge_index, subg.edge_time).unsqueeze(1)
                final = mha(tgn_out)
                s, idx = final.view(-1, num_classes).sort(dim=1, descending=True)
                conf = (s[:,0]-s[:,1])/s[:,0]; conf = (conf-conf.min())/conf.max()
                pred = idx[:,0]
                cond = (pred == subg.y) & (conf >= 0.8)
                mask[subg.n_id[cond]] = False

        torch.save(gat.state_dict(),  f'{weights_dir}/gnn{epoch}.pth')
        torch.save(tgn.state_dict(),  f'{weights_dir}/tgn{epoch}.pth')
        torch.save(mha.state_dict(),  f'{weights_dir}/fusion{epoch}.pth')
        torch.save(proj.state_dict(), f'{weights_dir}/proj{epoch}.pth')

    print(f"[TIMING] train_seconds={time.time()-t0:.1f}")
    print(f"[RESUME] emb=bert gnn=gat fusion=mha params={p_total}")
    print("done")


if __name__ == "__main__":
    main()
