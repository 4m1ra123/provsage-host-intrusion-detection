"""Train C2 parametré, déterministe, sampling [15,10]/26000 (papier)."""
import os, time, argparse
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
from gensim.models import FastText

from utils.det_utils import set_deterministic
set_deterministic(42)

from model.PositionalEncoder import PositionalEncoder
from model.TGN import TemporalGraphNetwork
from model.models_c2 import build_gnn, build_fusion, count_parameters
from utils.graphutils_trace import add_attributes, prepare_graph
from utils.fasttext_utils import train_fasttext


encoder = PositionalEncoder(30)


def infer_fasttext_embedding(document, ftmodel):
    if not document:
        return np.zeros(30)
    word_embeddings = [ftmodel.wv[word] for word in document]
    out = np.array(word_embeddings)
    out = torch.tensor(out, dtype=torch.float)
    out = encoder.embed(out).detach().cpu().numpy()
    return np.mean(out, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gnn", default="gat", choices=["gat", "graphsage", "gcn"])
    parser.add_argument("--fusion", default="mha", choices=["mha", "pooling", "averaging"])
    args = parser.parse_args()
    print(f"=== C2 TRAIN : GNN={args.gnn} | FUSION={args.fusion} ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open("../data/trace/trace_train.txt") as f:
        data = f.read().split('\n')
    data = [line.split('\t') for line in data if line]
    df = pd.DataFrame(data, columns=['actorID','actor_type','objectID','object','action','timestamp'])
    df = df.dropna()
    df.sort_values(by='timestamp', ascending=True, inplace=True)
    df = add_attributes(df, "../data/trace/ta1-trace-e3-official-1.json")
    phrases, labels, edges, mapp, node_types, edges_attr = prepare_graph(df)
    print(f"Phrases: {len(phrases)}")

    model_path = "fasttext_trace.model"
    if os.path.exists(model_path):
        ftmodel = FastText.load(model_path)
        print("FastText model loaded.")
    else:
        ftmodel = train_fasttext(phrases, "trace",
                                 vector_size=30, window=5, min_count=1,
                                 workers=1, epochs=300, seed=42)
        print("FastText model trained and saved.")

    num_classes = 12
    gnn_model = build_gnn(args.gnn, in_channel=12, out_channel=30, device=device)
    tgn_model = TemporalGraphNetwork(in_channels=60, out_channels=32).to(device)
    fusion_model = build_fusion(args.fusion, in_channels=32, out_channels=12,
                                num_classes=num_classes, device=device, num_heads=4)
    p_total = count_parameters(gnn_model)+count_parameters(tgn_model)+count_parameters(fusion_model)
    print(f"[PARAMS] TOTAL={p_total:,}")

    optimizer = optim.Adam(
        list(fusion_model.parameters())+list(tgn_model.parameters())+list(gnn_model.parameters()),
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

    weights_dir = f"../trained_weights/trace/c2_{args.gnn}_{args.fusion}"
    os.makedirs(weights_dir, exist_ok=True)

    t0 = time.time()
    for epoch in range(50):
        print(f"Epoch: {epoch}")
        loader = NeighborLoader(graph, num_neighbors=[15,10], batch_size=26000, input_nodes=mask)
        gnn_model.train(); tgn_model.train(); fusion_model.train()
        for subg in tqdm(loader, desc="Training", leave=False):
            subg = subg.to(device)
            optimizer.zero_grad()
            cps = [phrases[i] for i in subg.n_id.tolist()]
            ft = [infer_fasttext_embedding(p, ftmodel) for p in cps]
            ft = torch.tensor(np.array(ft), dtype=torch.float).to(device)
            gnn_out = gnn_model(subg.x, subg.edge_index)
            combined = torch.cat((gnn_out, ft), dim=1)
            tgn_out = tgn_model(combined, subg.edge_index, subg.edge_time).unsqueeze(1)
            final = fusion_model(tgn_out)
            loss = criterion(final.view(-1, num_classes), subg.y)
            loss.backward(); optimizer.step()
            del subg; torch.cuda.empty_cache()

        loader = NeighborLoader(graph, num_neighbors=[15,10], batch_size=26000, input_nodes=mask)
        gnn_model.eval(); tgn_model.eval(); fusion_model.eval()
        for subg in tqdm(loader, desc="Eval-mask", leave=False):
            subg = subg.to(device)
            with torch.no_grad():
                cps = [phrases[i] for i in subg.n_id.tolist()]
                ft = [infer_fasttext_embedding(p, ftmodel) for p in cps]
                ft = torch.tensor(np.array(ft), dtype=torch.float).to(device)
                gnn_out = gnn_model(subg.x, subg.edge_index)
                combined = torch.cat((gnn_out, ft), dim=1)
                tgn_out = tgn_model(combined, subg.edge_index, subg.edge_time).unsqueeze(1)
                final = fusion_model(tgn_out)
                s, idx = final.view(-1, num_classes).sort(dim=1, descending=True)
                conf = (s[:,0]-s[:,1])/s[:,0]; conf = (conf-conf.min())/conf.max()
                pred = idx[:,0]
                cond = (pred == subg.y) & (conf >= 0.8)
                mask[subg.n_id[cond]] = False

        torch.save(gnn_model.state_dict(),    f'{weights_dir}/gnn{epoch}.pth')
        torch.save(tgn_model.state_dict(),    f'{weights_dir}/tgn{epoch}.pth')
        torch.save(fusion_model.state_dict(), f'{weights_dir}/fusion{epoch}.pth')

    print(f"[TIMING] train_seconds={time.time()-t0:.1f}")
    print(f"[RESUME] gnn={args.gnn} fusion={args.fusion} params={p_total}")
    print("done")


if __name__ == "__main__":
    main()
