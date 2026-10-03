"""Eval STGAN C2 avec embedding Word2Vec (cached pre-encoding)."""
import os, json, time, argparse
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm
from gensim.models import Word2Vec
from torch_geometric import utils

from utils.det_utils import set_deterministic
set_deterministic(42)

from utils.helper import helper
from utils.graphutils_trace import add_attributes, prepare_graph
from model.PositionalEncoder import PositionalEncoder
from model.TGN import TemporalGraphNetwork
from model.models_c2 import build_gnn, build_fusion, count_parameters


w2vmodel = Word2Vec.load("../train/word2vec_trace.model")
encoder = PositionalEncoder(30)


def infer_w2v(document):
    if not document:
        return np.zeros(30)
    we = []
    for w in document:
        if w in w2vmodel.wv:
            we.append(w2vmodel.wv[w])
    if not we:
        return np.zeros(30)
    out = np.array(we); out = torch.tensor(out, dtype=torch.float)
    out = encoder.embed(out).detach().cpu().numpy()
    return np.mean(out, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gnn", default="graphsage", choices=["gat","graphsage","gcn"])
    parser.add_argument("--fusion", default="mha", choices=["mha","pooling","averaging"])
    args = parser.parse_args()
    print(f"=== C2 EVAL W2V : GNN={args.gnn} | FUSION={args.fusion} ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

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

    num_classes = 12
    gnn_model = build_gnn(args.gnn, in_channel=12, out_channel=30, device=device)
    tgn_model = TemporalGraphNetwork(in_channels=60, out_channels=32).to(device)
    fusion_model = build_fusion(args.fusion, in_channels=32, out_channels=12,
                                num_classes=num_classes, device=device, num_heads=4)
    p_total = count_parameters(gnn_model)+count_parameters(tgn_model)+count_parameters(fusion_model)
    print(f"[PARAMS] TOTAL={p_total:,}")

    t0 = time.time()
    print("Pre-encoding W2V embeddings (une seule fois)...")
    w2v_all = np.zeros((len(phrases), 30), dtype=np.float32)
    for i, p in enumerate(tqdm(phrases, desc="W2V pre-encode")):
        w2v_all[i] = infer_w2v(p)
    w2v_all = torch.tensor(w2v_all, dtype=torch.float)

    graph = Data(
        x=torch.tensor(np.array(node_types), dtype=torch.float).to(device),
        y=torch.tensor(labels, dtype=torch.long).to(device),
        edge_index=torch.tensor(edges, dtype=torch.long).to(device),
        edge_time=torch.tensor(edges_attr, dtype=torch.float).to(device))
    graph.n_id = torch.arange(graph.num_nodes).to(device)
    flag = torch.tensor([True]*graph.num_nodes, dtype=torch.bool).to(device)

    weights_dir = f"../trained_weights/trace/c2_w2v_{args.gnn}_{args.fusion}"

    for m_n in range(50):
        print(f"Epoch: {m_n}")
        gnn_model.load_state_dict(torch.load(f'{weights_dir}/gnn{m_n}.pth', map_location=device))
        gnn_model.eval()
        tgn_model.load_state_dict(torch.load(f'{weights_dir}/tgn{m_n}.pth', map_location=device))
        tgn_model.eval()
        fusion_model.load_state_dict(torch.load(f'{weights_dir}/fusion{m_n}.pth', map_location=device))
        fusion_model.eval()

        loader = NeighborLoader(graph, num_neighbors=[-1,-1], batch_size=26000)
        for subg in tqdm(loader, desc="Eval", leave=False):
            subg.x = subg.x.to(device); subg.edge_index = subg.edge_index.to(device)
            subg.y = subg.y.to(device); subg.edge_time = subg.edge_time.to(device)
            with torch.no_grad():
                w2v = w2v_all[subg.n_id.cpu()].to(device)
                gnn_out = gnn_model(subg.x, subg.edge_index)
                combined = torch.cat((gnn_out, w2v), dim=1)
                tgn_out = tgn_model(combined, subg.edge_index, subg.edge_time).unsqueeze(1)
                final = fusion_model(tgn_out)
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
    print(f"[RESUME] emb=w2v gnn={args.gnn} fusion={args.fusion} params={p_total}")


if __name__ == "__main__":
    main()
