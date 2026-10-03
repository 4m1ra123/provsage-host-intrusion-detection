"""C3 — Visualisation du voisinage 1-hop des 7 TP-anchors.

Production : figure PNG haute résolution + JSON de métriques pour le chapitre 6 du mémoire."""

import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from scipy.sparse import csr_matrix

from utils.helper import helper
from utils.graphutils_trace import add_attributes, prepare_graph


def load_data():
    with open("../data/trace/trace_test.txt") as f:
        data = f.read().split('\n')
    data = [line.split('\t') for line in data if line]
    df = pd.DataFrame(data, columns=['actorID','actor_type','objectID','object','action','timestamp'])
    df = df.dropna(); df.sort_values(by='timestamp', ascending=True, inplace=True)
    df = add_attributes(df, "../data/trace/ta1-trace-e3-official-1.json.4")
    with open("../data/trace/trace.json") as f:
        GT_mal = set(json.load(f))
    phrases, labels, edges, mapp, node_types, edges_attr = prepare_graph(df)
    return GT_mal, phrases, labels, edges, mapp, node_types, edges_attr


def build_sparse_adj(edges, n_nodes):
    src = np.array(edges[0], dtype=np.int64)
    dst = np.array(edges[1], dtype=np.int64)
    data = np.ones(len(src), dtype=np.float32)
    return csr_matrix((data, (src, dst)), shape=(n_nodes, n_nodes))


def main():
    print("=== C3 — Attack Neighborhood Visualization ===\n")

    # Load saved flag mask
    flag = np.load("c3_outputs_v3_graphsage_mha/flag_mask.npy")
    print(f"Flag mask loaded: {flag.sum()} anchors")

    GT_mal, phrases, labels, edges, mapp, node_types, edges_attr = load_data()
    n_nodes = len(phrases)
    detected_idx = np.where(flag)[0]

    # ----- Identify the 7 TP-anchors -----
    tp_anchors_idx = np.array([i for i in detected_idx if mapp[i] in GT_mal])
    print(f"TP-anchors identified: {len(tp_anchors_idx)}")
    print(f"Their IDs: {[mapp[i] for i in tp_anchors_idx]}")
    print(f"Their types: {[int(np.argmax(node_types[i])) for i in tp_anchors_idx]}")

    # ----- Build adjacency + 1-hop closure -----
    adj_fwd = build_sparse_adj(edges, n_nodes)
    adj_sym = (adj_fwd + adj_fwd.T).tocsr()

    closure = set(tp_anchors_idx.tolist())
    for node in tp_anchors_idx:
        start, end = adj_sym.indptr[node], adj_sym.indptr[node + 1]
        closure.update(adj_sym.indices[start:end].tolist())
    closure_arr = np.array(sorted(closure))
    print(f"\n1-hop closure size: {len(closure_arr)}")

    # ----- Metrics -----
    closure_ids = set(mapp[i] for i in closure_arr)
    tp_in_closure = len(closure_ids & GT_mal)
    fp_in_closure = len(closure_ids - GT_mal)
    print(f"  TP in closure: {tp_in_closure}")
    print(f"  FP in closure: {fp_in_closure}")
    print(f"  Precision    : {tp_in_closure/(tp_in_closure+fp_in_closure):.4f}")
    print(f"  Recall (vs GT_mal): {tp_in_closure/len(GT_mal):.4f}")
    print(f"  Compression  : {len(closure_arr)}/{n_nodes} = {len(closure_arr)/n_nodes*100:.4f}%")

    # ----- Build subgraph + visualize -----
    closure_set = set(closure_arr.tolist())
    edges_array = np.array(edges)
    edge_times = np.array(edges_attr)
    mask = np.array([(s in closure_set and t in closure_set)
                     for s, t in zip(edges_array[0], edges_array[1])])
    sub_edges = edges_array[:, mask]
    sub_times = edge_times[mask]
    print(f"  Subgraph edges: {sub_edges.shape[1]}")

    # NetworkX
    G = nx.DiGraph()
    for i in closure_arr:
        G.add_node(int(i))
    for k in range(sub_edges.shape[1]):
        G.add_edge(int(sub_edges[0, k]), int(sub_edges[1, k]), time=float(sub_times[k]))
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Layout — spring (force-directed) gives meaningful clustering
    print("\nComputing layout...")
    pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)

    # ----- Visualization -----
    fig, ax = plt.subplots(figsize=(16, 12), dpi=120)

    tp_anchor_set = set(int(i) for i in tp_anchors_idx)
    gt_mal_set = set(int(i) for i in closure_arr if mapp[i] in GT_mal)

    # Node coloring
    node_colors = []
    node_sizes = []
    for n in G.nodes():
        if n in tp_anchor_set:
            node_colors.append("#e63946")     # red — TP anchor
            node_sizes.append(400)
        elif n in gt_mal_set:
            node_colors.append("#f4a261")     # orange — GT_mal in closure
            node_sizes.append(80)
        else:
            node_colors.append("#adb5bd")     # gray — benign in closure
            node_sizes.append(30)

    # Edge coloring by timestamp (older = lighter, newer = darker)
    if sub_times.size > 0:
        t_min, t_max = sub_times.min(), sub_times.max()
        edge_alphas = (sub_times - t_min) / (t_max - t_min + 1e-9) * 0.6 + 0.2
    else:
        edge_alphas = []

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.15, edge_color="#495057",
                            arrows=False, width=0.5)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                            node_size=node_sizes, edgecolors="#212529", linewidths=0.5)

    # Highlight TP-anchors with a star marker
    tp_pos = {n: pos[n] for n in tp_anchor_set if n in pos}
    if tp_pos:
        xs = [p[0] for p in tp_pos.values()]
        ys = [p[1] for p in tp_pos.values()]
        ax.scatter(xs, ys, s=600, marker="*", color="#e63946",
                   edgecolors="#000", linewidths=1.0, zorder=5, label="TP anchor")

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='#e63946',
                   markersize=18, markeredgecolor='black', label=f'TP anchor (n={len(tp_anchor_set)})'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#f4a261',
                   markersize=10, markeredgecolor='black', label=f'GT_mal in neighborhood (n={tp_in_closure - len(tp_anchor_set)})'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#adb5bd',
                   markersize=8, markeredgecolor='black', label=f'Benign neighbor (n={fp_in_closure})'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.95)

    # Title + metrics
    ax.set_title(
        f"C3 — 1-hop Attack Neighborhood around STGAN's TP anchors\n"
        f"DARPA TRACE | {len(closure_arr)} nodes ({len(closure_arr)/n_nodes*100:.4f}% of graph) | "
        f"Precision = {tp_in_closure/(tp_in_closure+fp_in_closure):.3f}",
        fontsize=13)
    ax.axis("off")

    out_dir = "c3_outputs_visualization"
    os.makedirs(out_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/attack_neighborhood.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{out_dir}/attack_neighborhood.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"\n[SAVED] {out_dir}/attack_neighborhood.png")
    print(f"[SAVED] {out_dir}/attack_neighborhood.pdf  ← format vectoriel pour LaTeX")

    # Save metrics
    metrics = {
        "tp_anchors_count": len(tp_anchors_idx),
        "tp_anchors_ids": [mapp[i] for i in tp_anchors_idx],
        "closure_size": len(closure_arr),
        "tp_in_closure": tp_in_closure,
        "fp_in_closure": fp_in_closure,
        "precision": tp_in_closure/(tp_in_closure+fp_in_closure),
        "recall_vs_full_gt": tp_in_closure/len(GT_mal),
        "compression_vs_graph": len(closure_arr)/n_nodes,
        "subgraph_edges": int(sub_edges.shape[1]),
    }
    with open(f"{out_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[SAVED] {out_dir}/metrics.json")
    print("\ndone")


if __name__ == "__main__":
    main()
