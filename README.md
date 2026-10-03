# ProvSAGE — A Lightweight Spatio-Temporal GNN for Provenance-Based Host Intrusion Detection

**Engineering Thesis Project (PFE Ingénieur 2025-2026)**

| | |
|---|---|
| **Author** | Amira Farah ABDELLI |
| **Specialty** | Cyber Security |
| **Institution** | École Supérieure en Sciences et Technologies de l'Informatique et du Numérique (ESTIN), Algeria |
| **Supervisors** | Dr. Samiha AIT TALEB, Mr. Nasreddine ABDELLI (ESTIN) |
| **Academic year** | 2025–2026 |

📄 Full dissertation: [`docs/memoire.pdf`](docs/memoire.pdf)

Advanced Persistent Threats (APTs) on enterprise endpoints unfold as long causal chains of system operations that elude signature- and rule-based detectors. Reasoning over system provenance graphs — which capture the causal history of all system interactions — has emerged as the principled response. **STGAN** [Sang et al., WWW 2025] is a recent learning-based detector built on this paradigm, fusing structural, semantic and temporal features to reach state-of-the-art performance on DARPA TRACE. Its published ablation isolates each module's contribution but does not extend the comparison to semantic embeddings beyond Word2Vec, to neighborhood-aggregation backbones such as GraphSAGE, or to non-attention fusion modules.

This project introduces **ProvSAGE**, a lightweight variant of STGAN derived from three empirical contributions on DARPA TRACE:

- **Contribution 1 (C1) — Semantic embedding study.** Systematic comparison of four semantic embedding alternatives: Word2Vec (baseline), FastText, BERT (sentence-transformers MiniLM), and EmbeddingGemma (raw and task-prompted variants).
- **Contribution 2 (C2) — Architectural complexity ablation.** Systematic ablation of the GNN backbone (GAT vs GraphSAGE vs GCN) and the fusion module (Multi-Head Attention vs Pooling vs Averaging).
- **Contribution 3 (C3) — Post-detection investigation module.** A module that compresses STGAN's raw flag set into a compact attack-neighborhood subgraph by combining entity-type filtering, multi-window temporal scoping, topological 1-hop expansion, and compactness-based ranking — without consulting ground-truth labels.

The recommended ProvSAGE configuration (GraphSAGE + MHA) improves detection accuracy over STGAN while substantially reducing both parameter count and training time. All experiments are run with **paper-faithful** sampling (`num_neighbors=[-1, -1]`, `batch_size=26000`) and deterministic settings (`seed=42`, `cudnn.deterministic=True`), ensuring reproducibility.

---

## 2. Key results

### Reproduction

| Setting | F1 | Notes |
|---|---|---|
| Paper claim | 0.9916 | From the original STGAN paper |
| **This work — baseline** | **0.9875** | W2V + GAT + MHA, deterministic, from scratch |
| Paper weights eval | 0.5869 | Paper's released weights with our own W2V — precision collapses due to W2V mismatch (the paper does not publish the Word2Vec model). |

### C1 — Embedding study (with GAT + MHA, full sampling)

| Embedding | F1 | Precision | FP |
|---|---|---|---|
| **Word2Vec** (baseline) | **0.9875** | 0.9866 | ~915 |
| BERT (MiniLM-L6-v2) | 0.9769 | 0.9657 | 2 391 |
| EmbeddingGemma prompted | 0.9738 | 0.9596 | 2 834 |
| FastText | 0.9725 | 0.9572 | 3 015 |
| EmbeddingGemma raw | 0.9408 | 0.8976 | 7 685 |

**Key finding.** At full sampling with GAT, simpler embeddings outperform richer ones. This inverts the order observed under reduced sampling on a smaller GPU, indicating a structural interaction between embedding richness and sampling completeness.

### C2 — Architectural ablation (with FastText, full sampling)

| GNN | Fusion | F1 | Parameters | Train time |
|---|---|---|---|---|
| **GraphSAGE** | **MHA** | **0.9932** | 387 938 | 33 min |
| GAT | MHA (baseline) | 0.9725 | 452 894 | 2 h 30 |
| GCN | MHA | 0.9540 | 381 154 | 1 h 42 |
| GAT | Averaging | 0.9292 | 448 670 | 1 h 18 |
| GAT | Pooling | 0.9254 | 448 670 | 1 h 04 |

**Cross-check.** W2V + GraphSAGE + MHA reaches **F1 = 0.9933** (new top), confirming that GraphSAGE + MHA is invariant to the embedding choice.

**Key finding.** GraphSAGE + MHA is the recommended architecture: better F1, 14 % fewer parameters, and 5× faster training than the GAT-based baseline.

### C3 — Investigation exploratory

STGAN's 1 248 flag anchors are **graph-isolated** (no anchor shares an edge with any other anchor), invalidating topology-only investigation strategies. The 1-hop neighborhood around the seven confirmed true-positive anchors reveals a dichotomy: six sparse kill-chain entities + one hub-connected anchor (figure available at `results/c3_outputs_visualization/attack_neighborhood.pdf`).

---

## 3. Repository structure

```
provsage-host-intrusion-detection/
├── README.md
├── requirements.txt
├── overview.png
│
├── docs/
│   └── memoire.pdf                   Full engineering thesis (dissertation)
│
├── code/
│   ├── train/                        Training scripts
│   ├── eval/                         Evaluation scripts (incl. C3 visualization)
│   ├── model/                        GAT, TGN, MHA, PositionalEncoder, models_c2
│   └── utils/                        Determinism, embeddings, helpers
│
├── notebooks/                        Reproduction notebooks (in order)
│   ├── 00_setup_and_data.ipynb       Environment + dataset download
│   ├── 01_baseline_reproduction.ipynb   Paper weights eval + W2V deterministic baseline
│   ├── 02_C1_embeddings.ipynb        BERT + Gemma raw + Gemma prompted
│   ├── 03_C2_architectures.ipynb     Full C2 ablation + W2V cross-check
│   └── 04_C3_investigation.ipynb     C3 visualization
│
├── trained_weights/
│   └── trace/                        50-epoch checkpoints for all 10 configurations
│       ├── baseline_w2v_gat_mha/
│       ├── c1_bert/
│       ├── c1_gemma/
│       ├── c1_gemma_prompted/
│       ├── c2_gat_mha/               (= C1 FastText)
│       ├── c2_graphsage_mha/         C2 winner
│       ├── c2_gcn_mha/
│       ├── c2_gat_pooling/
│       ├── c2_gat_averaging/
│       └── c2_w2v_graphsage_mha/     C1 × C2 cross-check (top absolute)
│
├── embedding_models/
│   ├── word2vec_trace.model          Deterministic Word2Vec (workers=1, seed=42)
│   └── fasttext_trace.model          Deterministic FastText
│
└── results/
    └── c3_outputs_visualization/
        ├── attack_neighborhood.png
        ├── attack_neighborhood.pdf   Vector format for LaTeX inclusion
        └── metrics.json
```
> **Note.** The `trained_weights/` directory (checkpoints for all 10 configurations, ~850 MB) is not included in this archive to comply with the 100 MB submission size limit. Weights are available upon request.
---

## 4. Quick start

### Hardware requirements

- NVIDIA A100 GPU (or compatible CUDA device, ≥ 20 GB VRAM)
- 250 GB RAM (the test set graph + embeddings can occupy 30 GB)
- 100 GB free disk

### Software setup

```bash
# 1. Clone the repository and change directory
git clone https://github.com/<ton-identifiant-github>/provsage-host-intrusion-detection.git
cd provsage-host-intrusion-detection

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Set environment variables for gated resources
export KAGGLE_API_TOKEN="<your-kaggle-token>"
export HF_TOKEN="<your-huggingface-token>"   # required to access google/embeddinggemma-300m
```

### Reproduction order

Run the notebooks in the following order:

1. **`notebooks/00_setup_and_data.ipynb`** — environment check, package install, dataset download from Kaggle, data preprocessing (~10 min plus download).
2. **`notebooks/01_baseline_reproduction.ipynb`** — paper weights evaluation + W2V deterministic baseline training and evaluation (~5 hours).
3. **`notebooks/02_C1_embeddings.ipynb`** — BERT, EmbeddingGemma raw, EmbeddingGemma prompted (~3 hours total).
4. **`notebooks/03_C2_architectures.ipynb`** — FastText baseline + GraphSAGE + GCN + Pooling + Averaging + W2V cross-check (~6 hours).
5. **`notebooks/04_C3_investigation.ipynb`** — C3 visualization of the attack neighborhood (~5 minutes).

Total reproduction time: ~15 hours on a single A100 (20 GB MIG slice).

### Skipping training (evaluation only)

All trained weights are provided under `trained_weights/trace/`. To skip training and only run evaluation, jump to the `eval_trace_*.py` cells in each notebook and read the weights directly.

---

## 5. Notes on reproducibility

- All Word2Vec and FastText models are trained with `workers=1` and `seed=42` for full determinism, at the cost of slower training.
- PyTorch is configured with `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False`. `torch.use_deterministic_algorithms(True)` is *not* set because it conflicts with PyG scatter operations on A100 MIG slices.
- The Word2Vec model in `embedding_models/` is the one used by `baseline_w2v_gat_mha` and `c2_w2v_graphsage_mha`. The FastText model is shared by all six configurations using FastText embeddings.
- Embedding caches for BERT and EmbeddingGemma are **not** included in this repository (they are several gigabytes each). They are regenerated automatically on first run via the cache mechanism in each training script.

---

## 6. Citation

If you use this code in academic work, please cite:

```
Abdelli, A. F. (2026). ProvSAGE: A Lightweight Spatio-Temporal GNN for
Provenance-Based Host Intrusion Detection. PFE Ingénieur, École Supérieure
en Sciences et Technologies de l'Informatique et du Numérique (ESTIN),
Algeria.
```

And the original STGAN paper:

```
Sang, A., Fan, X., Yang, L., Wang, Y., Zhou, L., Jia, J., Yang, H. (2025).
STGAN: Detecting Host Threats via Fusion of Spatial-Temporal Features in
Host Provenance Graphs. In Proceedings of the ACM Web Conference 2025 (WWW),
pp. 1046–1057. https://doi.org/10.1145/3696410.3714925
```

---

## 7. Acknowledgements

- Supervisors: Dr. Samiha AIT TALEB and Mr. Nasreddine ABDELLI (ESTIN)
- Jury: Dr. Lydia BAZIZI (President), Dr. Layachi DJEDOUI (Examiner), Ms. Amal ADRAR (Examiner)
- Computing resources: ESTIN A100 MIG 20 GB compute slice (24/7 SSH access)
- Dataset: DARPA Transparent Computing programme, Engagement 3 — Trace scenario

---

## 8. License

The original contributions of this thesis project (C1/C2/C3 code, notebooks, documentation, and results) are released under the [MIT License](LICENSE). The STGAN baseline method and architecture this work reproduces are from Sang et al., WWW 2025 — see the citation above.
