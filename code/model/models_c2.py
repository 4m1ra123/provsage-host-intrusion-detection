"""Composants alternatifs pour la contribution 2 (réduction de complexité)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, BatchNorm


class GraphSAGE(torch.nn.Module):
    """Alternative légère au GAT : agrégation de voisinage sans attention."""
    def __init__(self, in_channel, out_channel, hidden_channels=64,
                 num_layers=3, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channel, hidden_channels))
        self.bns.append(BatchNorm(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(BatchNorm(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channel))
        self.bns.append(BatchNorm(out_channel))

    def forward(self, x, edge_index):
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.elu(x)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)


class GCN(torch.nn.Module):
    """GCN — convolution de graphe sans attention."""
    def __init__(self, in_channel, out_channel, hidden_channels=64,
                 num_layers=3, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channel, hidden_channels))
        self.bns.append(BatchNorm(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(BatchNorm(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channel))
        self.bns.append(BatchNorm(out_channel))

    def forward(self, x, edge_index):
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.elu(x)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)


class PoolingFusionModel(nn.Module):
    """Remplace la MHA : Linear + ReLU + max-pooling + Linear."""
    def __init__(self, in_channels, out_channels, num_classes, num_heads=4):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.fc2 = nn.Linear(out_channels, num_classes)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = x.max(dim=1)[0]
        return self.fc2(x)


class AveragingFusionModel(nn.Module):
    """Remplace la MHA : Linear + ReLU + moyenne + Linear."""
    def __init__(self, in_channels, out_channels, num_classes, num_heads=4):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.fc2 = nn.Linear(out_channels, num_classes)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = x.mean(dim=1)
        return self.fc2(x)


def build_gnn(name, in_channel, out_channel, device):
    name = name.lower()
    if name == "gat":
        from model.GAT import GAT
        return GAT(in_channel=in_channel, out_channel=out_channel).to(device)
    if name == "graphsage":
        return GraphSAGE(in_channel=in_channel, out_channel=out_channel).to(device)
    if name == "gcn":
        return GCN(in_channel=in_channel, out_channel=out_channel).to(device)
    raise ValueError(f"GNN inconnu : {name}")


def build_fusion(name, in_channels, out_channels, num_classes, device, num_heads=4):
    name = name.lower()
    if name == "mha":
        from model.MultiHeadSelfAttention import MultiHeadSelfAttentionModel
        return MultiHeadSelfAttentionModel(
            in_channels=in_channels, out_channels=out_channels,
            num_classes=num_classes, num_heads=num_heads).to(device)
    if name == "pooling":
        return PoolingFusionModel(in_channels, out_channels, num_classes).to(device)
    if name == "averaging":
        return AveragingFusionModel(in_channels, out_channels, num_classes).to(device)
    raise ValueError(f"Fusion inconnue : {name}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
