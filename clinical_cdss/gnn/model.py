from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F


class HypergraphAttentionLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.node_score = nn.Linear(hidden_dim, 1)
        self.edge_score = nn.Linear(hidden_dim, 1)
        self.node_update = nn.Linear(hidden_dim * 2, hidden_dim)
        self.edge_update = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_x: torch.Tensor, incidence: torch.Tensor):
        mask = incidence > 0

        node_logits = self.node_score(node_x).squeeze(-1).unsqueeze(1)
        node_logits = node_logits.masked_fill(~mask, -1e9)
        alpha = torch.softmax(node_logits, dim=0).masked_fill(~mask, 0.0)

        edge_x = alpha.transpose(0, 1) @ node_x
        edge_x = self.dropout(edge_x)

        edge_logits = self.edge_score(edge_x).squeeze(-1).unsqueeze(0)
        edge_logits = edge_logits.masked_fill(~mask, -1e9)
        beta = torch.softmax(edge_logits, dim=1).masked_fill(~mask, 0.0)

        edge_messages = beta @ edge_x
        node_x = torch.cat([node_x, edge_messages], dim=-1)
        node_x = F.elu(self.node_update(node_x))

        node_messages = alpha.transpose(0, 1) @ node_x
        edge_x = torch.cat([edge_x, node_messages], dim=-1)
        edge_x = F.elu(self.edge_update(edge_x))

        return node_x, edge_x, alpha, beta


class ClinicalHGAT(nn.Module):
    def __init__(
        self,
        in_dim_dict: Dict[str, int],
        hidden: int = 64,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.node_types = ["patient", "evidence", "symptom", "concept", "rule"]
        self.encoders = nn.ModuleDict({
            node_type: nn.Sequential(
                nn.Linear(in_dim_dict[node_type], hidden),
                nn.LayerNorm(hidden),
                nn.ELU(),
                nn.Dropout(dropout),
            )
            for node_type in self.node_types
        })
        self.layer1 = HypergraphAttentionLayer(hidden, dropout=dropout)
        self.layer2 = HypergraphAttentionLayer(hidden, dropout=dropout * 0.7)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes),
        )

    def encode_nodes(self, x_dict, node_type_indices, total_nodes: int):
        device = next(self.parameters()).device
        hidden = next(iter(self.encoders.values()))[0].out_features
        node_x = torch.zeros((total_nodes, hidden), dtype=torch.float32, device=device)
        for node_type in self.node_types:
            x = x_dict[node_type].to(device)
            idx = node_type_indices[node_type].to(device)
            if idx.numel() == 0:
                continue
            node_x[idx] = self.encoders[node_type](x)
        return node_x

    def forward(self, data):
        device = next(self.parameters()).device
        incidence = data.incidence.to(device)
        node_x = self.encode_nodes(
            data.x_dict,
            data.node_type_indices,
            total_nodes=incidence.shape[0],
        )
        node_x, _, alpha1, beta1 = self.layer1(node_x, incidence)
        node_x, edge_x, alpha2, beta2 = self.layer2(node_x, incidence)
        patient_x = node_x[data.patient_node_indices.to(device)]
        logits = self.classifier(patient_x)
        return {
            "logits": logits,
            "node_embeddings": node_x,
            "edge_embeddings": edge_x,
            "attention_alpha": alpha2,
            "attention_beta": beta2,
            "attention_alpha_layer1": alpha1,
            "attention_beta_layer1": beta1,
        }
