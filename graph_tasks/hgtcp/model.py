"""Faithful graph-level HGTCP adaptation with contrastive regularization.

This version follows the draft equations and the repository's node-level
implementation: an edge-aware local encoder, an input projection for the
global view, bottom-up attention pooling, HPDE-biased top-down attention,
per-stage feed-forward transforms, local/global interpolation, and the
local-global discriminator used by the contrastive objective.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool
from torch_geometric.utils import softmax
from torch_scatter import scatter


def _normalization(kind: str, hidden: int) -> nn.Module:
    if kind == "batch":
        return nn.BatchNorm1d(hidden)
    if kind == "layer":
        return nn.LayerNorm(hidden)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"Unsupported normalization: {kind}")


class _FeedForward(nn.Module):
    def __init__(self, hidden: int, dropout: float, normalization: str,
                 depth: int = 2):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.extend([
                nn.Linear(hidden, hidden), _normalization(normalization, hidden),
                nn.ReLU(), nn.Dropout(dropout),
            ])
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class _LocalGINE(nn.Module):
    def __init__(self, hidden: int, dataset: str, dropout: float,
                 normalization: str, train_eps: bool):
        super().__init__()
        self.dataset = dataset
        self.edge_encoder = (nn.Embedding(4, hidden) if dataset == "zinc"
                             else nn.Linear(1, hidden))
        self.conv = GINEConv(
            nn.Sequential(
                nn.Linear(hidden, hidden * 2), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden * 2, hidden),
            ),
            train_eps=train_eps,
        )
        self.norm = _normalization(normalization, hidden)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr):
        if self.dataset == "zinc":
            edge = self.edge_encoder(edge_attr.view(-1).long())
        else:
            edge = self.edge_encoder(edge_attr.float().view(-1, 1))
        update = self.conv(x, edge_index, edge)
        return self.norm(x + F.dropout(update, self.dropout, self.training))


class _DistanceEncoder(nn.Module):
    """Original HGTCP categorical HPDE: embedding followed by scalar map."""

    def __init__(self, max_distance: int, hidden: int, heads: int):
        super().__init__()
        self.embedding = nn.Embedding(max_distance + 1, hidden)
        self.scalar = nn.Linear(hidden, heads)

    def forward(self, distances):
        return self.scalar(self.embedding(distances))


class HGTCPV2Block(nn.Module):
    def __init__(self, hidden: int, heads: int, clusters: int,
                 max_distance: int, dropout: float, normalization: str,
                 ffn_depth: int, use_cluster_size_bias: bool):
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        self.hidden = hidden
        self.heads = heads
        self.clusters = clusters
        self.head_dim = hidden // heads
        self.dropout = dropout
        self.use_cluster_size_bias = use_cluster_size_bias

        # Bottom-up inter-level attention, matching VerticalTransformerBlock.
        self.vertical_projection = nn.Linear(hidden, hidden)
        self.vertical_query = nn.Linear(hidden, hidden)
        self.vertical_key = nn.Linear(hidden, hidden)
        self.vertical_value = nn.Linear(hidden, hidden)
        self.vertical_ffn = _FeedForward(
            hidden, dropout, normalization, ffn_depth
        )

        # Top-down global backtracking with HPDE.
        self.global_projection = nn.Linear(hidden, hidden)
        self.global_query = nn.Linear(hidden, hidden)
        self.global_key = nn.Linear(hidden, hidden)
        self.global_value = nn.Linear(hidden, hidden)
        self.hpde = _DistanceEncoder(max_distance, hidden, heads)
        self.global_ffn = _FeedForward(
            hidden, dropout, normalization, ffn_depth
        )

    def forward(self, x, batch, membership, distances, num_graphs):
        total_clusters = num_graphs * self.clusters
        group = batch * self.clusters + membership
        counts = scatter(
            x.new_ones(x.size(0)), group, dim=0,
            dim_size=total_clusters, reduce="sum",
        )
        means = scatter(
            x, group, dim=0, dim_size=total_clusters, reduce="sum"
        ) / counts.clamp_min(1).unsqueeze(-1)

        projected = self.vertical_projection(x)
        q = self.vertical_query(means).view(
            total_clusters, self.heads, self.head_dim
        )
        k = self.vertical_key(projected).view(
            x.size(0), self.heads, self.head_dim
        )
        v = self.vertical_value(projected).view(
            x.size(0), self.heads, self.head_dim
        )
        vertical_scores = (q[group] * k).sum(-1) / math.sqrt(self.head_dim)
        vertical_attention = softmax(
            vertical_scores, group, num_nodes=total_clusters
        )
        vertical_attention = F.dropout(
            vertical_attention, self.dropout, self.training
        )
        communities = scatter(
            vertical_attention.unsqueeze(-1) * v, group, dim=0,
            dim_size=total_clusters, reduce="sum",
        ).reshape(total_clusters, self.hidden)
        communities = self.vertical_ffn(communities).view(
            num_graphs, self.clusters, self.hidden
        )

        projected_nodes = self.global_projection(x)
        q = self.global_query(projected_nodes).view(
            x.size(0), self.heads, self.head_dim
        )
        k = self.global_key(communities).view(
            num_graphs, self.clusters, self.heads, self.head_dim
        )
        v = self.global_value(communities).view(
            num_graphs, self.clusters, self.heads, self.head_dim
        )
        scores = torch.einsum(
            "nhd,nchd->nhc", q, k[batch]
        ) / math.sqrt(self.head_dim)
        scores = scores + self.hpde(distances).permute(0, 2, 1)
        graph_counts = counts.view(num_graphs, self.clusters)
        if self.use_cluster_size_bias:
            scores = scores + graph_counts[batch].clamp_min(1).log().unsqueeze(1)
        scores = scores.masked_fill(
            graph_counts[batch].eq(0).unsqueeze(1), -torch.inf
        )
        attention = F.softmax(scores, dim=-1)
        attention = F.dropout(attention, self.dropout, self.training)
        global_x = torch.einsum(
            "nhc,nchd->nhd", attention, v[batch]
        ).reshape_as(x)
        return self.global_ffn(global_x)


class HGTCPV2Graph(nn.Module):
    def __init__(self, dataset: str, hidden: int, global_layers: int,
                 local_layers: int, heads: int, clusters: int, alpha: float,
                 dropout: float, out_dim: int, graph_pooling: str = "add",
                 normalization: str = "batch", ffn_depth: int = 2,
                 train_eps: bool = True, hpde_level_offset: int = 1,
                 use_cluster_size_bias: bool = True):
        super().__init__()
        self.dataset = dataset.lower()
        self.clusters = clusters
        self.alpha = alpha
        self.hpde_level_offset = hpde_level_offset
        if graph_pooling not in {"mean", "add"}:
            raise ValueError(f"Unsupported graph pooling: {graph_pooling}")
        self.graph_pooling = graph_pooling
        self.node_encoder = (nn.Embedding(28, hidden) if self.dataset == "zinc"
                             else nn.Linear(3, hidden))
        self.local_layers = nn.ModuleList(
            _LocalGINE(
                hidden, self.dataset, dropout, normalization, train_eps
            ) for _ in range(local_layers)
        )
        self.global_input = _FeedForward(
            hidden, dropout, normalization, depth=1
        )
        self.global_layers = nn.ModuleList(
            HGTCPV2Block(
                hidden, heads, clusters, 30, dropout, normalization,
                ffn_depth, use_cluster_size_bias,
            ) for _ in range(global_layers)
        )
        self.discriminator = nn.Bilinear(hidden, hidden, 1)
        nn.init.xavier_uniform_(self.discriminator.weight)
        if self.discriminator.bias is not None:
            nn.init.zeros_(self.discriminator.bias)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    @staticmethod
    def _within_graph_negative(global_x, ptr):
        negative = torch.empty_like(global_x)
        for graph_index in range(ptr.numel() - 1):
            start = int(ptr[graph_index])
            end = int(ptr[graph_index + 1])
            negative[start:end] = global_x[start:end].roll(1, dims=0)
        return negative

    def forward(self, data):
        if getattr(data, "edge_attr", None) is None:
            raise ValueError("HGTCP v2 requires dataset edge attributes")
        if self.dataset == "zinc":
            x = self.node_encoder(data.x.view(-1).long())
        else:
            if getattr(data, "pos", None) is None:
                raise ValueError("MNIST data must provide 2-D node positions")
            features = torch.cat((data.x.float(), data.pos.float()), dim=-1)
            if features.size(-1) != 3:
                raise ValueError(
                    f"Expected 3 MNIST node features, got {features.size(-1)}"
                )
            x = self.node_encoder(features)
        for layer in self.local_layers:
            x = layer(x, data.edge_index, data.edge_attr)
        local_x = x

        global_x = self.global_input(local_x)
        # Cached values are coarse-level SPD.  Eq. (10) defines HPD as
        # coarse SPD plus the number of traversed hierarchy levels.
        distances = (data.hgtcp_distance.long() + self.hpde_level_offset).clamp_max(30)
        for layer in self.global_layers:
            global_x = layer(
                global_x, data.batch, data.hgtcp_membership.long(),
                distances, data.num_graphs,
            )

        negative_global = self._within_graph_negative(global_x, data.ptr)
        positive_logits = self.discriminator(local_x, global_x)
        negative_logits = self.discriminator(local_x, negative_global)
        contrastive_logits = torch.cat((positive_logits, negative_logits), dim=0)

        x = self.alpha * local_x + (1.0 - self.alpha) * global_x
        graph_x = (global_add_pool(x, data.batch) if self.graph_pooling == "add"
                   else global_mean_pool(x, data.batch))
        return self.head(graph_x), contrastive_logits
