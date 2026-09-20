#!/usr/bin/env python3
"""Data-free forward/backward smoke tests for both HGTCP entry points."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch_geometric.data import Batch, Data

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "graph_tasks"))
sys.path.insert(0, str(ROOT / "node_classification" / "medium"))

from hgtcp.model import HGTCPV2Graph  # noqa: E402
from model import Transformer  # noqa: E402


def smoke_node_model() -> None:
    nodes, features, classes, clusters = 20, 8, 3, 4
    source = torch.arange(nodes)
    edge_index = torch.stack((source, torch.roll(source, -1)))
    data = SimpleNamespace(
        graph={
            "node_feat": torch.randn(nodes, features),
            "edge_index": edge_index,
            "distance_matrix": torch.randint(0, 5, (nodes, clusters)),
            "nodes_to_community_tensor": torch.arange(nodes) % clusters,
        }
    )
    model = Transformer(
        in_channels=features,
        hidden_channels=16,
        out_channels=classes,
        global_dim=16,
        num_layers=1,
        heads=4,
        ff_dropout=0.1,
        attn_dropout=0.1,
        num_centroids=clusters,
        no_bn=False,
        norm_type="batch_norm",
        gnum_layers=2,
        ghidden_channels=16,
        gdropout=0.1,
        no_gnn=False,
        alpha=0.5,
    )
    output, contrastive = model(data)
    (output.sum() + contrastive.sum()).backward()
    assert output.shape == (nodes, classes)


def graph_example(dataset: str) -> Data:
    nodes = 12
    source = torch.arange(nodes)
    target = torch.roll(source, -1)
    edge_index = torch.stack(
        (torch.cat((source, target)), torch.cat((target, source)))
    )
    membership = torch.arange(nodes) % 8
    distances = torch.randint(0, 5, (nodes, 8))
    if dataset == "zinc":
        return Data(
            x=torch.randint(0, 28, (nodes, 1)),
            edge_index=edge_index,
            edge_attr=torch.randint(0, 4, (edge_index.size(1), 1)),
            y=torch.randn(1),
            hgtcp_membership=membership,
            hgtcp_distance=distances,
        )
    return Data(
        x=torch.rand(nodes, 1),
        pos=torch.rand(nodes, 2),
        edge_index=edge_index,
        edge_attr=torch.rand(edge_index.size(1), 1),
        y=torch.randint(0, 10, (1,)),
        hgtcp_membership=membership,
        hgtcp_distance=distances,
    )


def smoke_graph_models() -> None:
    for dataset in ("zinc", "mnist"):
        batch = Batch.from_data_list([graph_example(dataset), graph_example(dataset)])
        model = HGTCPV2Graph(
            dataset=dataset,
            hidden=16,
            global_layers=1,
            local_layers=1,
            heads=4,
            clusters=8,
            alpha=0.5,
            dropout=0.1,
            out_dim=1 if dataset == "zinc" else 10,
        )
        output, contrastive = model(batch)
        (output.sum() + contrastive.sum()).backward()
        assert output.shape == (2, 1 if dataset == "zinc" else 10)


if __name__ == "__main__":
    torch.manual_seed(0)
    smoke_node_model()
    smoke_graph_models()
    print("HGTCP smoke tests passed")
