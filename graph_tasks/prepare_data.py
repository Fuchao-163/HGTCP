#!/usr/bin/env python3
"""Download public graph benchmarks and build HGTCP structural caches.

No dataset is distributed with this repository. PyG downloads ZINC or MNIST
to ``data/raw`` and this script writes derived tensors to ``data/cache``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import networkx as nx
import pymetis
import torch
from torch_geometric.datasets import GNNBenchmarkDataset, ZINC

ROOT = Path(__file__).resolve().parents[1]


def partition_graph(graph: nx.Graph, parts: int) -> list[int]:
    nodes = list(graph.nodes())
    positions = {node: index for index, node in enumerate(nodes)}
    adjacency = [
        [positions[neighbor] for neighbor in graph.neighbors(node)]
        for node in nodes
    ]
    options = pymetis.Options(seed=0)
    _, membership = pymetis.part_graph(
        parts, adjacency=adjacency, recursive=True, options=options
    )
    result = [0] * len(nodes)
    for node, cluster in zip(nodes, membership):
        result[int(node)] = int(cluster)
    return result


def load_splits(dataset: str, data_root: Path):
    if dataset == "zinc":
        return {
            split: ZINC(data_root / "ZINC", subset=True, split=split)
            for split in ("train", "val", "test")
        }
    if dataset == "mnist":
        return {
            split: GNNBenchmarkDataset(
                data_root / "GNNBenchmark", "MNIST", split=split
            )
            for split in ("train", "val", "test")
        }
    raise ValueError(f"Unsupported dataset: {dataset}")


def hgtcp_transform(data, clusters: int):
    data = data.clone()
    graph = nx.Graph()
    graph.add_nodes_from(range(data.num_nodes))
    graph.add_edges_from(data.edge_index.t().tolist())

    if data.num_nodes <= clusters:
        membership = list(range(data.num_nodes))
    else:
        membership = partition_graph(graph, clusters)
    membership = torch.as_tensor(membership, dtype=torch.long)

    coarse = nx.Graph()
    coarse.add_nodes_from(range(clusters))
    for source, target in graph.edges():
        source_cluster = int(membership[source])
        target_cluster = int(membership[target])
        if source_cluster != target_cluster:
            coarse.add_edge(source_cluster, target_cluster)

    distances = torch.full((data.num_nodes, clusters), 30, dtype=torch.long)
    coarse_distances = dict(nx.all_pairs_shortest_path_length(coarse))
    for node in range(data.num_nodes):
        source_cluster = int(membership[node])
        for target_cluster, distance in coarse_distances[source_cluster].items():
            distances[node, target_cluster] = min(int(distance), 30)

    data.hgtcp_membership = membership
    data.hgtcp_distance = distances
    return data


def cache_split(items, output: Path, clusters: int) -> None:
    if output.exists():
        print(f"[skip] {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    processed = []
    for index, item in enumerate(items, start=1):
        processed.append(hgtcp_transform(item, clusters))
        if index % 1000 == 0 or index == len(items):
            print(f"[{output.stem}] {index}/{len(items)}", flush=True)
    temporary = output.with_suffix(".pt.tmp")
    torch.save(processed, temporary)
    os.replace(temporary, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["zinc", "mnist"])
    parser.add_argument("--clusters", type=int, default=8)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--cache-root", type=Path, default=ROOT / "data" / "cache")
    args = parser.parse_args()

    expected_sizes = {
        "zinc": {"train": 10000, "val": 1000, "test": 1000},
        "mnist": {"train": 55000, "val": 5000, "test": 10000},
    }
    for dataset in args.datasets:
        splits = load_splits(dataset, args.data_root)
        sizes = {split: len(items) for split, items in splits.items()}
        if sizes != expected_sizes[dataset]:
            raise RuntimeError(f"Unexpected {dataset} split sizes: {sizes}")
        for split, items in splits.items():
            cache_split(
                items,
                args.cache_root / dataset / f"{split}.pt",
                args.clusters,
            )


if __name__ == "__main__":
    main()
