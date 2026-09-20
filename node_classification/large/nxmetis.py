"""Subset of nxmetis used by HGTCP, implemented with pymetis."""

from __future__ import annotations

from metis import part_graph


def partition(graph, nparts, **kwargs):
    nodes = list(graph.nodes())
    edgecuts, membership = part_graph(graph, nparts, **kwargs)
    communities = [set() for _ in range(nparts)]
    for node, part in zip(nodes, membership):
        communities[int(part)].add(node)
    return edgecuts, communities
