"""Small python-metis compatible shim backed by pymetis.

The released Cluster-GT code imports ``metis.part_graph``.  The server does not
ship libmetis/python-metis, while pymetis is available and implements the same
partitioning algorithm.  Only the API used by Cluster-GT is provided here.
"""

from __future__ import annotations

import pymetis


def part_graph(graph, nparts, recursive=False, **_kwargs):
    nodes = list(graph.nodes())
    node_to_pos = {node: i for i, node in enumerate(nodes)}
    adjacency = [
        [node_to_pos[nbr] for nbr in graph.neighbors(node)] for node in nodes
    ]
    options = pymetis.Options(seed=0)
    edgecuts, membership = pymetis.part_graph(
        nparts,
        adjacency=adjacency,
        recursive=recursive,
        options=options,
    )
    return edgecuts, list(membership)
