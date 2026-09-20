# Paper hyperparameters

These are the HGTCP settings reported in Appendix C, Table 2 of the manuscript.
The node-task launch scripts encode the same values.

| Dataset | LR | Hidden | Centroids | GNN layers | GT layers | Heads | alpha | lambda |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Cora | 0.01 | 128 | 128 | 2 | 1 | 4 | 0.7 | 0.1 |
| Citeseer | 0.01 | 128 | 64 | 2 | 2 | 2 | 0.7 | 0.05 |
| Pubmed | 0.01 | 128 | 128 | 2 | 1 | 1 | 0.7 | 0.05 |
| Amazon-Photo | 0.01 | 128 | 64 | 2 | 2 | 1 | 0.7 | 0.05 |
| Actor | 0.01 | 128 | 128 | 2 | 2 | 2 | 0.5 | 0 |
| Squirrel | 0.01 | 128 | 256 | 3 | 3 | 1 | 0.3 | 1 |
| Chameleon | 0.01 | 256 | 256 | 3 | 3 | 1 | 0.3 | 0.7 |
| Flickr | 0.0005 | 128 | 256 | 4 | 1 | 1 | 0.3 | 0.5 |
| ogbn-proteins | 0.0005 | 128 | 1024 | 4 | 1 | 2 | 0.7 | 0.05 |
| ogbn-arxiv | 0.0005 | 256 | 1024 | 3 | 1 | 1 | 0.5 | 0.05 |
| Reddit | 0.0005 | 128 | 1024 | 3 | 1 | 1 | 0.5 | 0.05 |
| ogbn-products | 0.0005 | 256 | 1024 | 5 | 1 | 4 | 0.5 | 0.05 |

The graph-task JSON files record the validation-selected settings used for the
ZINC and MNIST extension. They are not part of the original 12-dataset node
classification table.
