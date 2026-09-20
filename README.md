# HGTCP

Official reproduction package for **HGTCP: Hierarchical Graph Transformer
with Cross-level Propagation**, accompanying the manuscript *Long-range
Dependencies in Large Graphs: A Scalable Hierarchical Architecture*.

HGTCP constructs a graph hierarchy, performs attention-based bottom-up local
fusion, and backtracks global information with Hierarchical Propagation
Distance Encoding (HPDE). A contrastive objective aligns local and hierarchical
global views.

## What is included

- `node_classification/medium`: Cora, Citeseer, Pubmed, Amazon-Photo, Actor,
  Squirrel, and Chameleon.
- `node_classification/large`: Flickr, Reddit, and ogbn-arxiv.
- `node_classification/ogbn_proteins`: ogbn-proteins.
- `node_classification/ogbn_products`: ogbn-products.
- `graph_tasks`: the validation-selected HGTCP graph-level extension for ZINC
  and MNIST.
- `configs/paper_hyperparameters.md`: the manuscript's per-dataset settings.
- `scripts`: reproducible launchers and result aggregation.

No data, preprocessed structural cache, checkpoint, embedding, or experiment
log is included.

## Environment

The cleaned package was checked with Python 3.9, PyTorch 2.4, PyG 2.6, and
CUDA 12.4. Create the environment with:

```bash
conda env create -f environment.yml
conda activate hgtcp
```

If the CUDA version differs, install the matching PyTorch, `torch-scatter`,
and `torch-sparse` wheels first, then run `pip install -r requirements.txt`.

## Data

Set one root for downloaded datasets:

```bash
export HGTCP_DATA_ROOT=/absolute/path/to/hgtcp-data
```

PyG/OGB datasets are downloaded automatically by their loaders. Actor,
Squirrel, and Chameleon use the benchmark splits described in the manuscript:

```text
$HGTCP_DATA_ROOT/
└── geom-gcn/
    ├── film/
    │   ├── out1_graph_edges.txt
    │   └── out1_node_feature_label.txt
    ├── splits/film_split_0.6_0.2_{0..9}.npz
    ├── chameleon/chameleon_filtered.npz
    └── squirrel/squirrel_filtered.npz
```

The filtered Chameleon/Squirrel files come from the heterophily evaluation
splits cited by the paper. Do not commit this data directory.

## Reproduce node-classification experiments

Run the seven medium-scale datasets:

```bash
HGTCP_DEVICE=0 bash scripts/run_medium_node_tasks.sh
```

Run large datasets individually (preprocessing may take substantial CPU time,
RAM, and disk):

```bash
bash scripts/run_large_node_task.sh flickr 0
bash scripts/run_large_node_task.sh reddit 0
bash scripts/run_large_node_task.sh ogbn-arxiv 0
bash scripts/run_large_node_task.sh ogbn-proteins 0
bash scripts/run_large_node_task.sh ogbn-products 0
```

All paper results are reported over five runs. The medium/proteins launchers
perform five runs directly. The other large-task commands should be scheduled
five times with separately recorded seeds; the preserved original large-graph
entry points expose their historical preprocessing and training behavior.

## Reproduce graph-level extension

ZINC uses MAE; MNIST uses accuracy. Public splits are downloaded by PyG and
preprocessed locally. The test split is evaluated only after selecting the
best epoch on validation data.

```bash
HGTCP_DEVICE=cuda:0 bash scripts/run_graph_tasks.sh
```

This writes per-seed JSON and checkpoints under `results/graph/`; all are
ignored by Git. The script prints the mean and sample standard deviation over
five seeds.

## Data-free release audit

Before publishing, run:

```bash
find . -type f -size +5M -print
find . -type f \
  \( -name '*.pt' -o -name '*.pth' -o -name '*.pkl' -o -name '*.npz' \
     -o -name '*.mat' -o -name '*.ckpt' \) -print
```

Both commands should produce no output from the tracked release tree.

## Reproducibility notes

- Dataset splits follow the manuscript: PyG public splits for
  Cora/Citeseer/Pubmed, OGB public splits for OGB datasets, predefined splits
  for Flickr/Reddit, Lim et al. splits for Actor, and filtered Platonov et al.
  splits for Squirrel/Chameleon.
- Amazon-Photo follows the experiment code's five deterministic 60/20/20
  random splits (base seed 42). Actor/Squirrel/Chameleon use the first five of
  their ten published split files; public-split datasets repeat training with
  five model seeds.
- Node classification reports accuracy, except ogbn-proteins, which reports
  ROC-AUC. ZINC reports MAE and graph MNIST reports accuracy.
- Structural partitions and HPDE matrices are generated on first use and can
  be reused across seeds.
- Graph-task configurations were selected using validation data. They are an
  extension and are not part of the original twelve-dataset node table.

Run the data-free model smoke test with:

```bash
python tests/smoke_models.py
```

## Citation and license

Update `CITATION.cff` with the final paper identifier and repository URL before
release. A software license has intentionally not been guessed: the authors
must select an OSI-approved license and add its `LICENSE` file before making
the repository public. See `RELEASE_CHECKLIST.md`.
