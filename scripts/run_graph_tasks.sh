#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
device="${HGTCP_DEVICE:-cuda:0}"

cd "$repo_root"
python graph_tasks/prepare_data.py --datasets zinc mnist
for dataset in zinc mnist; do
  for seed in 0 1 2 3 4; do
    python graph_tasks/train.py \
      --config "graph_tasks/configs/${dataset}.json" \
      --seed "$seed" \
      --device "$device" \
      --output "results/graph/${dataset}/seed_${seed}.json"
  done
done
python scripts/summarize_graph_results.py results/graph
