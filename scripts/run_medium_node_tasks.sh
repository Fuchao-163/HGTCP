#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root/node_classification/medium"
data_root="${HGTCP_DATA_ROOT:-$repo_root/data/node}"
device="${HGTCP_DEVICE:-0}"

run() {
  python train.py --method hgtcp --device "$device" --data_dir "$data_root" \
    --runs 5 --epochs 500 --no_feat_norm "$@"
}

run --dataset cora --lr 0.01 --hidden_channels 128 --ghidden_channels 128 \
  --gnum_layers 2 --num_layers 1 --num_heads 4 --num_centroids 128 \
  --alpha 0.7 --lamda 0.1 --weight_decay 0.0005

run --dataset citeseer --lr 0.01 --hidden_channels 128 --ghidden_channels 128 \
  --gnum_layers 2 --num_layers 2 --num_heads 2 --num_centroids 64 \
  --alpha 0.7 --lamda 0.05 --weight_decay 0.01

run --dataset pubmed --lr 0.01 --hidden_channels 128 --ghidden_channels 128 \
  --gnum_layers 2 --num_layers 1 --num_heads 1 --num_centroids 128 \
  --alpha 0.7 --lamda 0.05 --weight_decay 0.0005

run --dataset Photo --lr 0.01 --hidden_channels 128 --ghidden_channels 128 \
  --gnum_layers 2 --num_layers 2 --num_heads 1 --num_centroids 64 \
  --alpha 0.7 --lamda 0.05 --weight_decay 0.0005 --rand_split

run --dataset film --lr 0.01 --hidden_channels 128 --ghidden_channels 128 \
  --gnum_layers 2 --num_layers 2 --num_heads 2 --num_centroids 128 \
  --alpha 0.5 --lamda 0 --weight_decay 0.0005

run --dataset squirrel --lr 0.01 --hidden_channels 128 --ghidden_channels 128 \
  --gnum_layers 3 --num_layers 3 --num_heads 1 --num_centroids 256 \
  --alpha 0.3 --lamda 1 --weight_decay 0.0005

run --dataset chameleon --lr 0.01 --hidden_channels 256 --ghidden_channels 256 \
  --gnum_layers 3 --num_layers 3 --num_heads 1 --num_centroids 256 \
  --alpha 0.3 --lamda 0.7 --weight_decay 0.001
