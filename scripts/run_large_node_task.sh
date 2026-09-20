#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 {flickr|reddit|ogbn-arxiv|ogbn-proteins|ogbn-products} [device]" >&2
  exit 2
fi

task="$1"
device="${2:-${HGTCP_DEVICE:-0}}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_root="${HGTCP_DATA_ROOT:-$repo_root/data/node}"

case "$task" in
  flickr)
    cd "$repo_root/node_classification/large"
    python arxiv_node2vec.py --dataset Flickr --embedding_dim 128 --seed 0 \
      --data_root "$data_root"
    for seed in 0 1 2 3 4; do
      python arxiv_ERM_ns.py --dataset Flickr --data_root "$data_root" \
        --device "$device" --lr 0.0005 --hidden_dim 128 --global_dim 128 \
        --num_layers 1 --num_heads 1 --num_centroids 256 --alpha 0.3 \
        --lamda 0.5 --conv_type full --epochs 500 --seed "$seed"
    done
    ;;
  reddit)
    cd "$repo_root/node_classification/large"
    python arxiv_node2vec.py --dataset Reddit --embedding_dim 128 --seed 0 \
      --data_root "$data_root"
    for seed in 0 1 2 3 4; do
      python arxiv_ERM_ns.py --dataset Reddit --data_root "$data_root" \
        --device "$device" --lr 0.0005 --hidden_dim 128 --global_dim 128 \
        --num_layers 1 --num_heads 1 --num_centroids 1024 --alpha 0.5 \
        --lamda 0.05 --conv_type full --epochs 500 --seed "$seed"
    done
    ;;
  ogbn-arxiv)
    cd "$repo_root/node_classification/large"
    python arxiv_node2vec.py --dataset ogbn-arxiv --embedding_dim 256 --seed 0 \
      --data_root "$data_root"
    for seed in 0 1 2 3 4; do
      python arxiv_ERM_ns.py --dataset ogbn-arxiv --data_root "$data_root" \
        --device "$device" --lr 0.0005 --hidden_dim 256 --global_dim 256 \
        --num_layers 1 --num_heads 1 --num_centroids 1024 --alpha 0.5 \
        --lamda 0.05 --conv_type full --epochs 500 --seed "$seed"
    done
    ;;
  ogbn-proteins)
    cd "$repo_root/node_classification/ogbn_proteins"
    python proteins.py --device "$device" --runs 5 --seed 0 --data_root "$data_root" --lr 0.0005 \
      --hidden_channels 128 --ghidden_channels 128 --gnum_layers 4 \
      --num_layers 1 --num_heads 2 --num_centroids 1024 --alpha 0.7 \
      --lamda 0.05
    ;;
  ogbn-products)
    cd "$repo_root/node_classification/ogbn_products"
    for seed in 0 1 2 3 4; do
      python train.py --device "$device" --data_root "$data_root" --seed "$seed" \
        --lr 0.0005 --num_centroids 1024 --alpha 0.5 --lamda 0.05
    done
    ;;
  *)
    echo "Unknown task: $task" >&2
    exit 2
    ;;
esac
