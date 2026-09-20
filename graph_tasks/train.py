#!/usr/bin/env python3
"""Train one validation-selected HGTCP graph-task run."""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch_geometric.loader import DataLoader

from hgtcp.model import HGTCPV2Graph
from hgtcp.utils import atomic_json, load_json, seed_everything

ROOT = Path(__file__).resolve().parents[1]


def metric_and_loss(logits, target, task):
    if task == "regression":
        logits = logits.view(-1)
        target = target.view(-1).float()
        loss = F.l1_loss(logits, target)
        return loss, float(torch.abs(logits - target).sum()), target.numel()
    target = target.view(-1).long()
    loss = F.cross_entropy(logits, target)
    return loss, int((logits.argmax(-1) == target).sum()), target.numel()


@torch.no_grad()
def evaluate(model, loader, device, task):
    model.eval()
    score_sum = item_count = 0
    loss_sum = 0.0
    for data in loader:
        data = data.to(device)
        logits, _ = model(data)
        loss, score, count = metric_and_loss(logits, data.y, task)
        score_sum += score
        item_count += count
        loss_sum += float(loss) * count
    key = "mae" if task == "regression" else "accuracy"
    return {"loss": loss_sum / item_count, key: score_sum / item_count}


def build_model(config):
    model = config["model"]
    return HGTCPV2Graph(
        dataset=config["dataset"],
        hidden=model["hidden"],
        global_layers=model["global_layers"],
        local_layers=model["local_layers"],
        heads=model["heads"],
        clusters=model["clusters"],
        alpha=model["alpha"],
        dropout=model["dropout"],
        out_dim=1 if config["task"] == "regression" else 10,
        graph_pooling=model["graph_pooling"],
        normalization=model["normalization"],
        ffn_depth=model["ffn_depth"],
        train_eps=model["train_eps"],
        hpde_level_offset=model["hpde_level_offset"],
        use_cluster_size_bias=model["use_cluster_size_bias"],
    )


def save_checkpoint(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def torch_load(path: Path, *, map_location):
    """Load PyG objects across both old and new torch defaults."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=ROOT / "data" / "cache")
    args = parser.parse_args()

    config = load_json(args.config)
    seed_everything(args.seed)
    training = config["training"]
    torch.set_num_threads(training.get("cpu_threads", 4))
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(args.device)

    cache_dir = args.cache_root / config["dataset"]
    datasets = {
        split: torch_load(cache_dir / f"{split}.pt", map_location="cpu")
        for split in ("train", "val", "test")
    }
    loaders = {
        split: DataLoader(
            values,
            batch_size=training["batch_size"],
            shuffle=split == "train",
            num_workers=training["num_workers"],
            persistent_workers=training["num_workers"] > 0,
        )
        for split, values in datasets.items()
    }

    model = build_model(config).to(device)
    optimizer_class = (
        torch.optim.Adam
        if training["optimizer"].lower() == "adam"
        else torch.optim.AdamW
    )
    optimizer = optimizer_class(
        model.parameters(), lr=training["lr"], weight_decay=training["weight_decay"]
    )
    if training["scheduler"] == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min" if config["task"] == "regression" else "max",
            factor=training.get("lr_decay", 0.5),
            patience=training.get("lr_patience", 20),
            min_lr=training["min_lr"],
        )
    else:
        warmup = training["warmup_epochs"]
        epochs = training["epochs"]

        def lr_scale(epoch):
            if epoch < warmup:
                return float(epoch) / max(1, warmup)
            progress = (epoch - warmup) / max(1, epochs - warmup)
            minimum = training["min_lr"] / training["lr"]
            return minimum + (1.0 - minimum) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)

    output = args.output.resolve()
    checkpoint = output.parent / f"seed_{args.seed}_last.pt"
    best_model = output.parent / f"seed_{args.seed}_best.pt"
    best_value = math.inf if config["task"] == "regression" else -math.inf
    best_epoch = -1
    best_validation = None
    started = time.time()
    key = "mae" if config["task"] == "regression" else "accuracy"

    for epoch in range(training["epochs"]):
        model.train()
        for data in loaders["train"]:
            data = data.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, contrastive_logits = model(data)
            task_loss, _, _ = metric_and_loss(logits, data.y, config["task"])
            half = contrastive_logits.size(0) // 2
            targets = torch.cat(
                (
                    torch.ones(half, device=device),
                    torch.zeros(contrastive_logits.size(0) - half, device=device),
                )
            ).view_as(contrastive_logits)
            contrastive_loss = F.binary_cross_entropy_with_logits(
                contrastive_logits, targets
            )
            loss = task_loss + training["contrastive_weight"] * contrastive_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training["grad_clip"])
            optimizer.step()

        validation = evaluate(model, loaders["val"], device, config["task"])
        if training["scheduler"] == "plateau":
            scheduler.step(validation[key])
        else:
            scheduler.step()
        improved = (
            validation[key] < best_value
            if config["task"] == "regression"
            else validation[key] > best_value
        )
        if improved:
            best_value = validation[key]
            best_epoch = epoch
            best_validation = validation
            save_checkpoint(best_model, {"model": model.state_dict(), "epoch": epoch})
        save_checkpoint(
            checkpoint,
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_value": best_value,
            },
        )
        print(
            f"epoch={epoch} val_{key}={validation[key]:.6f} "
            f"best={best_value:.6f}",
            flush=True,
        )

    state = torch_load(best_model, map_location=device)
    model.load_state_dict(state["model"])
    test = evaluate(model, loaders["test"], device, config["task"])
    atomic_json(
        output,
        {
            "status": "complete",
            "dataset": config["dataset"],
            "task": config["task"],
            "seed": args.seed,
            "best_epoch": best_epoch,
            "validation": best_validation,
            "test": test,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "elapsed_seconds": time.time() - started,
            "config": config,
        },
    )


if __name__ == "__main__":
    main()
