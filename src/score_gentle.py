#!/usr/bin/env python3
"""Scores of the gentler fine-tune on the validation and test molecules."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from model import DualEncoder
from train_panel import GraphStore, auroc, batch_graphs, encode_pockets

ROOT = Path(__file__).resolve().parents[1]


def topk_hit_rate(scores, potent, k=5):
    has = potent.any(axis=1)
    chosen = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    hits = np.array([potent[i, chosen[i]].any() for i in np.where(has)[0]])
    return float(hits.mean())


def report(name, scores, observed, rows):
    target = observed[rows]
    mask = np.isfinite(target)
    pair = auroc(target[mask].astype(np.int32), scores[mask])
    potent = target == 1
    panel = auroc(potent.any(axis=1).astype(np.int32), scores.max(axis=1))
    top5 = topk_hit_rate(scores, potent, 5)
    print("%s pair %.4f panel %.4f top5 %.4f" % (name, pair, panel, top5), flush=True)


def encode_split(model, ligands, pocket_embedding, indices, device, batch_size):
    rows = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            chosen = indices[start:start + batch_size]
            batch = batch_graphs(ligands, chosen, device)
            ligand = model.encode_ligand(
                batch["features"], batch["positions"], batch["edge_index"], batch["batch"],
            )
            rows.append(model.similarity(ligand, pocket_embedding).cpu().numpy())
            print("encoded %d/%d" % (min(start + batch_size, len(indices)), len(indices)), flush=True)
    return np.concatenate(rows, axis=0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = np.load(ROOT / "data" / "features" / "labels.npz")
    observed = labels["observed"]
    split = labels["split"]
    valid_rows = np.where(split == 1)[0]
    test_rows = np.where(split == 2)[0]
    ligands = GraphStore(ROOT / "data" / "features" / "ligands.npz")
    checkpoint = torch.load(
        ROOT / "data" / "models" / "panel_rank_gentle" / "panel_rank.pt",
        map_location=device, weights_only=True,
    )
    model = DualEncoder().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print("device %s checkpoint epoch %d" % (device, int(checkpoint["epoch"])), flush=True)
    with torch.no_grad():
        pocket_embedding = encode_pockets(model, GraphStore(ROOT / "data" / "features" / "pockets.npz"), device)
    valid_scores = encode_split(model, ligands, pocket_embedding, valid_rows, device, 128)
    test_scores = encode_split(model, ligands, pocket_embedding, test_rows, device, 128)
    report("valid", valid_scores, observed, valid_rows)
    report("test", test_scores, observed, test_rows)
    out = ROOT / "data" / "models" / "baselines" / "gentle_scores.npz"
    np.savez_compressed(
        out,
        valid_index=valid_rows.astype(np.int32),
        test_index=test_rows.astype(np.int32),
        valid_scores=valid_scores.astype(np.float32),
        test_scores=test_scores.astype(np.float32),
    )
    print("wrote %s" % out, flush=True)


if __name__ == "__main__":
    main()
