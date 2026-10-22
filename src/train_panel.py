#!/usr/bin/env python3
"""Train the pocket-ligand encoder on the safety-panel scaffold split.

Each step encodes the query ligands and all 71 pockets. Observed potent and
weak pairs train a binder score. Potent pairs also train a pChEMBL head.
Unmeasured pairs are masked. The checkpoint is chosen on validation pair AUROC.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn

from model import DualEncoder

ROOT = Path(__file__).resolve().parents[1]


class GraphStore:
    def __init__(self, path):
        data = np.load(path)
        self.x = data["x"]
        self.pos = data["pos"]
        self.edge = data["edge"]
        self.node_ptr = data["node_ptr"]
        self.edge_ptr = data["edge_ptr"]

    def get(self, index):
        node_start, node_end = int(self.node_ptr[index]), int(self.node_ptr[index + 1])
        edge_start, edge_end = int(self.edge_ptr[index]), int(self.edge_ptr[index + 1])
        return (
            self.x[node_start:node_end],
            self.pos[node_start:node_end],
            self.edge[:, edge_start:edge_end],
        )


def batch_graphs(store, indices, device):
    features, positions, edges, batches = [], [], [], []
    offset = 0
    for batch_index, index in enumerate(indices):
        x, pos, edge = store.get(int(index))
        features.append(torch.from_numpy(np.array(x)))
        positions.append(torch.from_numpy(np.array(pos)))
        edges.append(torch.from_numpy(np.array(edge)) + offset)
        batches.append(torch.full((len(x),), batch_index, dtype=torch.long))
        offset += len(x)
    return {
        "features": torch.cat(features).to(device),
        "positions": torch.cat(positions).to(device),
        "edge_index": torch.cat(edges, dim=1).long().to(device),
        "batch": torch.cat(batches).to(device),
    }


def auroc(labels, scores):
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        average = 0.5 * (start + 1 + end)
        ranks[order[start:end]] = average
        start = end
    sum_pos = ranks[labels == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def encode_pockets(model, pockets, device):
    batch = batch_graphs(pockets, range(len(pockets.node_ptr) - 1), device)
    return model.encode_pocket(batch["features"], batch["positions"], batch["edge_index"], batch["batch"])


def run_epoch(model, ligands, pockets, observed, potency, indices, device, optimizer, pos_weight, batch_size, train):
    order = np.array(indices)
    if train:
        order = np.random.permutation(order)
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    total_pairs = 0
    pair_labels, pair_scores = [], []
    molecule_labels, molecule_scores = [], []
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for start in range(0, len(order), batch_size):
            chosen = order[start:start + batch_size]
            ligand_batch = batch_graphs(ligands, chosen, device)
            ligand_embedding = model.encode_ligand(
                ligand_batch["features"], ligand_batch["positions"],
                ligand_batch["edge_index"], ligand_batch["batch"],
            )
            pocket_embedding = encode_pockets(model, pockets, device)
            similarity = model.similarity(ligand_embedding, pocket_embedding)
            target = torch.from_numpy(observed[chosen]).to(device)
            mask = torch.isfinite(target)
            if int(mask.sum()) == 0:
                continue
            binder_loss = nn.functional.binary_cross_entropy_with_logits(
                similarity[mask].unsqueeze(-1), target[mask].unsqueeze(-1), pos_weight=pos_weight,
            )
            potent = mask & (target > 0.5)
            if int(potent.sum()) > 0:
                ligand_index, pocket_index = torch.where(potent)
                predicted = model.predict_potency(ligand_embedding[ligand_index], pocket_embedding[pocket_index])
                reference = torch.from_numpy(potency[chosen]).to(device)[ligand_index, pocket_index]
                potency_loss = nn.functional.mse_loss(predicted, reference)
            else:
                potency_loss = similarity.new_zeros(())
            loss = binder_loss + 0.1 * potency_loss
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            n_pairs = int(mask.sum())
            total_loss += float(loss.detach()) * n_pairs
            total_pairs += n_pairs
            pair_labels.append(target[mask].detach().cpu().numpy())
            pair_scores.append(similarity[mask].detach().cpu().numpy())
            molecule_labels.append(((observed[chosen] == 1).sum(axis=1) > 0).astype(np.int32))
            molecule_scores.append(similarity.detach().max(dim=1).values.cpu().numpy())
    labels = np.concatenate(pair_labels) if pair_labels else np.array([])
    scores = np.concatenate(pair_scores) if pair_scores else np.array([])
    mol_labels = np.concatenate(molecule_labels) if molecule_labels else np.array([])
    mol_scores = np.concatenate(molecule_scores) if molecule_scores else np.array([])
    return {
        "loss": total_loss / max(total_pairs, 1),
        "pair_auroc": auroc(labels, scores) if len(labels) else float("nan"),
        "panel_auroc": auroc(mol_labels, mol_scores) if len(mol_labels) else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data" / "features"))
    parser.add_argument("--out", default=str(ROOT / "data" / "models"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pockets = GraphStore(data / "pockets.npz")
    ligands = GraphStore(data / "ligands.npz")
    labels = np.load(data / "labels.npz")
    observed = labels["observed"]
    potency = labels["potency"]
    split = labels["split"]
    train_index = np.where(split == 0)[0]
    valid_index = np.where(split == 1)[0]
    n_pos = float(np.nansum(observed[train_index] == 1))
    n_neg = float(np.nansum(observed[train_index] == 0))
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=device)
    print("device %s train %d valid %d potent %.0f weak %.0f" % (
        device, len(train_index), len(valid_index), n_pos, n_neg,
    ), flush=True)

    model = DualEncoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best = -1.0
    waited = 0
    start_epoch = 1
    checkpoint_path = out / "dual_encoder.pt"
    if args.resume:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best = float(checkpoint["valid_pair_auroc"])
        print("resume from epoch %d best valid pair %.4f" % (int(checkpoint["epoch"]), best), flush=True)
    metrics_path = out / "metrics.csv"
    with metrics_path.open("a" if args.resume else "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "epoch", "train_loss", "train_pair_auroc", "valid_loss", "valid_pair_auroc", "valid_panel_auroc",
        ])
        if not args.resume:
            writer.writeheader()
        for epoch in range(start_epoch, args.epochs + 1):
            train_metrics = run_epoch(
                model, ligands, pockets, observed, potency, train_index, device,
                optimizer, pos_weight, args.batch_size, train=True,
            )
            valid_metrics = run_epoch(
                model, ligands, pockets, observed, potency, valid_index, device,
                optimizer, pos_weight, args.batch_size, train=False,
            )
            row = {
                "epoch": epoch,
                "train_loss": "%.4f" % train_metrics["loss"],
                "train_pair_auroc": "%.4f" % train_metrics["pair_auroc"],
                "valid_loss": "%.4f" % valid_metrics["loss"],
                "valid_pair_auroc": "%.4f" % valid_metrics["pair_auroc"],
                "valid_panel_auroc": "%.4f" % valid_metrics["panel_auroc"],
            }
            writer.writerow(row)
            handle.flush()
            print(
                "epoch %d loss %.4f valid_pair %.4f valid_panel %.4f" % (
                    epoch, train_metrics["loss"], valid_metrics["pair_auroc"], valid_metrics["panel_auroc"],
                ),
                flush=True,
            )
            score = valid_metrics["pair_auroc"]
            if score > best:
                best = score
                waited = 0
                torch.save({
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "valid_pair_auroc": score,
                    "valid_panel_auroc": valid_metrics["panel_auroc"],
                }, out / "dual_encoder.pt")
            else:
                waited += 1
                if waited >= args.patience:
                    print("early stop", flush=True)
                    break
    if checkpoint_path.exists() and len(np.where(split == 2)[0]):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        test_index = np.where(split == 2)[0]
        test_metrics = run_epoch(
            model, ligands, pockets, observed, potency, test_index, device,
            optimizer, pos_weight, args.batch_size, train=False,
        )
        print(
            "best valid pair AUROC %.4f test pair %.4f test panel %.4f" % (
                best, test_metrics["pair_auroc"], test_metrics["panel_auroc"],
            ),
            flush=True,
        )
        (out / "test_metrics.csv").write_text(
            "pair_auroc,panel_auroc\n%.4f,%.4f\n" % (test_metrics["pair_auroc"], test_metrics["panel_auroc"])
        )


if __name__ == "__main__":
    main()
