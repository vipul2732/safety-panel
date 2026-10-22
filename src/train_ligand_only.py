#!/usr/bin/env python3
"""Train a ligand-only panel model on the same scaffold split.

Each target is a learned vector. No pocket is used. The loss, split, and
early stopping match the pocket model so the two can be compared.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn

from model import LigandOnlyEncoder
from train_panel import GraphStore, auroc, batch_graphs

ROOT = Path(__file__).resolve().parents[1]


def run_epoch(model, ligands, observed, potency, indices, device, optimizer, pos_weight, batch_size, train):
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
            target_embedding = model.target_embedding()
            similarity = model.similarity(ligand_embedding, target_embedding)
            target = torch.from_numpy(observed[chosen]).to(device)
            mask = torch.isfinite(target)
            if int(mask.sum()) == 0:
                continue
            binder_loss = nn.functional.binary_cross_entropy_with_logits(
                similarity[mask].unsqueeze(-1), target[mask].unsqueeze(-1), pos_weight=pos_weight,
            )
            potent = mask & (target > 0.5)
            if int(potent.sum()) > 0:
                ligand_index, target_index = torch.where(potent)
                predicted = model.predict_potency(ligand_embedding[ligand_index], target_embedding[target_index])
                reference = torch.from_numpy(potency[chosen]).to(device)[ligand_index, target_index]
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


def subset(split, code, limit):
    index = np.where(split == code)[0]
    if limit:
        return index[:limit]
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data" / "features"))
    parser.add_argument("--out", default=str(ROOT / "data" / "models" / "ligand_only"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ligands = GraphStore(data / "ligands.npz")
    labels = np.load(data / "labels.npz")
    observed = labels["observed"]
    potency = labels["potency"]
    split = labels["split"]
    train_index = subset(split, 0, args.limit)
    valid_index = subset(split, 1, args.limit)
    test_index = subset(split, 2, args.limit)
    n_pos = float(np.nansum(observed[train_index] == 1))
    n_neg = float(np.nansum(observed[train_index] == 0))
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=device)
    print("device %s train %d valid %d test %d potent %.0f weak %.0f" % (
        device, len(train_index), len(valid_index), len(test_index), n_pos, n_neg,
    ), flush=True)

    model = LigandOnlyEncoder(observed.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best = -1.0
    waited = 0
    with (out / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "epoch", "train_loss", "train_pair_auroc", "valid_loss", "valid_pair_auroc", "valid_panel_auroc",
        ])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(
                model, ligands, observed, potency, train_index, device,
                optimizer, pos_weight, args.batch_size, train=True,
            )
            valid_metrics = run_epoch(
                model, ligands, observed, potency, valid_index, device,
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
                    "n_targets": int(observed.shape[1]),
                }, out / "ligand_only.pt")
                if len(test_index):
                    test_metrics = run_epoch(
                        model, ligands, observed, potency, test_index, device,
                        optimizer, pos_weight, args.batch_size, train=False,
                    )
                    (out / "test_metrics.csv").write_text(
                        "pair_auroc,panel_auroc\n%.4f,%.4f\n" % (
                            test_metrics["pair_auroc"], test_metrics["panel_auroc"],
                        )
                    )
                    print(
                        "test pair %.4f test panel %.4f" % (
                            test_metrics["pair_auroc"], test_metrics["panel_auroc"],
                        ),
                        flush=True,
                    )
            else:
                waited += 1
                if waited >= args.patience:
                    print("early stop", flush=True)
                    break
    print("best valid pair AUROC %.4f" % best, flush=True)


if __name__ == "__main__":
    main()
