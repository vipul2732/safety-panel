#!/usr/bin/env python3
"""Fine-tune the pass-15 pocket model so the highest score is meaningful.

The original pair loss is unchanged. Two extra terms are added. A known potent
target has to outrank the pockets that are not known potents for that molecule.
A molecule whose measurements are all weak has to bring its highest score down.
Unmeasured pairs are still left out of the potent-versus-weak loss.

Checkpoints go to a new directory. data/models/dual_encoder.pt is only read.
An epoch is kept only when its validation pair AUROC stays within 0.02 of the
pass-15 value. Among those epochs, the highest validation panel AUROC is saved.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn

from model import DualEncoder
from train_panel import GraphStore, auroc, batch_graphs, encode_pockets

ROOT = Path(__file__).resolve().parents[1]
PAIR_DROP_LIMIT = 0.02


def ranking_terms(similarity, target):
    """Return the rank hinge and the weak-only maximum loss."""
    potent = torch.isfinite(target) & (target > 0.5)
    weak = torch.isfinite(target) & (target < 0.5)
    has_potent = potent.any(dim=1)
    weak_only = weak.any(dim=1) & ~has_potent
    rank_loss = similarity.new_zeros(())
    weak_loss = similarity.new_zeros(())
    if bool(has_potent.any()):
        best_other = similarity.masked_fill(potent, -1e4).max(dim=1).values
        gap = best_other.unsqueeze(1) - similarity
        rank_loss = torch.relu(gap).masked_fill(~potent, 0).sum() / potent.sum().clamp(min=1)
    if bool(weak_only.any()):
        weak_loss = nn.functional.binary_cross_entropy_with_logits(
            similarity[weak_only].max(dim=1).values,
            torch.zeros(int(weak_only.sum()), device=similarity.device),
        )
    return rank_loss, weak_loss


def topk_hit_rate(scores, observed, k=5):
    potent = observed == 1
    has = potent.any(axis=1)
    if int(has.sum()) == 0:
        return float("nan")
    chosen = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    hits = [bool(potent[i, chosen[i]].any()) for i in np.where(has)[0]]
    return float(np.mean(hits))


def run_epoch(
    model, ligands, pockets, observed, potency, indices, device, optimizer,
    pos_weight, batch_size, train, rank_weight, weak_weight,
):
    order = np.array(indices)
    if train:
        order = np.random.permutation(order)
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    total_rank = 0.0
    total_weak = 0.0
    total_pairs = 0
    pair_labels, pair_scores = [], []
    molecule_labels, molecule_scores, full_scores = [], [], []
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
            rank_loss, weak_loss = ranking_terms(similarity, target)
            loss = binder_loss + 0.1 * potency_loss + rank_weight * rank_loss + weak_weight * weak_loss
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            n_pairs = int(mask.sum())
            total_loss += float(loss.detach()) * n_pairs
            total_rank += float(rank_loss.detach()) * n_pairs
            total_weak += float(weak_loss.detach()) * n_pairs
            total_pairs += n_pairs
            pair_labels.append(target[mask].detach().cpu().numpy())
            pair_scores.append(similarity[mask].detach().cpu().numpy())
            molecule_labels.append(((observed[chosen] == 1).sum(axis=1) > 0).astype(np.int32))
            molecule_scores.append(similarity.detach().max(dim=1).values.cpu().numpy())
            if not train:
                full_scores.append(similarity.detach().cpu().numpy())
    labels = np.concatenate(pair_labels) if pair_labels else np.array([])
    scores = np.concatenate(pair_scores) if pair_scores else np.array([])
    mol_labels = np.concatenate(molecule_labels) if molecule_labels else np.array([])
    mol_scores = np.concatenate(molecule_scores) if molecule_scores else np.array([])
    top5 = float("nan")
    if full_scores:
        top5 = topk_hit_rate(np.concatenate(full_scores, axis=0), observed[order])
    return {
        "loss": total_loss / max(total_pairs, 1),
        "rank_loss": total_rank / max(total_pairs, 1),
        "weak_loss": total_weak / max(total_pairs, 1),
        "pair_auroc": auroc(labels, scores) if len(labels) else float("nan"),
        "panel_auroc": auroc(mol_labels, mol_scores) if len(mol_labels) else float("nan"),
        "top5": top5,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data" / "features"))
    parser.add_argument("--init", default=str(ROOT / "data" / "models" / "dual_encoder.pt"))
    parser.add_argument("--out", default=str(ROOT / "data" / "models" / "panel_rank"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--rank-weight", type=float, default=0.1)
    parser.add_argument("--weak-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if Path(args.out).resolve() == (ROOT / "data" / "models").resolve():
        raise SystemExit("refusing to write into data/models")
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
    test_index = np.where(split == 2)[0]
    if args.limit:
        train_index = train_index[:args.limit]
        valid_index = valid_index[:args.limit]
        test_index = test_index[:args.limit]
    n_pos = float(np.nansum(observed[train_index] == 1))
    n_neg = float(np.nansum(observed[train_index] == 0))
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=device)

    model = DualEncoder().to(device)
    checkpoint = torch.load(args.init, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    baseline_pair = float(checkpoint["valid_pair_auroc"])
    baseline_panel = float(checkpoint["valid_panel_auroc"])
    floor = baseline_pair - PAIR_DROP_LIMIT
    print(
        "device %s init epoch %d pair %.4f panel %.4f floor %.4f" % (
            device, int(checkpoint["epoch"]), baseline_pair, baseline_panel, floor,
        ),
        flush=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_panel = -1.0
    best_pair = -1.0
    waited = 0
    checkpoint_path = out / "panel_rank.pt"
    fields = [
        "epoch", "train_loss", "train_rank_loss", "train_weak_loss", "train_pair_auroc",
        "valid_loss", "valid_pair_auroc", "valid_panel_auroc", "valid_top5", "eligible",
    ]
    with (out / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(
                model, ligands, pockets, observed, potency, train_index, device,
                optimizer, pos_weight, args.batch_size, True, args.rank_weight, args.weak_weight,
            )
            valid_metrics = run_epoch(
                model, ligands, pockets, observed, potency, valid_index, device,
                optimizer, pos_weight, args.batch_size, False, args.rank_weight, args.weak_weight,
            )
            eligible = int(valid_metrics["pair_auroc"] >= floor)
            writer.writerow({
                "epoch": epoch,
                "train_loss": "%.4f" % train_metrics["loss"],
                "train_rank_loss": "%.4f" % train_metrics["rank_loss"],
                "train_weak_loss": "%.4f" % train_metrics["weak_loss"],
                "train_pair_auroc": "%.4f" % train_metrics["pair_auroc"],
                "valid_loss": "%.4f" % valid_metrics["loss"],
                "valid_pair_auroc": "%.4f" % valid_metrics["pair_auroc"],
                "valid_panel_auroc": "%.4f" % valid_metrics["panel_auroc"],
                "valid_top5": "%.4f" % valid_metrics["top5"],
                "eligible": eligible,
            })
            handle.flush()
            print(
                "epoch %d loss %.4f rank %.4f weak %.4f valid_pair %.4f valid_panel %.4f top5 %.4f" % (
                    epoch, train_metrics["loss"], train_metrics["rank_loss"], train_metrics["weak_loss"],
                    valid_metrics["pair_auroc"], valid_metrics["panel_auroc"], valid_metrics["top5"],
                ),
                flush=True,
            )
            if eligible and valid_metrics["panel_auroc"] > best_panel:
                best_panel = valid_metrics["panel_auroc"]
                best_pair = valid_metrics["pair_auroc"]
                waited = 0
                torch.save({
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "valid_pair_auroc": valid_metrics["pair_auroc"],
                    "valid_panel_auroc": valid_metrics["panel_auroc"],
                    "valid_top5": valid_metrics["top5"],
                    "init_epoch": int(checkpoint["epoch"]),
                    "baseline_pair_auroc": baseline_pair,
                    "baseline_panel_auroc": baseline_panel,
                }, checkpoint_path)
            else:
                waited += 1
                if waited >= args.patience:
                    print("early stop", flush=True)
                    break
    if not checkpoint_path.exists():
        print("no epoch stayed within %.2f of pass 15" % PAIR_DROP_LIMIT, flush=True)
        return
    saved = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(saved["model"])
    test_metrics = run_epoch(
        model, ligands, pockets, observed, potency, test_index, device,
        optimizer, pos_weight, args.batch_size, False, args.rank_weight, args.weak_weight,
    )
    print(
        "saved epoch %d valid_pair %.4f valid_panel %.4f test_pair %.4f test_panel %.4f test_top5 %.4f" % (
            int(saved["epoch"]), best_pair, best_panel,
            test_metrics["pair_auroc"], test_metrics["panel_auroc"], test_metrics["top5"],
        ),
        flush=True,
    )
    (out / "test_metrics.csv").write_text(
        "pair_auroc,panel_auroc,top5_hit_rate\n%.4f,%.4f,%.4f\n" % (
            test_metrics["pair_auroc"], test_metrics["panel_auroc"], test_metrics["top5"],
        )
    )


if __name__ == "__main__":
    main()
