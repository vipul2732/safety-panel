#!/usr/bin/env python3
"""Train ligand plus protein sequence on the same scaffold split.

Each target is its UniProt sequence. Pocket coordinates are not used. The
loss and early stopping match the pocket and ligand-only models.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from model import SequenceLigandEncoder
from train_ligand_only import run_epoch, subset
from train_panel import GraphStore

ROOT = Path(__file__).resolve().parents[1]
AMINO = "ACDEFGHIKLMNPQRSTVWY"


def read_fasta(path):
    sequences = {}
    accession = None
    chunks = []
    for line in Path(path).read_text().splitlines():
        if line.startswith(">"):
            if accession:
                sequences[accession] = "".join(chunks)
            accession = line[1:].split()[0]
            chunks = []
        else:
            chunks.append(line.strip())
    if accession:
        sequences[accession] = "".join(chunks)
    return sequences


def sequence_tensors(panel_ids, targets_csv, fasta_path):
    sequences = read_fasta(fasta_path)
    accessions = {
        row["panel_id"]: row["uniprot_accessions"].split(";")
        for row in csv.DictReader(open(targets_csv))
    }
    chains = []
    target_index = []
    for index, panel in enumerate(panel_ids):
        for accession in accessions[str(panel)]:
            sequence = sequences[accession]
            chains.append([AMINO.index(letter) + 1 if letter in AMINO else 21 for letter in sequence])
            target_index.append(index)
    width = max(len(chain) for chain in chains)
    tokens = torch.zeros(len(chains), width, dtype=torch.long)
    mask = torch.zeros(len(chains), width, dtype=torch.float32)
    for row, chain in enumerate(chains):
        tokens[row, :len(chain)] = torch.tensor(chain)
        mask[row, :len(chain)] = 1
    return tokens, torch.tensor(target_index, dtype=torch.long), mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data" / "features"))
    parser.add_argument("--targets", default=str(ROOT / "data" / "modeling" / "targets.csv"))
    parser.add_argument("--fasta", default=str(ROOT / "data" / "sequences" / "uniprot.fasta"))
    parser.add_argument("--out", default=str(ROOT / "data" / "models" / "sequence"))
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
    tokens, target_index, mask = sequence_tensors(labels["panels"], args.targets, args.fasta)
    train_index = subset(split, 0, args.limit)
    valid_index = subset(split, 1, args.limit)
    test_index = subset(split, 2, args.limit)
    n_pos = float(np.nansum(observed[train_index] == 1))
    n_neg = float(np.nansum(observed[train_index] == 0))
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=device)
    print("device %s train %d valid %d test %d chains %d length %d" % (
        device, len(train_index), len(valid_index), len(test_index), tokens.size(0), tokens.size(1),
    ), flush=True)

    model = SequenceLigandEncoder(tokens, target_index, mask).to(device)
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
                }, out / "sequence.pt")
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
