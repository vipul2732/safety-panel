#!/usr/bin/env python3
"""Held-out scores for the pocket, ligand-only, and sequence models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from model import DualEncoder, LigandOnlyEncoder, SequenceLigandEncoder
from score_panel import pocket_scores
from train_panel import GraphStore, auroc, batch_graphs, encode_pockets
from train_sequence import sequence_tensors

ROOT = Path(__file__).resolve().parents[1]


def topk_hit_rate(scores, potent, k=5):
    has = potent.any(axis=1)
    chosen = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    hits = np.array([potent[i, chosen[i]].any() for i in np.where(has)[0]])
    return float(hits.mean()), int(has.sum())


def encode_scores(model, ligands, indices, device, batch_size):
    rows = []
    model.eval()
    with torch.no_grad():
        targets = model.target_embedding()
        for start in range(0, len(indices), batch_size):
            chosen = indices[start:start + batch_size]
            batch = batch_graphs(ligands, chosen, device)
            ligand = model.encode_ligand(
                batch["features"], batch["positions"], batch["edge_index"], batch["batch"],
            )
            rows.append(model.similarity(ligand, targets).cpu().numpy())
            print("encoded %d/%d" % (min(start + batch_size, len(indices)), len(indices)), flush=True)
    return np.concatenate(rows, axis=0)


def report(name, scores, observed, rows, path):
    target = observed[rows]
    mask = np.isfinite(target)
    pair = auroc(target[mask].astype(np.int32), scores[mask])
    potent = target == 1
    panel = auroc((np.nansum(potent, axis=1) > 0).astype(np.int32), scores.max(axis=1))
    top5, n_mol = topk_hit_rate(scores, potent, 5)
    path.write_text("pair_auroc,panel_auroc,top5_hit_rate,n_molecules\n%.4f,%.4f,%.4f,%d\n" % (pair, panel, top5, n_mol))
    print("%s pair %.4f panel %.4f top5 %.4f" % (name, pair, panel, top5), flush=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = np.load(ROOT / "data" / "features" / "labels.npz")
    observed = labels["observed"]
    split = labels["split"]
    test_rows = np.where(split == 2)[0]
    ligands = GraphStore(ROOT / "data" / "features" / "ligands.npz")
    out = ROOT / "data" / "models" / "baselines"
    out.mkdir(parents=True, exist_ok=True)
    print("device %s test %d" % (device, len(test_rows)), flush=True)

    stored = np.load(ROOT / "data" / "models" / "ligand_embeddings.npz")
    checkpoint = torch.load(ROOT / "data" / "models" / "dual_encoder.pt", map_location=device, weights_only=True)
    pocket_model = DualEncoder().to(device)
    pocket_model.load_state_dict(checkpoint["model"])
    pocket_model.eval()
    with torch.no_grad():
        pocket_embedding = encode_pockets(pocket_model, GraphStore(ROOT / "data" / "features" / "pockets.npz"), device).cpu().numpy()
        scale = float(pocket_model.logit_scale.exp().clamp(max=100))
    pocket_scores_all = pocket_scores(stored["embedding"], pocket_embedding, scale)
    pocket = pocket_scores_all[test_rows]
    report("pocket", pocket, observed, test_rows, out / "pocket_metrics.csv")

    ligand_model = LigandOnlyEncoder(observed.shape[1]).to(device)
    ligand_state = torch.load(ROOT / "data" / "models" / "ligand_only" / "ligand_only.pt", map_location=device, weights_only=True)
    ligand_model.load_state_dict(ligand_state["model"])
    ligand = encode_scores(ligand_model, ligands, test_rows, device, 128)
    report("ligand_only", ligand, observed, test_rows, out / "ligand_only_metrics.csv")

    tokens, target_index, mask = sequence_tensors(
        labels["panels"], ROOT / "data" / "modeling" / "targets.csv", ROOT / "data" / "sequences" / "uniprot.fasta",
    )
    sequence_model = SequenceLigandEncoder(tokens, target_index, mask).to(device)
    sequence_state = torch.load(ROOT / "data" / "models" / "sequence" / "sequence.pt", map_location=device, weights_only=True)
    sequence_model.load_state_dict(sequence_state["model"])
    sequence = encode_scores(sequence_model, ligands, test_rows, device, 64)
    report("sequence", sequence, observed, test_rows, out / "sequence_metrics.csv")

    np.savez_compressed(
        out / "model_scores.npz",
        test_index=test_rows.astype(np.int32),
        pocket=pocket.astype(np.float32),
        ligand_only=ligand.astype(np.float32),
        sequence=sequence.astype(np.float32),
    )
    print("wrote %s" % out, flush=True)


if __name__ == "__main__":
    main()
