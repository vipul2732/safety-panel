#!/usr/bin/env python3
"""Score the saved panel model and fit the molecule-level cutoff.

Validation scaffolds choose two score thresholds and an unfamiliar-molecule
distance. Test scaffolds are scored once with those fixed rules.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from model import DualEncoder
from train_panel import GraphStore, auroc, batch_graphs, encode_pockets

ROOT = Path(__file__).resolve().parents[1]
CLEAN_SENSITIVITY = 0.95
LIKELY_PRECISION = 0.95
MIN_LIKELY = 100
OOD_QUANTILE = 0.99


def encode_ligands(model, ligands, indices, device, batch_size):
    encoded = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            chosen = indices[start:start + batch_size]
            batch = batch_graphs(ligands, chosen, device)
            encoded.append(model.encode_ligand(
                batch["features"], batch["positions"], batch["edge_index"], batch["batch"],
            ).cpu())
            done = min(start + batch_size, len(indices))
            if done == len(indices) or start % (batch_size * 50) == 0:
                print("encoded ligands %d/%d" % (done, len(indices)), flush=True)
    return torch.cat(encoded, dim=0).numpy()


def pocket_scores(ligand_embedding, pocket_embedding, scale):
    scores = np.empty((len(ligand_embedding), len(pocket_embedding)), dtype=np.float32)
    pocket = pocket_embedding.T
    for start in range(0, len(ligand_embedding), 4096):
        end = min(start + 4096, len(ligand_embedding))
        scores[start:end] = scale * ligand_embedding[start:end] @ pocket
    return scores


def nearest_cosine(query, library, device, exclude_library_row=None):
    query_tensor = torch.from_numpy(np.ascontiguousarray(query))
    library_tensor = torch.from_numpy(np.ascontiguousarray(library))
    best = np.full(len(query), -2.0, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(query), 256):
            end = min(start + 256, len(query))
            chunk = query_tensor[start:end].to(device)
            local_best = torch.full((end - start,), -2.0, device=device)
            rows = None if exclude_library_row is None else exclude_library_row[start:end]
            for library_start in range(0, len(library), 8192):
                library_end = min(library_start + 8192, len(library))
                dots = chunk @ library_tensor[library_start:library_end].to(device).transpose(0, 1)
                if rows is not None:
                    active = np.where((rows >= library_start) & (rows < library_end))[0]
                    if len(active):
                        local = torch.from_numpy(active.astype(np.int64)).to(device)
                        column = torch.from_numpy((rows[active] - library_start).astype(np.int64)).to(device)
                        dots[local, column] = -2
                local_best = torch.maximum(local_best, dots.max(dim=1).values)
            best[start:end] = local_best.cpu().numpy()
            if end == len(query) or start % 10240 == 0:
                print("nearest neighbor %d/%d" % (end, len(query)), flush=True)
    return best


def pair_arrays(observed, similarity, indices):
    target = observed[indices]
    mask = np.isfinite(target)
    return target[mask].astype(np.int32), similarity[mask]


def most_inclusive_threshold(scores, labels, min_precision, min_count):
    for threshold in np.unique(scores):
        chosen = scores >= threshold
        count = int(chosen.sum())
        if count < min_count:
            continue
        precision = float(labels[chosen].mean())
        if precision + 1e-12 >= min_precision:
            return float(threshold), precision, count
    return None


def assign_call(score, distance, t_low, t_high, d_unfamiliar):
    if distance > d_unfamiliar and score < t_low:
        return "unfamiliar"
    if score >= t_high:
        return "likely"
    if score < t_low:
        return "very_unlikely"
    return "uncertain"


def operating_point(scores, labels, distances, t_low, t_high, d_unfamiliar):
    calls = np.array([
        assign_call(float(score), float(distance), t_low, t_high, d_unfamiliar)
        for score, distance in zip(scores, distances)
    ])
    positive = labels == 1
    negative = ~positive
    very = calls == "very_unlikely"
    likely = calls == "likely"
    return {
        "n": int(len(calls)),
        "n_positive": int(positive.sum()),
        "n_negative": int(negative.sum()),
        "n_likely": int((calls == "likely").sum()),
        "n_uncertain": int((calls == "uncertain").sum()),
        "n_very_unlikely": int(very.sum()),
        "n_unfamiliar": int((calls == "unfamiliar").sum()),
        "false_clean_rate": float(very[positive].mean()) if positive.any() else float("nan"),
        "clean_specificity": float(very[negative].mean()) if negative.any() else float("nan"),
        "likely_precision": float(labels[likely].mean()) if likely.any() else float("nan"),
    }, calls


def write_summary(path, rows):
    fields = [
        "split", "n", "n_positive", "n_negative", "n_likely", "n_uncertain",
        "n_very_unlikely", "n_unfamiliar", "false_clean_rate", "clean_specificity", "likely_precision",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data" / "features"))
    parser.add_argument("--model", default=str(ROOT / "data" / "models" / "dual_encoder.pt"))
    parser.add_argument("--out", default=str(ROOT / "data" / "models"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    labels = np.load(data / "labels.npz")
    observed = labels["observed"]
    split = labels["split"]
    molecule_ids = labels["ids"]
    panels = labels["panels"]
    ligands = GraphStore(data / "ligands.npz")
    pockets = GraphStore(data / "pockets.npz")
    n_molecules = len(split)
    if args.limit:
        chosen = []
        for code in (0, 1, 2):
            pool = np.where(split == code)[0][:args.limit]
            chosen.append(pool)
        indices = np.concatenate(chosen)
    else:
        indices = np.arange(n_molecules)

    checkpoint = torch.load(args.model, map_location=device, weights_only=True)
    model = DualEncoder().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print("device %s molecules %d checkpoint_epoch %s" % (
        device, len(indices), checkpoint.get("epoch"),
    ), flush=True)

    with torch.no_grad():
        pocket_embedding = encode_pockets(model, pockets, device).cpu().numpy()
        scale = float(model.logit_scale.exp().clamp(max=100))
    ligand_embedding = encode_ligands(model, ligands, indices, device, args.batch_size)
    similarity = pocket_scores(ligand_embedding, pocket_embedding, scale)
    max_score = similarity.max(axis=1)
    top_panel = similarity.argmax(axis=1).astype(np.int16)
    panel_positive = ((observed[indices] == 1).sum(axis=1) > 0).astype(np.int8)

    row_of = {int(index): row for row, index in enumerate(indices)}
    train_index = np.array([row_of[int(index)] for index in np.where(split == 0)[0] if int(index) in row_of])
    valid_index = np.array([row_of[int(index)] for index in np.where(split == 1)[0] if int(index) in row_of])
    test_index = np.array([row_of[int(index)] for index in np.where(split == 2)[0] if int(index) in row_of])
    library_row = np.full(len(indices), -1, dtype=np.int32)
    library_row[train_index] = np.arange(len(train_index), dtype=np.int32)

    global_index = indices
    valid_pair_labels, valid_pair_scores = pair_arrays(observed, similarity[valid_index], global_index[valid_index])
    test_pair_labels, test_pair_scores = pair_arrays(observed, similarity[test_index], global_index[test_index])
    valid_pair = auroc(valid_pair_labels, valid_pair_scores)
    valid_panel = auroc(panel_positive[valid_index], max_score[valid_index])
    test_pair = auroc(test_pair_labels, test_pair_scores)
    test_panel = auroc(panel_positive[test_index], max_score[test_index])
    print("valid pair %.4f panel %.4f" % (valid_pair, valid_panel), flush=True)
    print("test pair %.4f panel %.4f" % (test_pair, test_panel), flush=True)
    if not args.limit:
        expected = float(checkpoint["valid_pair_auroc"])
        if abs(valid_pair - expected) > 0.01:
            raise SystemExit("validation pair AUROC %.4f disagrees with checkpoint %.4f" % (valid_pair, expected))

    distance = np.empty(len(indices), dtype=np.float32)
    distance[train_index] = 1.0 - nearest_cosine(
        ligand_embedding[train_index], ligand_embedding[train_index], device, library_row[train_index],
    )
    distance[valid_index] = 1.0 - nearest_cosine(
        ligand_embedding[valid_index], ligand_embedding[train_index], device,
    )
    distance[test_index] = 1.0 - nearest_cosine(
        ligand_embedding[test_index], ligand_embedding[train_index], device,
    )

    np.savez_compressed(
        out / "ligand_embeddings.npz",
        ids=molecule_ids[indices],
        global_index=indices.astype(np.int32),
        split=split[indices],
        panel_positive=panel_positive,
        score=max_score.astype(np.float32),
        top_panel=top_panel,
        nearest_distance=distance,
        embedding=ligand_embedding.astype(np.float32),
    )
    (out / "test_metrics.csv").write_text(
        "pair_auroc,panel_auroc\n%.4f,%.4f\n" % (test_pair, test_panel)
    )

    valid_scores = max_score[valid_index]
    valid_labels = panel_positive[valid_index]
    positive_scores = valid_scores[valid_labels == 1]
    t_low = float(np.quantile(positive_scores, 1.0 - CLEAN_SENSITIVITY))
    min_likely = MIN_LIKELY if len(valid_scores) >= MIN_LIKELY else max(1, len(valid_scores) // 5)
    likely = None
    precision_target = LIKELY_PRECISION
    for target in (0.95, 0.90, 0.85):
        likely = most_inclusive_threshold(valid_scores, valid_labels, target, min_likely)
        precision_target = target
        if likely is not None:
            break
    if likely is None:
        raise SystemExit("no likely threshold covered %d validation molecules" % MIN_LIKELY)
    t_high, likely_precision, likely_count = likely
    if t_high < t_low:
        t_high = t_low
        chosen = valid_scores >= t_high
        likely_precision = float(valid_labels[chosen].mean())
        likely_count = int(chosen.sum())
    d_unfamiliar = float(np.quantile(distance[valid_index], OOD_QUANTILE))

    rules = [
        ("clean_sensitivity_target", "%.2f" % CLEAN_SENSITIVITY),
        ("t_low", "%.6f" % t_low),
        ("t_low_rule", "5th percentile of validation scores among molecules potent on at least one panel target"),
        ("likely_precision_target", "%.2f" % precision_target),
        ("t_high", "%.6f" % t_high),
        ("t_high_validation_precision", "%.4f" % likely_precision),
        ("t_high_validation_count", str(likely_count)),
        ("t_high_rule", "widest validation threshold whose precision is at least the recorded target"),
        ("ood_quantile", "%.2f" % OOD_QUANTILE),
        ("unfamiliar_distance", "%.6f" % d_unfamiliar),
        ("unfamiliar_rule", "nearest-training-ligand distance above the 99th percentile of the validation set"),
        ("checkpoint_epoch", str(int(checkpoint["epoch"]))),
        ("valid_pair_auroc", "%.4f" % valid_pair),
        ("valid_panel_auroc", "%.4f" % valid_panel),
        ("test_pair_auroc", "%.4f" % test_pair),
        ("test_panel_auroc", "%.4f" % test_panel),
    ]
    with (out / "panel_cutoff.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["parameter", "value"])
        writer.writerows(rules)

    summary_rows = []
    call_rows = []
    for name, rows in (("valid", valid_index), ("test", test_index)):
        stats, calls = operating_point(
            max_score[rows], panel_positive[rows], distance[rows], t_low, t_high, d_unfamiliar,
        )
        stats["split"] = name
        for key in ("false_clean_rate", "clean_specificity", "likely_precision"):
            stats[key] = "%.4f" % stats[key]
        summary_rows.append(stats)
        for row, call in zip(rows, calls):
            call_rows.append({
                "parent_molecule_chembl_id": molecule_ids[indices[row]],
                "split": name,
                "panel_positive": int(panel_positive[row]),
                "score": "%.4f" % max_score[row],
                "top_panel": panels[int(top_panel[row])],
                "nearest_train_distance": "%.4f" % distance[row],
                "call": call,
            })
        print(
            "%s likely %s uncertain %s very_unlikely %s unfamiliar %s false_clean %s"
            % (name, stats["n_likely"], stats["n_uncertain"], stats["n_very_unlikely"], stats["n_unfamiliar"], stats["false_clean_rate"]),
            flush=True,
        )
    write_summary(out / "panel_cutoff_summary.csv", summary_rows)
    with (out / "panel_calls.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "parent_molecule_chembl_id", "split", "panel_positive", "score", "top_panel",
            "nearest_train_distance", "call",
        ])
        writer.writeheader()
        writer.writerows(call_rows)
    print("wrote %s" % out, flush=True)


if __name__ == "__main__":
    main()
