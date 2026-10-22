#!/usr/bin/env python3
"""Laplacian-modified naive Bayes, and a Bayes affinity fingerprint.

One Bayes model is fit per target on that target's measured training labels.
Potent is the positive class and weak is the negative class. Unmeasured pairs
are left out. The molecule score is the sum of the bit weights, the same
Laplacian correction used by the multiple-category models of Nidhi et al.

The affinity fingerprint is the vector of those 71 scores. A target score is
the highest Pearson correlation with a training molecule that is potent on
that target. Training profiles used as the reference are leave-one-out scores.

Coefficients come from the training split. The validation split decides.
The test split is scored once.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "models" / "bayes"
RDLogger.DisableLog("rdApp.*")


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
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    sum_pos = ranks[labels == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def ranking_metrics(scores, observed):
    mask = np.isfinite(observed)
    pair = auroc(observed[mask].astype(np.int32), scores[mask])
    potent = observed == 1
    has = potent.any(axis=1)
    panel = auroc(has.astype(np.int32), scores.max(axis=1))
    order = np.argsort(-scores, axis=1, kind="mergesort")
    ranks = np.empty_like(order)
    ranks[np.arange(len(scores))[:, None], order] = np.arange(1, scores.shape[1] + 1)
    true_rank = np.where(potent, ranks, np.inf).min(axis=1)
    top5 = float((true_rank[has] <= 5).mean()) if has.any() else float("nan")
    top1 = float((true_rank[has] <= 1).mean()) if has.any() else float("nan")
    median = float(np.median(true_rank[has])) if has.any() else float("nan")
    return pair, panel, top5, top1, median, int(has.sum())


def pack_morgan(smiles):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    packed = np.zeros((len(smiles), 256), dtype=np.uint8)
    for index, text in enumerate(smiles):
        molecule = Chem.MolFromSmiles(text) if text else None
        if molecule is None:
            continue
        array = np.zeros((2048,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(molecule), array)
        packed[index] = np.packbits(array)
    return packed


def load_smiles(ids):
    wanted = {str(molecule_id) for molecule_id in ids}
    found = {}
    with open(ROOT / "data" / "modeling" / "compounds.csv") as handle:
        for row in csv.DictReader(handle):
            molecule_id = row["parent_molecule_chembl_id"]
            if molecule_id in wanted and molecule_id not in found:
                found[molecule_id] = row["canonical_smiles"]
    return [found.get(str(molecule_id), "") for molecule_id in ids]


def bit_weights(bits, labels):
    """Laplacian weight per ECFP bit. labels are 1 for potent and 0 for weak."""
    potent = labels == 1
    count = len(labels)
    n_potent = int(potent.sum())
    prior = n_potent / count
    active_bits = bits[potent].sum(axis=0).astype(np.float64)
    total_bits = bits.sum(axis=0).astype(np.float64)
    weights = np.log((active_bits + prior) / ((total_bits + 1.0) * prior))
    return weights, prior, n_potent, count


def loo_scores(bits, labels):
    """Leave-one-out Bayes score for each measured training molecule."""
    count = len(labels)
    n_potent = int((labels == 1).sum())
    active_bits = bits[labels == 1].sum(axis=0).astype(np.float64)
    total_bits = bits.sum(axis=0).astype(np.float64)
    scores = np.zeros(count, dtype=np.float64)
    for is_potent, prior_loo in (
        (True, (n_potent - 1) / (count - 1)),
        (False, n_potent / (count - 1)),
    ):
        chosen = labels == (1 if is_potent else 0)
        if not np.any(chosen) or prior_loo <= 0:
            continue
        removed = 1.0 if is_potent else 0.0
        present = total_bits > 0
        weights = np.zeros(bits.shape[1], dtype=np.float64)
        weights[present] = np.log(
            (np.maximum(active_bits[present] - removed, 0.0) + prior_loo)
            / (total_bits[present] * prior_loo)
        )
        scores[chosen] = bits[chosen] @ weights
    return scores


def max_correlation(query, library, block=2048):
    centered_library = library - library.mean(axis=1, keepdims=True)
    library_norm = np.linalg.norm(centered_library, axis=1)
    keep = library_norm > 1e-8
    centered_library = centered_library[keep]
    library_norm = library_norm[keep]
    scores = np.zeros(len(query), dtype=np.float32)
    if len(centered_library) == 0:
        return scores
    centered_query = query - query.mean(axis=1, keepdims=True)
    query_norm = np.linalg.norm(centered_query, axis=1)
    usable = query_norm > 1e-8
    for start in range(0, len(query), block):
        stop = min(start + block, len(query))
        chosen = usable[start:stop]
        if not np.any(chosen):
            continue
        dots = centered_query[start:stop][chosen] @ centered_library.T
        correlation = dots / (query_norm[start:stop][chosen][:, None] * library_norm[None, :])
        scores[start:stop][chosen] = correlation.max(axis=1)
    return scores


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    labels = np.load(ROOT / "data" / "features" / "labels.npz", allow_pickle=True)
    observed = labels["observed"]
    split = labels["split"]
    train_rows = np.where(split == 0)[0]
    reference = np.load(ROOT / "data" / "models" / "fingerprint_cutoff" / "reference.npz")
    if not np.array_equal(reference["train_index"], train_rows):
        raise SystemExit("reference fingerprints are not in training-row order")
    heldout = np.load(ROOT / "data" / "models" / "baselines" / "fingerprint_heldout.npz")
    valid_rows = heldout["valid_index"]
    test_rows = heldout["test_index"]
    print("unpack training fingerprints", flush=True)
    train_bits = np.unpackbits(reference["bits"], axis=1)
    n_targets = observed.shape[1]
    weights = np.zeros((n_targets, train_bits.shape[1]), dtype=np.float64)
    profiles = np.zeros((len(train_rows), n_targets), dtype=np.float32)
    print("fit", flush=True)
    for target in range(n_targets):
        measured = np.isfinite(observed[train_rows, target])
        bits = train_bits[measured]
        target_labels = (observed[train_rows[measured], target] == 1).astype(np.int8)
        weights[target], prior, n_potent, count = bit_weights(bits, target_labels)
        corrected = loo_scores(bits, target_labels)
        profiles[np.flatnonzero(measured), target] = corrected
        outside = np.flatnonzero(~measured)
        if len(outside):
            profiles[outside, target] = train_bits[outside] @ weights[target]
        print(
            "target %d measured %d potent %d prior %.3f" % (target, count, n_potent, prior),
            flush=True,
        )

    print("score held-out molecules", flush=True)
    smiles = load_smiles(labels["ids"])
    query_rows = np.concatenate([valid_rows, test_rows])
    query_bits = np.unpackbits(pack_morgan([smiles[int(row)] for row in query_rows]), axis=1)
    bayes = query_bits @ weights.T
    valid_bayes = bayes[:len(valid_rows)]
    test_bayes = bayes[len(valid_rows):]

    print("affinity correlations", flush=True)
    affinity = np.zeros_like(bayes)
    train_observed = observed[train_rows]
    for target in range(n_targets):
        library = profiles[train_observed[:, target] == 1]
        affinity[:, target] = max_correlation(bayes, library)
        print("affinity %d library %d" % (target, len(library)), flush=True)
    valid_affinity = affinity[:len(valid_rows)]
    test_affinity = affinity[len(valid_rows):]

    methods = {
        "fingerprint": (heldout["valid_scores"], heldout["test_scores"]),
        "naive_bayes": (valid_bayes, test_bayes),
        "bayes_affinity": (valid_affinity, test_affinity),
    }
    fields = [
        "variant", "split", "pair_auroc", "panel_auroc",
        "top5_hit_rate", "top1_hit_rate", "median_rank", "n_molecules",
    ]
    lines = [",".join(fields)]
    records = []
    for split_name, rows, column in (
        ("valid", valid_rows, 0),
        ("test", test_rows, 1),
    ):
        for variant, scores in methods.items():
            metrics = ranking_metrics(scores[column], observed[rows])
            values = [variant, split_name] + ["%.4f" % value for value in metrics[:-1]] + [str(metrics[-1])]
            lines.append(",".join(values))
            records.append((variant, split_name, metrics))
            print(
                split_name, variant,
                "pair %.4f panel %.4f top5 %.4f top1 %.4f" % metrics[:4],
                flush=True,
            )
    (OUT / "comparison.csv").write_text("\n".join(lines) + "\n")
    np.savez_compressed(
        OUT / "scores.npz",
        valid_index=valid_rows,
        test_index=test_rows,
        valid_bayes=valid_bayes.astype(np.float32),
        test_bayes=test_bayes.astype(np.float32),
        valid_affinity=valid_affinity.astype(np.float32),
        test_affinity=test_affinity.astype(np.float32),
    )
    baseline = next(metrics for variant, split_name, metrics in records if variant == "fingerprint" and split_name == "valid")
    candidates = [
        (variant, metrics)
        for variant, split_name, metrics in records
        if split_name == "valid" and variant != "fingerprint"
    ]
    eligible = [
        item for item in candidates
        if item[1][2] >= baseline[2] - 0.002 and item[1][1] > baseline[1]
    ]
    if eligible:
        winner, winner_metrics = max(eligible, key=lambda item: (item[1][1], item[1][2]))
        reason = "validation panel AUROC improved and top-5 stayed within 0.002 of the fingerprint"
    else:
        winner, winner_metrics = "none", baseline
        reason = "no variant raised the validation panel AUROC without giving up more than 0.002 top-5"
    (OUT / "choice.txt").write_text(
        "adopted %s\nreason %s\nbaseline_valid_panel %.4f baseline_valid_top5 %.4f\n"
        "winner_valid_panel %.4f winner_valid_top5 %.4f\n" % (
            winner, reason, baseline[1], baseline[2], winner_metrics[1], winner_metrics[2],
        )
    )
    print("adopted", winner, reason, flush=True)


if __name__ == "__main__":
    main()
