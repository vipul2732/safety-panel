#!/usr/bin/env python3
"""Held-out approved drugs, and the far subset, for both ranking methods.

Far means the fingerprint score is below 0.40. That line was set before
looking at which method wins. Drugs in the training or validation split are
counted and then left out of the scores.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FAR_SCORE = 0.40


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


def metrics(scores, observed):
    mask = np.isfinite(observed)
    pair = auroc(observed[mask].astype(np.int32), scores[mask])
    potent = observed == 1
    panel = auroc(potent.any(axis=1).astype(np.int32), scores.max(axis=1))
    has = potent.any(axis=1)
    if int(has.sum()) == 0:
        return pair, panel, float("nan"), 0
    chosen = np.argpartition(-scores, kth=min(4, scores.shape[1] - 1), axis=1)[:, :5]
    hits = np.array([potent[i, chosen[i]].any() for i in np.where(has)[0]])
    return pair, panel, float(hits.mean()), int(has.sum())


def main():
    labels = np.load(ROOT / "data" / "features" / "labels.npz", allow_pickle=True)
    observed = labels["observed"]
    split = labels["split"]
    ids = np.array([str(molecule_id) for molecule_id in labels["ids"]])
    approved = {
        row["molecule_chembl_id"]
        for row in csv.DictReader(open(ROOT / "data" / "external" / "chembl_approved.csv"))
    }
    fingerprints = np.load(ROOT / "data" / "models" / "baselines" / "fingerprint_heldout.npz")
    gentle = np.load(ROOT / "data" / "models" / "baselines" / "gentle_scores.npz")
    if not np.array_equal(fingerprints["test_index"], gentle["test_index"]):
        raise SystemExit("test rows differ")
    test_rows = fingerprints["test_index"]
    where = {molecule_id: index for index, molecule_id in enumerate(ids)}
    counts = {"train": 0, "valid": 0, "test": 0, "outside": 0}
    for molecule_id in approved:
        index = where.get(molecule_id)
        if index is None:
            counts["outside"] += 1
        else:
            counts[{"0": "train", "1": "valid", "2": "test"}[str(int(split[index]))]] += 1
    test_ids = set(ids[test_rows])
    chosen = np.array([molecule_id in approved for molecule_id in ids[test_rows]])
    fp_scores = fingerprints["test_scores"]
    pocket_scores = gentle["test_scores"]
    far = chosen & (fp_scores.max(axis=1) < FAR_SCORE)
    out = ROOT / "data" / "external"
    lines = [
        "subset,n_molecules,n_panel_positive,pair_auroc,panel_auroc,top5_hit_rate,method",
    ]
    for name, mask in (("approved_test", chosen), ("approved_test_far", far)):
        block_observed = observed[test_rows][mask]
        for method, scores in (("fingerprint", fp_scores[mask]), ("pocket_finetune", pocket_scores[mask])):
            pair, panel, top5, n_positive = metrics(scores, block_observed)
            lines.append("%s,%d,%d,%.4f,%.4f,%.4f,%s" % (
                name, int(mask.sum()), n_positive, pair, panel, top5, method,
            ))
    text = "\n".join(lines) + "\n"
    (out / "approved_heldout.csv").write_text(text)
    print(
        "approved %d train %d valid %d test %d outside %d far %d" % (
            len(approved), counts["train"], counts["valid"], counts["test"], counts["outside"], int(far.sum()),
        ),
        flush=True,
    )
    print(text, flush=True)


if __name__ == "__main__":
    main()
