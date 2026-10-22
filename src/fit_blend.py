#!/usr/bin/env python3
"""One linear blend of the fine-tuned pocket scores and ECFP4 max similarity.

The weight is chosen on the validation molecules. A weight is eligible when
both the panel call and the top 5 beat the fingerprint on validation. Among
eligible weights, the best top 5 is kept. That one weight is applied to the
held-out molecules once.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


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
    target = observed
    mask = np.isfinite(target)
    pair = auroc(target[mask].astype(np.int32), scores[mask])
    potent = target == 1
    panel = auroc(potent.any(axis=1).astype(np.int32), scores.max(axis=1))
    has = potent.any(axis=1)
    chosen = np.argpartition(-scores, kth=4, axis=1)[:, :5]
    hits = np.array([potent[i, chosen[i]].any() for i in np.where(has)[0]])
    return pair, panel, float(hits.mean())


def standardize(train, apply_to):
    """One scale for every target, so each method's ranking stays intact."""
    center = float(train.mean())
    spread = float(train.std())
    if spread < 1e-6:
        spread = 1.0
    return (apply_to - center) / spread


def blend(weight, pocket, fingerprint):
    return weight * pocket + (1.0 - weight) * fingerprint


def main():
    labels = np.load(ROOT / "data" / "features" / "labels.npz")
    observed = labels["observed"]
    gentle = np.load(ROOT / "data" / "models" / "baselines" / "gentle_scores.npz")
    fingerprints = np.load(ROOT / "data" / "models" / "baselines" / "fingerprint_heldout.npz")
    if not np.array_equal(gentle["valid_index"], fingerprints["valid_index"]):
        raise SystemExit("validation rows differ")
    if not np.array_equal(gentle["test_index"], fingerprints["test_index"]):
        raise SystemExit("test rows differ")
    valid_rows = gentle["valid_index"]
    test_rows = gentle["test_index"]
    pocket_valid = standardize(gentle["valid_scores"], gentle["valid_scores"])
    pocket_test = standardize(gentle["valid_scores"], gentle["test_scores"])
    fp_valid = standardize(fingerprints["valid_scores"], fingerprints["valid_scores"])
    fp_test = standardize(fingerprints["valid_scores"], fingerprints["test_scores"])
    valid_observed = observed[valid_rows]
    test_observed = observed[test_rows]
    out = ROOT / "data" / "models" / "baselines"
    curve = ["weight,valid_pair_auroc,valid_panel_auroc,valid_top5"]
    rows = []
    for step in range(0, 21):
        weight = step / 20.0
        pair, panel, top5 = metrics(blend(weight, pocket_valid, fp_valid), valid_observed)
        curve.append("%.2f,%.4f,%.4f,%.4f" % (weight, pair, panel, top5))
        rows.append((weight, pair, panel, top5))
    (out / "blend_validation.csv").write_text("\n".join(curve) + "\n")
    fingerprint = rows[0]
    eligible = [row for row in rows if row[2] > fingerprint[2] and row[3] > fingerprint[3]]
    if eligible:
        chosen = sorted(eligible, key=lambda row: (row[3], row[2], -row[0]))[-1]
    else:
        chosen = fingerprint
    test_pair, test_panel, test_top5 = metrics(
        blend(chosen[0], pocket_test, fp_test), test_observed,
    )
    fingerprint_test = metrics(fp_test, test_observed)
    adopted = int(test_panel > fingerprint_test[1] and test_top5 > fingerprint_test[2] and chosen[0] > 0)
    (out / "blend_choice.csv").write_text(
        "weight,adopted,valid_pair_auroc,valid_panel_auroc,valid_top5,"
        "test_pair_auroc,test_panel_auroc,test_top5,"
        "fingerprint_test_pair,fingerprint_test_panel,fingerprint_test_top5\n"
        "%.2f,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n" % (
            chosen[0], adopted, chosen[1], chosen[2], chosen[3],
            test_pair, test_panel, test_top5,
            fingerprint_test[0], fingerprint_test[1], fingerprint_test[2],
        )
    )
    print(
        "chosen weight %.2f adopted %d valid panel %.4f top5 %.4f test pair %.4f panel %.4f top5 %.4f" % (
            chosen[0], adopted, chosen[2], chosen[3], test_pair, test_panel, test_top5,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
