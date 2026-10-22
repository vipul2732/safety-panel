#!/usr/bin/env python3
"""Shift pass-15 scores by a per-target testing prior and refit the cutoff.

The prior for a target is the fraction of its training measurements that are
potent. The model similarity is treated as a logit at the global training
rate, then moved to that target's own rate. Thresholds are fit on validation
molecules only.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from model import DualEncoder
from score_panel import (
    CLEAN_SENSITIVITY,
    LIKELY_PRECISION,
    MIN_LIKELY,
    OOD_QUANTILE,
    most_inclusive_threshold,
    operating_point,
    pocket_scores,
    write_summary,
)
from train_panel import GraphStore, auroc, encode_pockets

ROOT = Path(__file__).resolve().parents[1]


def target_rates(observed, train_rows):
    measured = observed[train_rows]
    potent = np.nansum(measured == 1, axis=0)
    weak = np.nansum(measured == 0, axis=0)
    rate = (potent + 0.5) / (potent + weak + 1.0)
    return potent, weak, rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data" / "features"))
    parser.add_argument("--model", default=str(ROOT / "data" / "models" / "dual_encoder.pt"))
    parser.add_argument("--embeddings", default=str(ROOT / "data" / "models" / "ligand_embeddings.npz"))
    parser.add_argument("--out", default=str(ROOT / "data" / "models"))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    labels = np.load(Path(args.data) / "labels.npz")
    observed = labels["observed"]
    split = labels["split"]
    panels = labels["panels"]
    stored = np.load(args.embeddings)
    if not np.array_equal(stored["global_index"], np.arange(len(split))):
        raise SystemExit("saved embeddings are not in molecule order")

    checkpoint = torch.load(args.model, map_location=device, weights_only=True)
    model = DualEncoder().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    pockets = GraphStore(Path(args.data) / "pockets.npz")
    with torch.no_grad():
        pocket_embedding = encode_pockets(model, pockets, device).cpu().numpy()
        scale = float(model.logit_scale.exp().clamp(max=100))
    similarity = pocket_scores(stored["embedding"], pocket_embedding, scale)
    raw_max = similarity.max(axis=1)

    train_rows = np.where(split == 0)[0]
    valid_rows = np.where(split == 1)[0]
    test_rows = np.where(split == 2)[0]
    potent, weak, rate = target_rates(observed, train_rows)
    global_rate = float((potent.sum() + 0.5) / (potent.sum() + weak.sum() + 1.0))
    offset = np.log(rate / (1.0 - rate)) - np.log(global_rate / (1.0 - global_rate))
    adjusted = similarity + offset.astype(np.float32)
    adjusted_max = adjusted.max(axis=1)
    panel_positive = ((observed == 1).sum(axis=1) > 0).astype(np.int8)
    distance = stored["nearest_distance"]

    print("checkpoint epoch %s global training potent rate %.4f" % (
        int(checkpoint["epoch"]), global_rate,
    ), flush=True)
    print("offset min %.3f median %.3f max %.3f" % (
        offset.min(), np.median(offset), offset.max(),
    ), flush=True)
    for name, rows in (("valid", valid_rows), ("test", test_rows)):
        print("%s raw panel %.4f prior panel %.4f" % (
            name,
            auroc(panel_positive[rows], raw_max[rows]),
            auroc(panel_positive[rows], adjusted_max[rows]),
        ), flush=True)

    valid_scores = adjusted_max[valid_rows]
    valid_labels = panel_positive[valid_rows]
    t_low = float(np.quantile(valid_scores[valid_labels == 1], 1.0 - CLEAN_SENSITIVITY))
    likely = None
    precision_target = LIKELY_PRECISION
    for target in (0.95, 0.90, 0.85):
        likely = most_inclusive_threshold(valid_scores, valid_labels, target, MIN_LIKELY)
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
    d_unfamiliar = float(np.quantile(distance[valid_rows], OOD_QUANTILE))

    with (out / "target_prior.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["panel_id", "train_potent", "train_weak", "potent_rate", "logit_offset"])
        for index, panel in enumerate(panels):
            writer.writerow([
                panel, int(potent[index]), int(weak[index]),
                "%.4f" % rate[index], "%.4f" % offset[index],
            ])
    with (out / "panel_prior.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["parameter", "value"])
        writer.writerows([
            ("checkpoint_epoch", str(int(checkpoint["epoch"]))),
            ("global_training_potent_rate", "%.4f" % global_rate),
            ("t_low", "%.6f" % t_low),
            ("t_high", "%.6f" % t_high),
            ("likely_precision_target", "%.2f" % precision_target),
            ("t_high_validation_precision", "%.4f" % likely_precision),
            ("t_high_validation_count", str(likely_count)),
            ("unfamiliar_distance", "%.6f" % d_unfamiliar),
            ("valid_panel_auroc_raw", "%.4f" % auroc(panel_positive[valid_rows], raw_max[valid_rows])),
            ("valid_panel_auroc_prior", "%.4f" % auroc(panel_positive[valid_rows], adjusted_max[valid_rows])),
            ("test_panel_auroc_raw", "%.4f" % auroc(panel_positive[test_rows], raw_max[test_rows])),
            ("test_panel_auroc_prior", "%.4f" % auroc(panel_positive[test_rows], adjusted_max[test_rows])),
        ])

    summary_rows = []
    for name, rows in (("valid", valid_rows), ("test", test_rows)):
        stats, _calls = operating_point(
            adjusted_max[rows], panel_positive[rows], distance[rows], t_low, t_high, d_unfamiliar,
        )
        stats["split"] = name
        for key in ("false_clean_rate", "clean_specificity", "likely_precision"):
            stats[key] = "%.4f" % stats[key]
        summary_rows.append(stats)
        print(
            "%s likely %s uncertain %s very_unlikely %s unfamiliar %s false_clean %s"
            % (name, stats["n_likely"], stats["n_uncertain"], stats["n_very_unlikely"], stats["n_unfamiliar"], stats["false_clean_rate"]),
            flush=True,
        )
    write_summary(out / "panel_prior_summary.csv", summary_rows)


if __name__ == "__main__":
    main()
