#!/usr/bin/env python3
"""Fit the four-way cutoff on the fingerprint score.

The score is the highest ECFP4 Tanimoto to a training molecule that is potent
on a panel target. Thresholds come from the validation molecules only. The
unfamiliar distance is one minus the highest Tanimoto to any training molecule.
Nothing is written into data/models/panel_cutoff.csv.
"""

from __future__ import annotations

import csv
import multiprocessing as mp
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

ROOT = Path(__file__).resolve().parents[1]
RDLogger.DisableLog("rdApp.*")
CLEAN_SENSITIVITY = 0.95
LIKELY_PRECISION = 0.95
MIN_LIKELY = 100
OOD_QUANTILE = 0.99


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


def nearest_slice(bounds):
    start, end = bounds
    distance = np.ones(end - start, dtype=np.float32)
    for offset, molecule in enumerate(QUERY_ROWS[start:end]):
        query = FINGERPRINTS[int(molecule)]
        if query is None:
            continue
        best = 0.0
        for library_start in range(0, len(TRAIN_FPS), 20000):
            block = TRAIN_FPS[library_start:library_start + 20000]
            best = max(best, max(DataStructs.BulkTanimotoSimilarity(query, block)))
        distance[offset] = 1.0 - best
    print("nearest %d/%d" % (end, len(QUERY_ROWS)), flush=True)
    return start, distance


def pack_bits(fingerprint):
    array = np.zeros((2048,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fingerprint, array)
    return np.packbits(array)


def main():
    global FINGERPRINTS, TRAIN_FPS, QUERY_ROWS
    out = ROOT / "data" / "models" / "fingerprint_cutoff"
    if out.resolve() == (ROOT / "data" / "models").resolve():
        raise SystemExit("refusing to write into data/models")
    out.mkdir(parents=True, exist_ok=True)
    labels = np.load(ROOT / "data" / "features" / "labels.npz", allow_pickle=True)
    observed = labels["observed"]
    split = labels["split"]
    panels = np.array([str(panel) for panel in labels["panels"]])
    heldout = np.load(ROOT / "data" / "models" / "baselines" / "fingerprint_heldout.npz")
    valid_rows = heldout["valid_index"]
    test_rows = heldout["test_index"]
    if not np.array_equal(valid_rows, np.where(split == 1)[0]):
        raise SystemExit("validation rows do not match the split")
    if not np.array_equal(test_rows, np.where(split == 2)[0]):
        raise SystemExit("test rows do not match the split")
    smiles = {
        row["parent_molecule_chembl_id"]: row["canonical_smiles"]
        for row in csv.DictReader(open(ROOT / "data" / "modeling" / "compounds.csv"))
    }
    names = {
        row["panel_id"]: row["gene_symbols"]
        for row in csv.DictReader(open(ROOT / "data" / "modeling" / "targets.csv"))
    }
    print("fingerprints", flush=True)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    FINGERPRINTS = []
    missing = 0
    for molecule_id in labels["ids"]:
        molecule = Chem.MolFromSmiles(smiles.get(str(molecule_id), ""))
        if molecule is None:
            FINGERPRINTS.append(None)
            missing += 1
        else:
            FINGERPRINTS.append(generator.GetFingerprint(molecule))
    train_rows = np.where(split == 0)[0]
    TRAIN_FPS = [FINGERPRINTS[int(row)] for row in train_rows if FINGERPRINTS[int(row)] is not None]
    usable_train = np.array([int(row) for row in train_rows if FINGERPRINTS[int(row)] is not None], dtype=np.int32)
    QUERY_ROWS = np.concatenate([valid_rows, test_rows])
    print("missing %d train %d queries %d" % (missing, len(TRAIN_FPS), len(QUERY_ROWS)), flush=True)
    bounds = []
    width = int(np.ceil(len(QUERY_ROWS) / 6))
    for start in range(0, len(QUERY_ROWS), width):
        bounds.append((start, min(start + width, len(QUERY_ROWS))))
    distance = np.ones(len(QUERY_ROWS), dtype=np.float32)
    with mp.get_context("fork").Pool(6) as pool:
        for start, values in pool.imap_unordered(nearest_slice, bounds):
            distance[start:start + len(values)] = values
    valid_distance = distance[:len(valid_rows)]
    test_distance = distance[len(valid_rows):]
    valid_target_scores = heldout["valid_scores"]
    test_target_scores = heldout["test_scores"]
    valid_score = valid_target_scores.max(axis=1)
    test_score = test_target_scores.max(axis=1)
    valid_positive = ((observed[valid_rows] == 1).sum(axis=1) > 0).astype(np.int32)
    test_positive = ((observed[test_rows] == 1).sum(axis=1) > 0).astype(np.int32)
    positive_scores = valid_score[valid_positive == 1]
    t_low = float(np.quantile(positive_scores, 1.0 - CLEAN_SENSITIVITY))
    likely = None
    precision_target = LIKELY_PRECISION
    for target in (0.95, 0.90, 0.85):
        likely = most_inclusive_threshold(valid_score, valid_positive, target, MIN_LIKELY)
        precision_target = target
        if likely is not None:
            break
    if likely is None:
        raise SystemExit("no likely threshold covered %d validation molecules" % MIN_LIKELY)
    t_high, likely_precision, likely_count = likely
    if t_high < t_low:
        t_high = t_low
        chosen = valid_score >= t_high
        likely_precision = float(valid_positive[chosen].mean())
        likely_count = int(chosen.sum())
    d_unfamiliar = float(np.quantile(valid_distance, OOD_QUANTILE))
    valid_pair = auroc(
        observed[valid_rows][np.isfinite(observed[valid_rows])].astype(np.int32),
        valid_target_scores[np.isfinite(observed[valid_rows])],
    )
    test_pair = auroc(
        observed[test_rows][np.isfinite(observed[test_rows])].astype(np.int32),
        test_target_scores[np.isfinite(observed[test_rows])],
    )
    valid_panel = auroc(valid_positive, valid_score)
    test_panel = auroc(test_positive, test_score)
    rules = [
        ("score", "max ECFP4 Tanimoto to training potent ligands"),
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
        ("unfamiliar_rule", "one minus the nearest training-fingerprint Tanimoto, above the 99th percentile of the validation set"),
        ("valid_pair_auroc", "%.4f" % valid_pair),
        ("valid_panel_auroc", "%.4f" % valid_panel),
        ("test_pair_auroc", "%.4f" % test_pair),
        ("test_panel_auroc", "%.4f" % test_panel),
    ]
    with (out / "panel_cutoff.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["parameter", "value"])
        writer.writerows(rules)
    summary_fields = [
        "split", "n", "n_positive", "n_negative", "n_likely", "n_uncertain",
        "n_very_unlikely", "n_unfamiliar", "false_clean_rate", "clean_specificity", "likely_precision",
    ]
    call_fields = [
        "parent_molecule_chembl_id", "split", "panel_positive", "score", "top_panel",
        "nearest_train_distance", "call",
    ]
    with (out / "panel_cutoff_summary.csv").open("w", newline="") as summary_handle, (
        out / "panel_calls.csv"
    ).open("w", newline="") as call_handle:
        summary_writer = csv.DictWriter(summary_handle, fieldnames=summary_fields)
        summary_writer.writeheader()
        call_writer = csv.DictWriter(call_handle, fieldnames=call_fields)
        call_writer.writeheader()
        for name, scores, target_scores, positive, distances, rows in (
            ("valid", valid_score, valid_target_scores, valid_positive, valid_distance, valid_rows),
            ("test", test_score, test_target_scores, test_positive, test_distance, test_rows),
        ):
            calls = [
                assign_call(float(score), float(dist), t_low, t_high, d_unfamiliar)
                for score, dist in zip(scores, distances)
            ]
            calls = np.array(calls)
            positive_mask = positive == 1
            negative_mask = ~positive_mask
            very = calls == "very_unlikely"
            likely_mask = calls == "likely"
            stats = {
                "split": name,
                "n": int(len(calls)),
                "n_positive": int(positive_mask.sum()),
                "n_negative": int(negative_mask.sum()),
                "n_likely": int(likely_mask.sum()),
                "n_uncertain": int((calls == "uncertain").sum()),
                "n_very_unlikely": int(very.sum()),
                "n_unfamiliar": int((calls == "unfamiliar").sum()),
                "false_clean_rate": "%.4f" % (float(very[positive_mask].mean()) if positive_mask.any() else float("nan")),
                "clean_specificity": "%.4f" % (float(very[negative_mask].mean()) if negative_mask.any() else float("nan")),
                "likely_precision": "%.4f" % (float(positive[likely_mask].mean()) if likely_mask.any() else float("nan")),
            }
            summary_writer.writerow(stats)
            top_panel = target_scores.argmax(axis=1)
            ids = labels["ids"]
            for index, call in enumerate(calls):
                call_writer.writerow({
                    "parent_molecule_chembl_id": ids[int(rows[index])],
                    "split": name,
                    "panel_positive": int(positive[index]),
                    "score": "%.4f" % scores[index],
                    "top_panel": panels[int(top_panel[index])],
                    "nearest_train_distance": "%.4f" % distances[index],
                    "call": call,
                })
            print(
                "%s likely %s uncertain %s very_unlikely %s unfamiliar %s false_clean %s" % (
                    name, stats["n_likely"], stats["n_uncertain"], stats["n_very_unlikely"],
                    stats["n_unfamiliar"], stats["false_clean_rate"],
                ),
                flush=True,
            )
    bits = np.stack([pack_bits(FINGERPRINTS[int(row)]) for row in usable_train])
    local_of = {int(global_index): local for local, global_index in enumerate(usable_train)}
    potent_local = []
    potent_ptr = [0]
    for target in range(observed.shape[1]):
        members = train_rows[observed[train_rows, target] == 1]
        for member in members:
            if int(member) in local_of:
                potent_local.append(local_of[int(member)])
        potent_ptr.append(len(potent_local))
    np.savez_compressed(
        out / "reference.npz",
        bits=bits,
        train_index=usable_train,
        potent_ptr=np.array(potent_ptr, dtype=np.int32),
        potent_local=np.array(potent_local, dtype=np.int32),
        panels=panels,
        genes=np.array([names.get(panel, "") for panel in panels]),
        t_low=np.array([t_low]),
        t_high=np.array([t_high]),
        unfamiliar_distance=np.array([d_unfamiliar]),
    )
    print(
        "t_low %.4f t_high %.4f unfamiliar %.4f test panel %.4f" % (t_low, t_high, d_unfamiliar, test_panel),
        flush=True,
    )
    print("wrote %s" % out, flush=True)


if __name__ == "__main__":
    main()
