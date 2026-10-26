#!/usr/bin/env python3
"""Fingerprint baselines on the held-out scaffolds.

Reference ligands are training molecules potent on each target. Queries are
the test split. Max ECFP4 Tanimoto is the 2D similarity. The SEA score is a z-score of
the summed similarity above a background threshold.
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


def topk_hit_rate(scores, potent, k=5):
    has = potent.any(axis=1)
    chosen = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    hits = np.array([potent[i, chosen[i]].any() for i in np.where(has)[0]])
    return float(hits.mean()), int(has.sum())


def pair_metrics(scores, observed, rows):
    target = observed[rows]
    mask = np.isfinite(target)
    return auroc(target[mask].astype(np.int32), scores[mask])


def load_fingerprints(ids, smiles):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = []
    missing = 0
    for molecule_id in ids:
        molecule = Chem.MolFromSmiles(smiles.get(str(molecule_id), ""))
        if molecule is None:
            fingerprints.append(None)
            missing += 1
        else:
            fingerprints.append(generator.GetFingerprint(molecule))
    return fingerprints, missing


def background(fingerprints, train_rows, n_pairs=100000):
    rng = np.random.default_rng(42)
    usable = [int(i) for i in train_rows if fingerprints[int(i)] is not None]
    left = rng.integers(0, len(usable), n_pairs)
    right = rng.integers(0, len(usable), n_pairs)
    values = []
    for a, b in zip(left, right):
        if a == b:
            continue
        values.append(DataStructs.TanimotoSimilarity(fingerprints[usable[a]], fingerprints[usable[b]]))
    values = np.asarray(values, dtype=np.float64)
    threshold = float(np.quantile(values, 0.99))
    excess = values[values >= threshold] - threshold
    if len(excess) < 20:
        threshold = float(np.quantile(values, 0.95))
        excess = values[values >= threshold] - threshold
    probability = float((values >= threshold).mean())
    mean = float(excess.mean()) if len(excess) else 0.0
    second = float(np.mean(excess ** 2)) if len(excess) else 0.0
    variance = max(probability * second - (probability * mean) ** 2, 1e-12)
    return threshold, probability, mean, variance, values


def score_target(target):
    references = [FINGERPRINTS[i] for i in POTENT[target] if FINGERPRINTS[i] is not None]
    n_ref = len(references)
    max_score = np.zeros(len(TEST_ROWS), dtype=np.float32)
    sea_score = np.zeros(len(TEST_ROWS), dtype=np.float32)
    if n_ref == 0:
        return target, max_score, sea_score
    expect = n_ref * PROBABILITY * MEAN
    scale = np.sqrt(max(n_ref * VARIANCE, 1e-12))
    for row, molecule in enumerate(TEST_ROWS):
        query = FINGERPRINTS[molecule]
        if query is None:
            continue
        similarities = DataStructs.BulkTanimotoSimilarity(query, references)
        max_score[row] = max(similarities)
        raw = 0.0
        for value in similarities:
            if value >= THRESHOLD:
                raw += value - THRESHOLD
        sea_score[row] = (raw - expect) / scale
    print("target %d refs %d" % (target, n_ref), flush=True)
    return target, max_score, sea_score


def main():
    global FINGERPRINTS, POTENT, TEST_ROWS, THRESHOLD, PROBABILITY, MEAN, VARIANCE
    labels = np.load(ROOT / "data" / "features" / "labels.npz", allow_pickle=True)
    observed = labels["observed"]
    split = labels["split"]
    ids = labels["ids"]
    smiles = {
        row["parent_molecule_chembl_id"]: row["canonical_smiles"]
        for row in csv.DictReader(open(ROOT / "data" / "modeling" / "compounds.csv"))
    }
    print("fingerprints", flush=True)
    FINGERPRINTS, missing = load_fingerprints(ids, smiles)
    train_rows = np.where(split == 0)[0]
    TEST_ROWS = np.where(split == 2)[0]
    POTENT = []
    for target in range(observed.shape[1]):
        rows = train_rows[observed[train_rows, target] == 1]
        POTENT.append(rows.astype(np.int32))
    THRESHOLD, PROBABILITY, MEAN, VARIANCE, values = background(FINGERPRINTS, train_rows)
    print(
        "missing %d background median %.3f p99 threshold %.3f" % (missing, np.median(values), THRESHOLD),
        flush=True,
    )
    out = ROOT / "data" / "models" / "baselines"
    out.mkdir(parents=True, exist_ok=True)
    max_scores = np.zeros((len(TEST_ROWS), observed.shape[1]), dtype=np.float32)
    sea_scores = np.zeros_like(max_scores)
    workers = 6
    with mp.get_context("fork").Pool(workers) as pool:
        for target, max_score, sea_score in pool.imap_unordered(score_target, range(observed.shape[1])):
            max_scores[:, target] = max_score
            sea_scores[:, target] = sea_score
    potent = observed[TEST_ROWS] == 1
    np.savez_compressed(
        out / "fingerprint_scores.npz",
        test_index=TEST_ROWS.astype(np.int32),
        max_similarity=max_scores,
        sea_z=sea_scores,
        threshold=np.array([THRESHOLD]),
    )
    for name, scores in (("max_similarity", max_scores), ("sea", sea_scores)):
        pair = pair_metrics(scores, observed, TEST_ROWS)
        panel = auroc((potent.sum(axis=1) > 0).astype(np.int32), scores.max(axis=1))
        top5, n_mol = topk_hit_rate(scores, potent, 5)
        print("%s pair %.4f panel %.4f top5 %.4f n %d" % (name, pair, panel, top5, n_mol), flush=True)
        (out / ("%s_metrics.csv" % name)).write_text(
            "pair_auroc,panel_auroc,top5_hit_rate,n_molecules\n%.4f,%.4f,%.4f,%d\n" % (pair, panel, top5, n_mol)
        )


if __name__ == "__main__":
    main()
