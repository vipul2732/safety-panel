#!/usr/bin/env python3
"""Compare molecule-dependent priors against the ECFP4 ranking.

Every coefficient is fit on training molecules. Variants are judged on the
validation split. The test split is scored once, after that choice, and is
not used to pick a variant.

The target base-rate offset is not a term. Per-target intercepts are fit only
to keep the slopes honest, then dropped from the score.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdFingerprintGenerator

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "models" / "priors"
FAR_SCORE = 0.40
CLASS_ALPHA = 0.25
PAIR_MIN = 20
SHRINK_N = 50.0
RDLogger.DisableLog("rdApp.*")
BASIC = Chem.MolFromSmarts("[#7;X3;H2,H1,H0;!$(*~[#8,#16]=O);!$(n);!$(N-a)]")


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
    target = observed
    mask = np.isfinite(target)
    pair = auroc(target[mask].astype(np.int32), scores[mask])
    potent = target == 1
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


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.clip(values, -20, 20)))


def irls(features, labels, anchor, anchor_l2):
    count, width = features.shape
    design = np.column_stack([np.ones(count), features])
    weights = np.zeros(width + 1) if anchor is None else np.asarray(anchor, dtype=np.float64).copy()
    penalty = np.full(width + 1, anchor_l2)
    penalty[0] = anchor_l2 * 0.05
    center = np.zeros(width + 1) if anchor is None else np.asarray(anchor, dtype=np.float64)
    for _ in range(40):
        eta = np.clip(design @ weights, -20, 20)
        probability = sigmoid(eta)
        variance = np.maximum(probability * (1.0 - probability), 1e-4)
        grad = design.T @ (probability - labels) + penalty * (weights - center)
        hessian = (design * variance[:, None]).T @ design
        hessian.flat[:: width + 2] += penalty
        try:
            step = np.linalg.solve(hessian, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, grad, rcond=None)[0]
        weights -= step
        if np.max(np.abs(step)) < 1e-7:
            break
    return weights


def pack_morgan(smiles):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    packed = []
    for text in smiles:
        molecule = Chem.MolFromSmiles(text) if text else None
        if molecule is None:
            packed.append(np.zeros(256, dtype=np.uint8))
            continue
        array = np.zeros((2048,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(molecule), array)
        packed.append(np.packbits(array))
    return np.stack(packed)


def max_tanimoto(query, library, query_local=None, library_local=None):
    """Max ECFP4 Tanimoto of each packed query against a packed library."""
    scores = np.zeros(len(query), dtype=np.float32)
    if len(library) == 0 or len(query) == 0:
        return scores
    lib = np.unpackbits(library, axis=1).astype(np.float32)
    lib_sum = lib.sum(axis=1)
    exclude = None
    if query_local is not None:
        position = {int(local): index for index, local in enumerate(library_local)}
        exclude = np.array([position.get(int(local), -1) for local in query_local], dtype=np.int32)
    for start in range(0, len(query), 512):
        stop = min(start + 512, len(query))
        block = np.unpackbits(query[start:stop], axis=1).astype(np.float32)
        dots = block @ lib.T
        union = block.sum(axis=1)[:, None] + lib_sum[None, :] - dots
        similarity = dots / np.maximum(union, 1.0)
        if exclude is not None:
            for row, column in enumerate(exclude[start:stop]):
                if column >= 0:
                    similarity[row, column] = -1.0
        scores[start:stop] = np.maximum(similarity.max(axis=1), 0.0)
    return scores


def properties(smiles):
    rows = np.zeros((len(smiles), 6), dtype=np.float32)
    for index, text in enumerate(smiles):
        molecule = Chem.MolFromSmiles(text) if text else None
        if molecule is None:
            continue
        rows[index] = (
            Descriptors.MolWt(molecule),
            Crippen.MolLogP(molecule),
            Descriptors.TPSA(molecule),
            float(Descriptors.NumHDonors(molecule)),
            float(Descriptors.NumHAcceptors(molecule)),
            float(len(molecule.GetSubstructMatches(BASIC))),
        )
    return rows


def class_log_odds(observed, train_rows, classes):
    matrix = np.zeros((observed.shape[1], observed.shape[1]), dtype=np.float64)
    block = observed[train_rows]
    for target in range(observed.shape[1]):
        partners = np.where(classes == classes[target])[0]
        measured_t = np.isfinite(block[:, target])
        for other in partners:
            if other == target:
                continue
            both_measured = measured_t & np.isfinite(block[:, other])
            count = int(both_measured.sum())
            if count < PAIR_MIN:
                continue
            left = block[both_measured, target] == 1
            right = block[both_measured, other] == 1
            both = float((left & right).sum())
            left_only = float((left & ~right).sum())
            right_only = float((~left & right).sum())
            neither = float((~left & ~right).sum())
            odds = ((both + 0.5) * (neither + 0.5)) / ((left_only + 0.5) * (right_only + 0.5))
            matrix[target, other] = (count / (count + SHRINK_N)) * np.log(odds)
    return matrix


def apply_class_log_odds(scores, log_odds):
    strength = log_odds / (1.0 + np.abs(log_odds))
    extra = np.zeros_like(scores)
    for target in range(scores.shape[1]):
        partners = np.flatnonzero(np.abs(strength[target]) > 0)
        if len(partners) == 0:
            continue
        gap = scores[:, partners] - scores[:, target][:, None]
        extra[:, target] = np.mean(strength[target, partners] * gap, axis=1)
    return scores + CLASS_ALPHA * extra


def apply_class_mean(scores, classes):
    mixed = scores.copy()
    for name in np.unique(classes):
        members = np.where(classes == name)[0]
        if len(members) < 2:
            continue
        class_mean = scores[:, members].mean(axis=1)[:, None]
        mixed[:, members] = (1.0 - CLASS_ALPHA) * scores[:, members] + CLASS_ALPHA * class_mean
    return mixed


def load_smiles(ids):
    wanted = {str(molecule_id) for molecule_id in ids}
    found = {}
    with open(ROOT / "data" / "modeling" / "compounds.csv") as handle:
        for row in csv.DictReader(handle):
            molecule_id = row["parent_molecule_chembl_id"]
            if molecule_id in wanted and molecule_id not in found:
                found[molecule_id] = row["canonical_smiles"]
    return [found.get(str(molecule_id), "") for molecule_id in ids]


def main():
    if BASIC is None:
        raise SystemExit("basic-nitrogen pattern did not parse")
    OUT.mkdir(parents=True, exist_ok=True)
    labels = np.load(ROOT / "data" / "features" / "labels.npz", allow_pickle=True)
    observed = labels["observed"]
    split = labels["split"]
    panels = np.array([str(panel) for panel in labels["panels"]])
    train_rows = np.where(split == 0)[0]
    reference = np.load(ROOT / "data" / "models" / "fingerprint_cutoff" / "reference.npz", allow_pickle=True)
    if not np.array_equal(reference["train_index"], train_rows):
        raise SystemExit("reference fingerprints are not in training-row order")
    heldout = np.load(ROOT / "data" / "models" / "baselines" / "fingerprint_heldout.npz")
    valid_rows = heldout["valid_index"]
    test_rows = heldout["test_index"]
    class_of = {}
    with open(ROOT / "data" / "modeling" / "targets.csv") as handle:
        for row in csv.DictReader(handle):
            class_of[row["panel_id"]] = row["target_class"]
    classes = np.array([class_of[panel] for panel in panels])
    bits = reference["bits"]
    local_of = np.full(len(split), -1, dtype=np.int32)
    local_of[train_rows] = np.arange(len(train_rows), dtype=np.int32)

    print("descriptors", flush=True)
    smiles = load_smiles(labels["ids"])
    raw_properties = properties(smiles)
    train_properties = raw_properties[train_rows]
    center = train_properties.mean(axis=0)
    spread = train_properties.std(axis=0)
    spread[spread < 1e-6] = 1.0
    scaled = ((raw_properties - center) / spread).astype(np.float32)

    print("query fingerprints", flush=True)
    query_rows = np.concatenate([valid_rows, test_rows])
    query_bits = pack_morgan([smiles[int(row)] for row in query_rows])

    print("weak similarities", flush=True)
    weak_scores = np.zeros((len(query_rows), observed.shape[1]), dtype=np.float32)
    train_features = []
    for target in range(observed.shape[1]):
        train_block = observed[train_rows, target]
        potent_local = local_of[train_rows[train_block == 1]]
        weak_local = local_of[train_rows[train_block == 0]]
        weak_scores[:, target] = max_tanimoto(query_bits, bits[weak_local])
        measured = train_rows[np.isfinite(observed[train_rows, target])]
        potent_similarity = max_tanimoto(
            bits[local_of[measured]], bits[potent_local], local_of[measured], potent_local,
        )
        weak_similarity = max_tanimoto(
            bits[local_of[measured]], bits[weak_local], local_of[measured], weak_local,
        )
        train_features.append((
            target,
            measured,
            (observed[measured, target] == 1).astype(np.float64),
            potent_similarity.astype(np.float64),
            weak_similarity.astype(np.float64),
            scaled[measured],
        ))
        print("target %d measured %d weaks %d" % (target, len(measured), len(weak_local)), flush=True)

    print("fit", flush=True)
    pooled_x = np.concatenate([
        np.column_stack([item[3], item[4]]) for item in train_features
    ])
    pooled_y = np.concatenate([item[2] for item in train_features])
    pooled_fp = irls(pooled_x, pooled_y, anchor=None, anchor_l2=1.0)
    pooled_property_x = np.concatenate([item[5] for item in train_features])
    pooled_property = irls(pooled_property_x, pooled_y, anchor=None, anchor_l2=5.0)
    fingerprint_weights = np.zeros((observed.shape[1], 3), dtype=np.float64)
    property_weights = np.zeros((observed.shape[1], 7), dtype=np.float64)
    for target, _measured, labels_t, potent_s, weak_s, features in train_features:
        positives = int(labels_t.sum())
        negatives = int(len(labels_t) - positives)
        if positives < 10 or negatives < 5:
            fingerprint_weights[target] = pooled_fp
            property_weights[target] = pooled_property
            continue
        fingerprint_weights[target] = irls(
            np.column_stack([potent_s, weak_s]), labels_t, pooled_fp, anchor_l2=20.0,
        )
        property_weights[target] = irls(features, labels_t, pooled_property, anchor_l2=20.0)
        if fingerprint_weights[target, 1] <= 0:
            fingerprint_weights[target] = pooled_fp

    slopes = fingerprint_weights[:, 1]
    weak_slopes = fingerprint_weights[:, 2]
    property_slopes = property_weights[:, 1:]
    gamma = np.where(slopes > 0, weak_slopes / slopes, pooled_fp[2] / pooled_fp[1])
    print(
        "pooled potent slope %.3f weak slope %.3f gamma %.3f" % (
            pooled_fp[1], pooled_fp[2], pooled_fp[2] / pooled_fp[1],
        ),
        flush=True,
    )
    log_odds = class_log_odds(observed, train_rows, classes)

    valid_fp = heldout["valid_scores"].astype(np.float64)
    test_fp = heldout["test_scores"].astype(np.float64)
    valid_weak = weak_scores[:len(valid_rows)].astype(np.float64)
    test_weak = weak_scores[len(valid_rows):].astype(np.float64)
    valid_far = valid_fp.max(axis=1) < FAR_SCORE
    test_far = test_fp.max(axis=1) < FAR_SCORE
    print("validation far %d/%d" % (int(valid_far.sum()), len(valid_far)), flush=True)

    property_cap = 0.15

    def weak_score(fingerprint, weak):
        return fingerprint + gamma * weak

    def with_property(base, rows, far):
        scored = base.copy()
        if far.any():
            delta = scaled[rows][far] @ property_slopes.T / pooled_fp[1]
            scored[far] = scored[far] + np.clip(delta, -property_cap, property_cap)
        return scored

    variants = {}
    for name, fp, weak, rows, far in (
        ("valid", valid_fp, valid_weak, valid_rows, valid_far),
        ("test", test_fp, test_weak, test_rows, test_far),
    ):
        weak_only = weak_score(fp, weak)
        property_on_fp = with_property(fp, rows, far)
        property_on_weak = with_property(weak_only, rows, far)
        variants[(name, "fingerprint")] = fp
        variants[(name, "weak")] = weak_only
        variants[(name, "class")] = apply_class_log_odds(fp, log_odds)
        variants[(name, "class_mean")] = apply_class_mean(fp, classes)
        variants[(name, "property")] = property_on_fp
        variants[(name, "weak_class")] = apply_class_log_odds(weak_only, log_odds)
        variants[(name, "weak_property")] = property_on_weak
        variants[(name, "class_property")] = apply_class_log_odds(property_on_fp, log_odds)
        variants[(name, "all_three")] = apply_class_log_odds(property_on_weak, log_odds)

    fields = [
        "variant", "split", "pair_auroc", "panel_auroc",
        "top5_hit_rate", "top1_hit_rate", "median_rank", "n_molecules",
    ]
    lines = [",".join(fields)]
    records = []
    for split_name, rows in (("valid", valid_rows), ("test", test_rows)):
        for variant in (
            "fingerprint", "weak", "class", "class_mean", "property",
            "weak_class", "weak_property", "class_property", "all_three",
        ):
            metrics = ranking_metrics(variants[(split_name, variant)], observed[rows])
            values = [variant, split_name] + ["%.4f" % value for value in metrics[:-1]] + [str(metrics[-1])]
            lines.append(",".join(values))
            records.append((variant, split_name, metrics))
            print(split_name, variant, "panel %.4f top5 %.4f top1 %.4f" % (metrics[1], metrics[2], metrics[3]), flush=True)
    text = "\n".join(lines) + "\n"
    (OUT / "comparison.csv").write_text(text)

    baseline = next(metrics for variant, split_name, metrics in records if variant == "fingerprint" and split_name == "valid")
    candidates = [
        (variant, metrics)
        for variant, split_name, metrics in records
        if split_name == "valid" and variant != "fingerprint"
    ]
    eligible = [
        (variant, metrics)
        for variant, metrics in candidates
        if metrics[2] >= baseline[2] - 0.002 and metrics[1] > baseline[1]
    ]
    if eligible:
        winner = max(eligible, key=lambda item: (item[1][1], item[1][3]))
        reason = "validation panel AUROC improved and top-5 stayed within 0.002 of the fingerprint"
    else:
        winner = ("none", baseline)
        reason = "no variant raised the validation panel AUROC without giving up more than 0.002 top-5"
    (OUT / "choice.txt").write_text(
        "winner %s\nreason %s\nbaseline_valid_panel %.4f baseline_valid_top5 %.4f\n"
        "winner_valid_panel %.4f winner_valid_top5 %.4f\n" % (
            winner[0], reason, baseline[1], baseline[2], winner[1][1], winner[1][2],
        )
    )
    print("winner", winner[0], reason, flush=True)
    print("wrote", OUT / "comparison.csv", flush=True)


if __name__ == "__main__":
    main()
