#!/usr/bin/env python3
"""ECFP4 max similarity for the validation and test molecules.

Reference ligands are the training potent molecules, the same sets used for
the held-out fingerprint baseline. SEA is not recomputed.
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


def score_target(target):
    references = [FINGERPRINTS[i] for i in POTENT[target] if FINGERPRINTS[i] is not None]
    max_score = np.zeros(len(QUERY_ROWS), dtype=np.float32)
    if not references:
        return target, max_score
    for row, molecule in enumerate(QUERY_ROWS):
        query = FINGERPRINTS[molecule]
        if query is None:
            continue
        max_score[row] = max(DataStructs.BulkTanimotoSimilarity(query, references))
    print("target %d refs %d" % (target, len(references)), flush=True)
    return target, max_score


def main():
    global FINGERPRINTS, POTENT, QUERY_ROWS
    labels = np.load(ROOT / "data" / "features" / "labels.npz", allow_pickle=True)
    observed = labels["observed"]
    split = labels["split"]
    smiles = {
        row["parent_molecule_chembl_id"]: row["canonical_smiles"]
        for row in csv.DictReader(open(ROOT / "data" / "modeling" / "compounds.csv"))
    }
    print("fingerprints", flush=True)
    FINGERPRINTS, missing = load_fingerprints(labels["ids"], smiles)
    train_rows = np.where(split == 0)[0]
    valid_rows = np.where(split == 1)[0]
    test_rows = np.where(split == 2)[0]
    QUERY_ROWS = np.concatenate([valid_rows, test_rows])
    POTENT = []
    for target in range(observed.shape[1]):
        POTENT.append(train_rows[observed[train_rows, target] == 1].astype(np.int32))
    print("missing %d queries %d" % (missing, len(QUERY_ROWS)), flush=True)
    scores = np.zeros((len(QUERY_ROWS), observed.shape[1]), dtype=np.float32)
    with mp.get_context("fork").Pool(6) as pool:
        for target, max_score in pool.imap_unordered(score_target, range(observed.shape[1])):
            scores[:, target] = max_score
    out = ROOT / "data" / "models" / "baselines" / "fingerprint_heldout.npz"
    np.savez_compressed(
        out,
        valid_index=valid_rows.astype(np.int32),
        test_index=test_rows.astype(np.int32),
        valid_scores=scores[:len(valid_rows)],
        test_scores=scores[len(valid_rows):],
    )
    print("wrote %s" % out, flush=True)


if __name__ == "__main__":
    main()
