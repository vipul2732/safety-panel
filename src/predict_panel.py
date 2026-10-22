#!/usr/bin/env python3
"""Rank one molecule over the 71 safety targets and apply the fingerprint cutoff."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

ROOT = Path(__file__).resolve().parents[1]
RDLogger.DisableLog("rdApp.*")


def tanimoto_matrix(query_bits, library_bits):
    scores = np.empty(len(library_bits), dtype=np.float32)
    query = np.unpackbits(query_bits).astype(np.float32)
    query_count = float(query.sum())
    for start in range(0, len(library_bits), 8192):
        block = np.unpackbits(library_bits[start:start + 8192], axis=1).astype(np.float32)
        dot = block @ query
        union = block.sum(axis=1) + query_count - dot
        scores[start:start + len(block)] = dot / np.maximum(union, 1.0)
    return scores


def fingerprint_bits(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise SystemExit("could not read SMILES")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    array = np.zeros((2048,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(generator.GetFingerprint(molecule), array)
    return np.packbits(array)


def assign_call(score, distance, t_low, t_high, d_unfamiliar):
    if distance > d_unfamiliar and score < t_low:
        return "unfamiliar"
    if score >= t_high:
        return "likely"
    if score < t_low:
        return "very_unlikely"
    return "uncertain"


def predict(smiles, reference):
    query = fingerprint_bits(smiles)
    library = reference["bits"]
    similarity = tanimoto_matrix(query, library)
    target_scores = np.zeros(len(reference["panels"]), dtype=np.float32)
    pointer = reference["potent_ptr"]
    members = reference["potent_local"]
    for target in range(len(target_scores)):
        chosen = members[int(pointer[target]):int(pointer[target + 1])]
        if len(chosen):
            target_scores[target] = similarity[chosen].max()
    score = float(target_scores.max())
    distance = float(1.0 - similarity.max())
    call = assign_call(
        score, distance,
        float(reference["t_low"][0]),
        float(reference["t_high"][0]),
        float(reference["unfamiliar_distance"][0]),
    )
    order = np.argsort(-target_scores)
    return target_scores, order, score, distance, call


def load_catalog():
    names = {}
    classes = {}
    path = ROOT / "data" / "modeling" / "targets.csv"
    with path.open() as handle:
        for row in csv.DictReader(handle):
            names[row["panel_id"]] = row["target_name"]
            classes[row["panel_id"]] = row["target_class"]
    return names, classes


def read_queries(smiles, path):
    if smiles and path:
        raise SystemExit("use either --smiles or --input")
    if smiles:
        return [("1", smiles)]
    if not path:
        raise SystemExit("give a SMILES with --smiles, or a file with --input")
    queries = []
    for line_number, raw in enumerate(Path(path).read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) == 1:
            queries.append((str(line_number), parts[0]))
        else:
            queries.append((parts[0], parts[1]))
    if not queries:
        raise SystemExit("no SMILES rows in %s" % path)
    return queries


def load_ligand():
    try:
        import torch
        from model import LigandOnlyEncoder
    except ImportError as exc:
        raise SystemExit("the ligand model needs PyTorch as well as RDKit (%s)" % exc)
    device = torch.device("cpu")
    checkpoint_path = ROOT / "data" / "models" / "ligand_only" / "ligand_only.pt"
    calibration_path = ROOT / "data" / "models" / "ligand_only" / "potent_calibration.npz"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = LigandOnlyEncoder(int(checkpoint["n_targets"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    calibration = np.load(calibration_path)
    return model, device, float(calibration["intercept"][0]), float(calibration["slope"][0])


def ligand_logits(smiles, model, device):
    import torch
    from featurize_panel import featurize_ligand

    _index, _name, graph = featurize_ligand((0, "query", smiles))
    if graph is None:
        raise SystemExit("could not build a structure from the SMILES")
    features, positions, edges, _coord_is_3d = graph
    with torch.no_grad():
        embedding = model.encode_ligand(
            torch.from_numpy(features).to(device),
            torch.from_numpy(positions).to(device),
            torch.from_numpy(np.array(edges)).long().to(device),
            torch.zeros(len(features), dtype=torch.long, device=device),
        )
        similarity = model.similarity(embedding, model.target_embedding())
    return similarity.squeeze(0).cpu().numpy()


def potent_chance(logits, intercept, slope):
    return 1.0 / (1.0 + np.exp(-np.clip(intercept + slope * logits, -30, 30)))


def ranked_rows(smiles, reference, names, classes, top, chance):
    scores, order, score, distance, call = predict(smiles, reference)
    rows = []
    for rank, target in enumerate(order[:top], start=1):
        panel = str(reference["panels"][target])
        rows.append({
            "call": call,
            "panel_score": score,
            "distance": distance,
            "rank": rank,
            "panel": panel,
            "gene": str(reference["genes"][target]),
            "name": names.get(panel, panel),
            "class": classes.get(panel, ""),
            "similarity": float(scores[target]),
            "potent_chance": float(chance[target]),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Rank a molecule on the 71-target safety panel.")
    parser.add_argument("--smiles", help="one SMILES string")
    parser.add_argument("--input", help="text file: SMILES, or id and SMILES, one molecule per line")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--output", help="write a CSV of the ranked targets")
    parser.add_argument("--reference", default=str(ROOT / "data" / "models" / "fingerprint_cutoff" / "reference.npz"))
    args = parser.parse_args()
    reference = np.load(args.reference, allow_pickle=True)
    names, classes = load_catalog()
    model, device, intercept, slope = load_ligand()
    queries = read_queries(args.smiles, args.input)
    table = []
    failures = 0
    print("Ranked list: fingerprint similarity to known potent ligands.")
    print("Chance: ligand model, the probability the interaction is potent rather than a close weak analog.")
    print("")
    for query_id, smiles in queries:
        try:
            chance = potent_chance(ligand_logits(smiles, model, device), intercept, slope)
            rows = ranked_rows(smiles, reference, names, classes, args.top, chance)
        except SystemExit as exc:
            failures += 1
            message = exc.code if isinstance(exc.code, str) else "could not read SMILES"
            print("%s  %s" % (query_id, message))
            continue
        for row in rows:
            row["id"] = query_id
            row["smiles"] = smiles
            table.append(row)
        first = rows[0]
        print("%s  %s  score %.3f  distance %.3f" % (query_id, first["call"], first["panel_score"], first["distance"]))
        print("Top target %s (%s): potent chance %.2f" % (first["name"], first["gene"], first["potent_chance"]))
        print("rank  similarity  chance  gene      class              target")
        for row in rows:
            print("%4d  %10.3f  %6.2f  %-8s  %-17s  %s" % (
                row["rank"], row["similarity"], row["potent_chance"], row["gene"], row["class"], row["name"],
            ))
        if len(queries) > 1:
            print("")
    if args.output:
        fields = ["id", "smiles", "call", "panel_score", "distance", "rank", "panel", "gene", "name", "class", "similarity", "potent_chance"]
        with open(args.output, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in table:
                saved = dict(row)
                saved["panel_score"] = "%.4f" % row["panel_score"]
                saved["distance"] = "%.4f" % row["distance"]
                saved["similarity"] = "%.4f" % row["similarity"]
                saved["potent_chance"] = "%.4f" % row["potent_chance"]
                writer.writerow(saved)
        print("wrote %s" % args.output)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
