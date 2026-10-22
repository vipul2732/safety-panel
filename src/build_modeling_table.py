#!/usr/bin/env python3
"""Freeze the modeling table for the 71 safety-panel targets.

Supervised labels are potent (max pChEMBL >= 6) and weak (max pChEMBL < 5).
Intermediate measurements are stored and withheld from both classes. A pair
with no row was not measured; that is the unknown class, and it is not an
inactive.

Molecules are grouped by Bemis-Murcko scaffold and assigned as whole groups
to train, valid, or test. The same molecule split is the one a ligand-only
baseline must use.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import Chem
from rdkit import __version__ as RDKIT_VERSION
from rdkit.Chem.Scaffolds import MurckoScaffold

ROOT = Path(__file__).resolve().parents[1]
FEASIBILITY = ROOT / "data" / "results" / "target_feasibility.csv"
LIGAND_DIR = ROOT / "data" / "ligands"
OUT_DIR = ROOT / "data" / "modeling"
SPLIT_FRACTIONS = (("train", 0.8), ("valid", 0.1), ("test", 0.1))

TARGET_FIELDS = [
    "panel_id", "gene_symbols", "target_name", "target_class", "in_bowes44",
    "in_vendor77", "uniprot_accessions", "chembl_target_id", "representative_pdb",
    "representative_resolution", "representative_ligand",
]
COMPOUND_FIELDS = [
    "parent_molecule_chembl_id", "canonical_smiles", "scaffold_smiles", "split",
    "n_potent_targets", "n_weak_targets", "n_intermediate_targets", "n_smiles_variants",
]
LABEL_FIELDS = [
    "parent_molecule_chembl_id", "panel_id", "bin", "max_pchembl", "n_measurements",
    "standard_type_at_max", "assay_type_at_max",
]


def qualifying_targets():
    with FEASIBILITY.open() as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row["meets_ligand_floor"] == "1" and row["has_human_druglike_structure"] == "1"
        ]
    rows.sort(key=lambda row: row["panel_id"])
    return rows


def load_labels(panel_ids):
    labels = []
    smiles_counts = defaultdict(Counter)
    for panel_id in panel_ids:
        path = LIGAND_DIR / ("%s.csv" % panel_id)
        with path.open() as handle:
            for row in csv.DictReader(handle):
                parent = row["parent_molecule_chembl_id"]
                smiles = row["canonical_smiles"]
                smiles_counts[parent][smiles] += 1
                labels.append({
                    "parent_molecule_chembl_id": parent,
                    "panel_id": panel_id,
                    "bin": row["bin"],
                    "max_pchembl": row["max_pchembl"],
                    "n_measurements": row["n_measurements"],
                    "standard_type_at_max": row["standard_type_at_max"],
                    "assay_type_at_max": row["assay_type_at_max"],
                })
    return labels, smiles_counts


def canonicalize(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None, None
    canonical = Chem.MolToSmiles(molecule, canonical=True)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=False)
    if not scaffold:
        scaffold = "ACYCLIC:" + canonical
    return canonical, scaffold


def choose_structure(variant_counts):
    canon_counts = Counter()
    representatives = {}
    for smiles, count in variant_counts.items():
        canonical, scaffold = canonicalize(smiles)
        if canonical is None:
            continue
        canon_counts[canonical] += count
        representatives[canonical] = scaffold
    if not canon_counts:
        raw = variant_counts.most_common(1)[0][0]
        return raw, "INVALID:" + raw, len(variant_counts)
    canonical = sorted(canon_counts, key=lambda item: (-canon_counts[item], item))[0]
    return canonical, representatives[canonical], len(variant_counts)


def assign_splits(compounds):
    groups = defaultdict(list)
    for compound in compounds:
        groups[compound["scaffold_smiles"]].append(compound)
    total = len(compounds)
    capacity = {name: fraction * total for name, fraction in SPLIT_FRACTIONS}
    used = {name: 0 for name, _fraction in SPLIT_FRACTIONS}
    ordered = sorted(groups, key=lambda scaffold: (-len(groups[scaffold]), scaffold))
    for scaffold in ordered:
        members = groups[scaffold]
        split = max(capacity, key=lambda name: (capacity[name] - used[name], name))
        for compound in members:
            compound["split"] = split
        used[split] += len(members)


def target_counts(labels):
    counts = defaultdict(lambda: Counter())
    for label in labels:
        counts[label["parent_molecule_chembl_id"]][label["bin"]] += 1
    return counts


def main():
    targets = qualifying_targets()
    panel_ids = [row["panel_id"] for row in targets]
    labels, smiles_counts = load_labels(panel_ids)
    per_parent = target_counts(labels)
    compounds = []
    for parent, variant_counts in smiles_counts.items():
        canonical, scaffold, n_variants = choose_structure(variant_counts)
        bins = per_parent[parent]
        compounds.append({
            "parent_molecule_chembl_id": parent,
            "canonical_smiles": canonical,
            "scaffold_smiles": scaffold,
            "split": "",
            "n_potent_targets": bins["potent"],
            "n_weak_targets": bins["weak"],
            "n_intermediate_targets": bins["intermediate"],
            "n_smiles_variants": n_variants,
        })
    compounds.sort(key=lambda row: row["parent_molecule_chembl_id"])
    assign_splits(compounds)
    labels.sort(key=lambda row: (row["parent_molecule_chembl_id"], row["panel_id"]))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "targets.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TARGET_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(targets)
    with (OUT_DIR / "compounds.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPOUND_FIELDS)
        writer.writeheader()
        writer.writerows(compounds)
    with (OUT_DIR / "labels.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_FIELDS)
        writer.writeheader()
        writer.writerows(labels)

    split_counts = Counter(row["split"] for row in compounds)
    bin_counts = Counter(row["bin"] for row in labels)
    scaffolds_by_split = defaultdict(set)
    for row in compounds:
        scaffolds_by_split[row["split"]].add(row["scaffold_smiles"])
    overlap = set.intersection(*scaffolds_by_split.values()) if len(scaffolds_by_split) == 3 else set()
    supervised_parents = {
        row["parent_molecule_chembl_id"] for row in labels if row["bin"] in ("potent", "weak")
    }
    provenance = {
        "rdkit_version": RDKIT_VERSION,
        "n_targets": len(targets),
        "n_compounds": len(compounds),
        "n_compounds_with_supervised_label": len(supervised_parents),
        "n_scaffolds": len({row["scaffold_smiles"] for row in compounds}),
        "split_fractions": {"train": 0.8, "valid": 0.1, "test": 0.1},
        "split_counts": dict(split_counts),
        "label_counts": dict(bin_counts),
        "scaffold_overlap_across_splits": len(overlap),
        "inclusion_rule": "meets_ligand_floor = 1 and has_human_druglike_structure = 1",
        "potent": "max pChEMBL >= 6",
        "weak": "max pChEMBL < 5",
        "intermediate": "5 <= max pChEMBL < 6; stored and withheld from both supervised classes",
        "unknown": "no row in labels.csv for that compound-target pair; not an inactive",
        "split": "Whole Bemis-Murcko scaffolds are assigned to the split with the most remaining molecule capacity. Acyclic molecules use their own canonical SMILES as the scaffold key. Invalid SMILES are singleton keys.",
        "baseline": "A ligand-only model uses compounds.csv split and does not add a second split.",
    }
    (OUT_DIR / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
