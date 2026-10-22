#!/usr/bin/env python3
"""Build pocket and ligand graphs for the dual encoder.

Pockets come from the 8 Angstrom mmCIF files. The co-crystal ligand is left
out of the pocket graph. Ligands get one ETKDGv3 conformer, with a 2D layout
only when embedding fails. Only potent and weak pairs are written; intermediate
and unmeasured pairs stay out of the training matrix.
"""

from __future__ import annotations

import argparse
import csv
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from schema import (
    AA_ALIASES,
    AMINO_ACIDS,
    ATOM_CLASSES,
    BACKBONE,
    ELEMENTS,
    HYBRIDIZATIONS,
    LIGAND_DIM,
    LIGAND_RADIUS,
    POCKET_DIM,
    POCKET_RADIUS,
)

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ROOT / "data" / "modeling" / "targets.csv"
COMPOUNDS = ROOT / "data" / "modeling" / "compounds.csv"
LABELS = ROOT / "data" / "modeling" / "labels.csv"
POCKET_DIR = ROOT / "data" / "structures" / "pockets"

AA_INDEX = {name: index for index, name in enumerate(AMINO_ACIDS)}
ELEMENT_INDEX = {atomic_number: index for index, atomic_number in enumerate(ELEMENTS)}


def one_hot(index, size):
    vector = np.zeros(size, dtype=np.float32)
    vector[index] = 1.0
    return vector


def element_features(atomic_number):
    return one_hot(ELEMENT_INDEX.get(int(atomic_number), len(ELEMENTS)), len(ELEMENTS) + 1)


def radius_edges(positions, cutoff, existing):
    if len(positions) == 0:
        return np.zeros((2, 0), dtype=np.int32)
    delta = positions[:, None, :] - positions[None, :, :]
    distance = np.sqrt((delta * delta).sum(-1))
    ii, jj = np.where((distance <= cutoff) & (distance > 1e-6))
    edges = {(int(i), int(j)) for i, j in existing}
    edges.update(zip(ii.tolist(), jj.tolist()))
    for index in range(len(positions)):
        edges.add((index, index))
    pairs = np.array(sorted(edges), dtype=np.int32).T
    return pairs


def pocket_graph(path):
    import gemmi

    structure = gemmi.read_structure(str(path))
    features = []
    positions = []
    for chain in structure[0]:
        for residue in chain:
            if residue.entity_type != gemmi.EntityType.Polymer:
                continue
            residue_name = AA_ALIASES.get(residue.name.strip().upper(), residue.name.strip().upper())
            aa = one_hot(AA_INDEX.get(residue_name, len(AMINO_ACIDS)), len(AMINO_ACIDS) + 1)
            for atom in residue:
                if atom.is_hydrogen():
                    continue
                atom_name = atom.name.strip().upper()
                atom_class = one_hot(
                    ATOM_CLASSES.index(atom_name) if atom_name in ATOM_CLASSES else len(ATOM_CLASSES),
                    len(ATOM_CLASSES) + 1,
                )
                backbone = np.array([1.0 if atom_name in BACKBONE else 0.0], dtype=np.float32)
                features.append(np.concatenate([element_features(atom.element.atomic_number), aa, atom_class, backbone]))
                positions.append([atom.pos.x, atom.pos.y, atom.pos.z])
    if not features:
        raise RuntimeError("no protein atoms in %s" % path)
    x = np.stack(features).astype(np.float32)
    pos = np.array(positions, dtype=np.float32)
    if x.shape[1] != POCKET_DIM:
        raise RuntimeError("pocket dim %s != %s" % (x.shape[1], POCKET_DIM))
    edges = radius_edges(pos, POCKET_RADIUS, [])
    return x, pos, edges


def featurize_ligand(task):
    os.environ["OMP_NUM_THREADS"] = "1"
    from rdkit import Chem
    from rdkit import RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")

    index, chembl_id, smiles = task
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or molecule.GetNumAtoms() == 0:
        return index, chembl_id, None
    params = AllChem.ETKDGv3()
    params.randomSeed = index % (2 ** 31 - 1)
    params.numThreads = 1
    params.timeout = 2
    coord_is_3d = 1
    if molecule.GetNumAtoms() > 80 or AllChem.EmbedMolecule(molecule, params) != 0:
        AllChem.Compute2DCoords(molecule)
        coord_is_3d = 0
    conformer = molecule.GetConformer()
    features = []
    positions = []
    for atom in molecule.GetAtoms():
        degree = one_hot(min(atom.GetDegree(), 6), 7)
        charge = one_hot(int(np.clip(atom.GetFormalCharge(), -2, 2)) + 2, 5)
        aromatic = np.array([1.0 if atom.GetIsAromatic() else 0.0], dtype=np.float32)
        hybridization = atom.GetHybridization().name
        if hybridization not in HYBRIDIZATIONS:
            hybridization = "OTHER"
        hybrid = one_hot(HYBRIDIZATIONS.index(hybridization), len(HYBRIDIZATIONS))
        hydrogens = one_hot(min(atom.GetTotalNumHs(), 4), 5)
        ring = np.array([1.0 if atom.IsInRing() else 0.0], dtype=np.float32)
        features.append(np.concatenate([
            element_features(atom.GetAtomicNum()), degree, charge, aromatic, hybrid, hydrogens, ring,
        ]))
        point = conformer.GetAtomPosition(atom.GetIdx())
        positions.append([point.x, point.y, point.z])
    bonds = []
    for bond in molecule.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bonds.append((begin, end))
        bonds.append((end, begin))
    x = np.stack(features).astype(np.float32)
    pos = np.array(positions, dtype=np.float32)
    if x.shape[1] != LIGAND_DIM:
        raise RuntimeError("ligand dim %s != %s" % (x.shape[1], LIGAND_DIM))
    edges = radius_edges(pos, LIGAND_RADIUS, bonds)
    return index, chembl_id, (x, pos, edges, coord_is_3d)


def load_supervised():
    panels = []
    with TARGETS.open() as handle:
        panels = [row["panel_id"] for row in csv.DictReader(handle)]
    panel_index = {panel_id: index for index, panel_id in enumerate(panels)}
    split_of = {}
    smiles_of = {}
    with COMPOUNDS.open() as handle:
        for row in csv.DictReader(handle):
            split_of[row["parent_molecule_chembl_id"]] = row["split"]
            smiles_of[row["parent_molecule_chembl_id"]] = row["canonical_smiles"]
    pairs = []
    molecules = []
    seen = set()
    with LABELS.open() as handle:
        for row in csv.DictReader(handle):
            if row["bin"] not in ("potent", "weak"):
                continue
            if row["panel_id"] not in panel_index:
                continue
            chembl_id = row["parent_molecule_chembl_id"]
            if chembl_id not in split_of:
                continue
            if chembl_id not in seen:
                seen.add(chembl_id)
                molecules.append(chembl_id)
            pairs.append((
                chembl_id,
                panel_index[row["panel_id"]],
                1.0 if row["bin"] == "potent" else 0.0,
                float(row["max_pchembl"]) if row["bin"] == "potent" else np.nan,
                split_of[chembl_id],
            ))
    return panels, molecules, smiles_of, pairs


def save_graphs(path, graphs, ids, extra=None):
    xs, poss, edges, node_ptr, edge_ptr = [], [], [], [0], [0]
    node_count = 0
    edge_count = 0
    flags = []
    for graph in graphs:
        x, pos, edge = graph[:3]
        xs.append(x)
        poss.append(pos)
        edges.append(edge)
        node_count += len(x)
        edge_count += edge.shape[1]
        node_ptr.append(node_count)
        edge_ptr.append(edge_count)
        if len(graph) == 4:
            flags.append(graph[3])
    payload = {
        "x": np.concatenate(xs).astype(np.float32),
        "pos": np.concatenate(poss).astype(np.float32),
        "edge": np.concatenate(edges, axis=1).astype(np.int32),
        "node_ptr": np.array(node_ptr, dtype=np.int64),
        "edge_ptr": np.array(edge_ptr, dtype=np.int64),
        "ids": np.array(ids),
    }
    if flags:
        payload["coord_is_3d"] = np.array(flags, dtype=np.int8)
    if extra:
        payload.update(extra)
    np.savez(path, **payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "data" / "features"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ["OMP_NUM_THREADS"] = "1"

    panels, molecules, smiles_of, pairs = load_supervised()
    if args.limit:
        molecules = molecules[:args.limit]
        keep = set(molecules)
        pairs = [pair for pair in pairs if pair[0] in keep]
    print("molecules %d pairs %d" % (len(molecules), len(pairs)), flush=True)

    pocket_graphs = []
    for panel_id in panels:
        pocket_graphs.append(pocket_graph(POCKET_DIR / ("%s.cif" % panel_id)))
        print("pocket %s atoms %d" % (panel_id, len(pocket_graphs[-1][0])), flush=True)
    save_graphs(out / "pockets.npz", pocket_graphs, panels)

    tasks = [(index, chembl_id, smiles_of[chembl_id]) for index, chembl_id in enumerate(molecules)]
    kept_ids = []
    kept_graphs = []
    failed = 0
    with Pool(processes=args.workers) as pool:
        for index, chembl_id, graph in pool.imap(featurize_ligand, tasks, chunksize=32):
            if graph is None:
                failed += 1
                continue
            kept_ids.append(chembl_id)
            kept_graphs.append(graph)
            if len(kept_ids) % 2000 == 0:
                print("ligands %d failed %d" % (len(kept_ids), failed), flush=True)
    print("ligands %d failed %d" % (len(kept_ids), failed), flush=True)
    save_graphs(out / "ligands.npz", kept_graphs, kept_ids)

    row_of = {chembl_id: index for index, chembl_id in enumerate(kept_ids)}
    split_code = {"train": 0, "valid": 1, "test": 2}
    split_of = {}
    with COMPOUNDS.open() as handle:
        for row in csv.DictReader(handle):
            split_of[row["parent_molecule_chembl_id"]] = split_code[row["split"]]
    n_pocket = len(panels)
    observed = np.full((len(kept_ids), n_pocket), np.nan, dtype=np.float32)
    potency = np.full((len(kept_ids), n_pocket), np.nan, dtype=np.float32)
    splits = np.array([split_of[chembl_id] for chembl_id in kept_ids], dtype=np.int8)
    for chembl_id, pocket_index, label, pchembl, _split in pairs:
        row = row_of.get(chembl_id)
        if row is None:
            continue
        observed[row, pocket_index] = label
        if label == 1.0:
            potency[row, pocket_index] = pchembl
    np.savez(
        out / "labels.npz",
        observed=observed,
        potency=potency,
        split=splits,
        ids=np.array(kept_ids),
        panels=np.array(panels),
    )
    print("wrote %s" % out, flush=True)


if __name__ == "__main__":
    main()
