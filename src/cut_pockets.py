#!/usr/bin/env python3
"""

The pocket is every amino-acid residue with a heavy atom within 8 Angstrom of
a heavy atom of the representative ligand, plus that ligand. When the ligand
is present more than once, the copy in contact with the most protein residues
is kept.
"""

from __future__ import annotations

import argparse
import csv
import json
import urllib.request
from pathlib import Path

import gemmi

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ROOT / "data" / "modeling" / "targets.csv"
CIF_DIR = ROOT / "data" / "structures" / "cif"
POCKET_DIR = ROOT / "data" / "structures" / "pockets"
SUMMARY = ROOT / "data" / "structures" / "pockets.csv"
RESIDUES = ROOT / "data" / "structures" / "pocket_residues.csv"
PROVENANCE = ROOT / "data" / "structures" / "provenance.json"
CUTOFF = 8.0
NUCLEIC = {"A", "C", "G", "U", "DA", "DC", "DG", "DT", "DI", "I"}

SUMMARY_FIELDS = [
    "panel_id", "pdb_id", "ligand_id", "ligand_chain", "ligand_seqid",
    "n_ligand_copies", "n_pocket_residues", "n_pocket_chains", "min_distance",
    "status", "note",
]
RESIDUE_FIELDS = [
    "panel_id", "pdb_id", "chain", "seqid", "residue_name", "min_distance", "role",
]


def download_cif(pdb_id):
    CIF_DIR.mkdir(parents=True, exist_ok=True)
    path = CIF_DIR / ("%s.cif" % pdb_id.lower())
    if path.exists() and path.stat().st_size > 0:
        return path
    url = "https://files.rcsb.org/download/%s.cif" % pdb_id.upper()
    request = urllib.request.Request(url, headers={"User-Agent": "siwss-target-like/0.1"})
    with urllib.request.urlopen(request, timeout=300) as response:
        path.write_bytes(response.read())
    return path


def selected_atoms(residue):
    groups = {}
    for atom in residue:
        groups.setdefault(atom.name, []).append(atom)
    chosen = []
    for atoms in groups.values():
        blank = [atom for atom in atoms if not atom.has_altloc()]
        if blank:
            chosen.append(blank[0])
            continue
        atoms.sort(key=lambda atom: (-atom.occ, atom.altloc))
        chosen.append(atoms[0])
    return chosen


def heavy_atoms(residue):
    return [atom for atom in selected_atoms(residue) if not atom.is_hydrogen()]


def is_protein(residue):
    return (
        residue.entity_type == gemmi.EntityType.Polymer
        and residue.name not in NUCLEIC
        and residue.name not in ("HOH", "WAT", "DOD")
    )


def residue_key(chain, residue):
    seqid = residue.seqid
    icode = (seqid.icode or "").strip(" \0")
    return "%s:%s%s" % (chain.name, seqid.num, icode)


def find_ligands(model, ligand_id):
    wanted = ligand_id.upper()
    found = []
    for chain in model:
        for residue in chain:
            if residue.entity_type == gemmi.EntityType.Water:
                continue
            if residue.name.upper() != wanted:
                continue
            atoms = heavy_atoms(residue)
            if atoms:
                found.append((chain, residue, atoms))
    return found


def protein_atoms(model):
    records = []
    for chain in model:
        for residue in chain:
            if not is_protein(residue):
                continue
            atoms = heavy_atoms(residue)
            if atoms:
                records.append((chain, residue, atoms))
    return records


def residue_min_distance(ligand_atoms, atoms):
    best = None
    for protein_atom in atoms:
        for ligand_atom in ligand_atoms:
            distance = protein_atom.pos.dist(ligand_atom.pos)
            if best is None or distance < best:
                best = distance
    return best


def choose_ligand(copies, proteins):
    ranked = []
    for chain, residue, atoms in copies:
        hits = []
        nearest = None
        for protein_chain, protein_residue, protein_atoms_ in proteins:
            distance = residue_min_distance(atoms, protein_atoms_)
            if distance is None:
                continue
            if nearest is None or distance < nearest:
                nearest = distance
            if distance <= CUTOFF:
                hits.append((distance, protein_chain, protein_residue))
        ranked.append((len(hits), -(nearest if nearest is not None else 1e9), chain, residue, atoms, hits, nearest))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return ranked[0]


def copy_residue(residue, atoms):
    copied = gemmi.Residue()
    copied.name = residue.name
    copied.seqid = residue.seqid
    copied.subchain = residue.subchain
    copied.entity_type = residue.entity_type
    copied.het_flag = residue.het_flag
    copied.entity_id = residue.entity_id
    copied.label_seq = residue.label_seq
    for atom in atoms:
        copied.add_atom(atom)
    return copied


def write_pocket(path, pdb_id, ligand_chain, ligand_residue, ligand_atoms, hits):
    path.parent.mkdir(parents=True, exist_ok=True)
    structure = gemmi.Structure()
    structure.name = pdb_id.lower()
    model = gemmi.Model("1")
    chains = {}

    def chain_for(source_chain):
        if source_chain.name not in chains:
            chains[source_chain.name] = gemmi.Chain(source_chain.name)
        return chains[source_chain.name]

    for _distance, chain, residue in sorted(hits, key=lambda item: (item[1].name, item[2].seqid.num)):
        chain_for(chain).add_residue(copy_residue(residue, selected_atoms(residue)))
    chain_for(ligand_chain).add_residue(copy_residue(ligand_residue, ligand_atoms))
    for chain in chains.values():
        model.add_chain(chain)
    structure.add_model(model)
    structure.setup_entities()
    structure.make_mmcif_document().write_file(str(path))


def finished_ids():
    if not SUMMARY.exists():
        return set()
    with SUMMARY.open() as handle:
        return {row["panel_id"] for row in csv.DictReader(handle) if row["status"] == "ok"}


def append_rows(summary_row, residue_rows):
    POCKET_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    new_summary = not SUMMARY.exists()
    new_residues = not RESIDUES.exists()
    with SUMMARY.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        if new_summary:
            writer.writeheader()
        writer.writerow(summary_row)
    with RESIDUES.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESIDUE_FIELDS)
        if new_residues:
            writer.writeheader()
        writer.writerows(residue_rows)


def process(row):
    panel_id = row["panel_id"]
    pdb_id = row["representative_pdb"]
    ligand_id = row["representative_ligand"]
    path = download_cif(pdb_id)
    structure = gemmi.read_structure(str(path))
    model = structure[0]
    copies = find_ligands(model, ligand_id)
    if not copies:
        return {
            "panel_id": panel_id,
            "pdb_id": pdb_id,
            "ligand_id": ligand_id,
            "ligand_chain": "",
            "ligand_seqid": "",
            "n_ligand_copies": 0,
            "n_pocket_residues": 0,
            "n_pocket_chains": 0,
            "min_distance": "",
            "status": "ligand_not_found",
            "note": "",
        }, []
    proteins = protein_atoms(model)
    _n_hits, _neg_dist, chain, residue, atoms, hits, nearest = choose_ligand(copies, proteins)
    if not hits:
        return {
            "panel_id": panel_id,
            "pdb_id": pdb_id,
            "ligand_id": ligand_id,
            "ligand_chain": chain.name,
            "ligand_seqid": residue_key(chain, residue).split(":", 1)[1],
            "n_ligand_copies": len(copies),
            "n_pocket_residues": 0,
            "n_pocket_chains": 0,
            "min_distance": "" if nearest is None else "%.2f" % nearest,
            "status": "no_pocket_contact",
            "note": "",
        }, []
    pocket_path = POCKET_DIR / ("%s.cif" % panel_id)
    write_pocket(pocket_path, pdb_id, chain, residue, atoms, hits)
    residue_rows = []
    for distance, protein_chain, protein_residue in hits:
        residue_rows.append({
            "panel_id": panel_id,
            "pdb_id": pdb_id,
            "chain": protein_chain.name,
            "seqid": residue_key(protein_chain, protein_residue).split(":", 1)[1],
            "residue_name": protein_residue.name,
            "min_distance": "%.2f" % distance,
            "role": "pocket",
        })
    residue_rows.append({
        "panel_id": panel_id,
        "pdb_id": pdb_id,
        "chain": chain.name,
        "seqid": residue_key(chain, residue).split(":", 1)[1],
        "residue_name": residue.name,
        "min_distance": "0.00",
        "role": "ligand",
    })
    summary = {
        "panel_id": panel_id,
        "pdb_id": pdb_id,
        "ligand_id": ligand_id,
        "ligand_chain": chain.name,
        "ligand_seqid": residue_key(chain, residue).split(":", 1)[1],
        "n_ligand_copies": len(copies),
        "n_pocket_residues": len(hits),
        "n_pocket_chains": len({item[1].name for item in hits}),
        "min_distance": "%.2f" % min(item[0] for item in hits),
        "status": "ok",
        "note": "",
    }
    return summary, residue_rows


def write_provenance():
    PROVENANCE.parent.mkdir(parents=True, exist_ok=True)
    PROVENANCE.write_text(json.dumps({
        "cutoff_angstrom": CUTOFF,
        "distance": "Minimum heavy-atom distance from the representative ligand to the residue.",
        "ligand_copy": "The copy contacting the most protein residues within the cutoff. Ties go to the closer copy.",
        "protein": "Polymer residues other than nucleic acids and water. Modified amino acids stay in.",
        "altloc": "Blank altloc, otherwise the highest occupancy.",
        "source": "RCSB mmCIF for the representative PDB id in data/modeling/targets.csv.",
    }, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="")
    args = parser.parse_args()
    selected = {item for item in args.only.split(",") if item}
    done = set() if selected else finished_ids()
    with TARGETS.open() as handle:
        rows = [row for row in csv.DictReader(handle) if (not selected or row["panel_id"] in selected) and row["panel_id"] not in done]
    print("cutting %d pockets" % len(rows), flush=True)
    write_provenance()
    for index, row in enumerate(rows, start=1):
        print("[%d/%d] %s %s %s" % (index, len(rows), row["panel_id"], row["representative_pdb"], row["representative_ligand"]), flush=True)
        try:
            summary, residue_rows = process(row)
        except Exception as exc:
            summary = {
                "panel_id": row["panel_id"],
                "pdb_id": row["representative_pdb"],
                "ligand_id": row["representative_ligand"],
                "ligand_chain": "",
                "ligand_seqid": "",
                "n_ligand_copies": 0,
                "n_pocket_residues": 0,
                "n_pocket_chains": 0,
                "min_distance": "",
                "status": "error",
                "note": str(exc)[:300],
            }
            residue_rows = []
        if selected:
            print(json.dumps(summary, indent=2), flush=True)
        else:
            append_rows(summary, residue_rows)
        print("  %s residues=%s" % (summary["status"], summary["n_pocket_residues"]), flush=True)


if __name__ == "__main__":
    main()
