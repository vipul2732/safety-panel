#!/usr/bin/env python3
"""Download ChEMBL max-phase-4 molecules: id and canonical SMILES."""

from __future__ import annotations

import csv
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = "https://www.ebi.ac.uk/chembl/api/data/molecule.json?max_phase=4&limit=500"


def main():
    url = URL
    rows = []
    while url:
        with urllib.request.urlopen(url, timeout=180) as handle:
            payload = json.load(handle)
        for molecule in payload["molecules"]:
            structures = molecule.get("molecule_structures") or {}
            rows.append((molecule["molecule_chembl_id"], structures.get("canonical_smiles") or ""))
        nxt = payload["page_meta"].get("next")
        url = ("https://www.ebi.ac.uk" + nxt) if nxt else None
        print("downloaded %d" % len(rows), flush=True)
    out = ROOT / "data" / "external"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "chembl_approved.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["molecule_chembl_id", "canonical_smiles"])
        writer.writerows(rows)
    print("wrote %d" % len(rows), flush=True)


if __name__ == "__main__":
    main()
