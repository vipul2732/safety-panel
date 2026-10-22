#!/usr/bin/env python3
"""Local page for the safety-panel fingerprint tool."""

from __future__ import annotations

import csv
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from predict_panel import predict

ROOT = Path(__file__).resolve().parents[1]
PAGE_PATH = Path(__file__).with_name("panel_page.html")
REFERENCE = np.load(ROOT / "data" / "models" / "fingerprint_cutoff" / "reference.npz", allow_pickle=True)
NAMES = {}
CLASSES = {}
with open(ROOT / "data" / "modeling" / "targets.csv") as handle:
    for row in csv.DictReader(handle):
        NAMES[row["panel_id"]] = row["target_name"]
        CLASSES[row["panel_id"]] = row["target_class"]


def molecule_svg(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise SystemExit("could not read SMILES")
    drawer = rdMolDraw2D.MolDraw2DSVG(320, 200)
    options = drawer.drawOptions()
    options.clearBackground = False
    options.bondLineWidth = 1.6
    options.padding = 0.1
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, molecule)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg[svg.find("<svg"):]


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.split("?", 1)[0] != "/":
            self.send_error(404)
            return
        body = PAGE_PATH.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/api/predict":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
            smiles = str(payload.get("smiles", "")).strip()
            if not smiles:
                raise ValueError("Enter a SMILES string")
            scores, order, score, distance, call = predict(smiles, REFERENCE)
            svg = molecule_svg(smiles)
        except (ValueError, SystemExit) as exc:
            message = exc.code if isinstance(exc, SystemExit) and isinstance(exc.code, str) else str(exc)
            self._json(400, {"error": message or "could not read SMILES"})
            return
        targets = []
        for rank, target in enumerate(order, start=1):
            panel = str(REFERENCE["panels"][target])
            targets.append({
                "rank": rank,
                "panel": panel,
                "gene": str(REFERENCE["genes"][target]),
                "name": NAMES.get(panel, panel),
                "class": CLASSES.get(panel, ""),
                "score": round(float(scores[target]), 4),
            })
        self._json(200, {
            "call": call,
            "score": round(float(score), 4),
            "distance": round(float(distance), 4),
            "svg": svg,
            "targets": targets,
        })

    def _json(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        print(fmt % args, flush=True)


def main():
    server = ThreadingHTTPServer(("0.0.0.0", 8765), Handler)
    print("open http://127.0.0.1:8765", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
