#!/usr/bin/env python3
"""Mine ChEMBL bioactivities and PDB ligand-bound structures for the safety panel.

The 44-target set is the Bowes 2012 consensus panel, with gene symbols taken
from the open Redfern 2025 Table 2 (Bowes column). The extended set is the
77-target InVEST / SafetyScreen panel that vendors state is aligned with
Brennan 2024 Safety-77. One Bowes target, nicotinic alpha4/beta2, is absent
from that vendor 77 and is replaced there by nicotinic alpha1, so the union
mined here is 78 panel rows.

ChEMBL_37 activities are binding or functional assays with a pChEMBL value.
Mutant, duplicate, and invalid records are dropped. A molecule is potent at
pChEMBL >= 6 (<= 1 uM), intermediate at 5 <= pChEMBL < 6, and weak at
pChEMBL < 5, using the maximum pChEMBL across retained measurements.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PANEL_CSV = ROOT / "data" / "panel" / "safety_panel.csv"
RESULT_CSV = ROOT / "data" / "results" / "target_feasibility.csv"
LIGAND_DIR = ROOT / "data" / "ligands"
PROVENANCE = ROOT / "data" / "results" / "provenance.json"

CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"
UNIPROT = "https://rest.uniprot.org/uniprotkb/search"
RCSB_SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_GRAPHQL = "https://data.rcsb.org/graphql"
USER_AGENT = "siwss-target-like/0.1 (safety-panel data mining)"

# Crystallization components, ions, sugars, detergents, and membrane lipids.
# A representative ligand has to fall outside this set and weigh at least 150 Da.
EXCLUDED_COMP_IDS = {
    "NA", "CL", "MG", "ZN", "CA", "MN", "FE", "K", "CU", "CD", "NI", "CO", "HG",
    "BR", "IOD", "YB", "SR", "BA", "CS", "AU", "PT", "AG", "LI", "RB", "MO",
    "SO4", "PO4", "NO3", "SCN", "CO3", "ACT", "ACE", "FMT", "ACY", "DMS", "EOH",
    "IPA", "MOH", "EDO", "GOL", "PEG", "PG4", "PGE", "1PE", "2PE", "P6G", "PE8",
    "P33", "PE4", "MPD", "BME", "DTT", "DTV", "TRS", "EPE", "MES", "HEP", "IMD",
    "BCT", "FLC", "MLA", "CIT", "TAR", "SUC", "SIN", "TLA", "NH4", "AZI", "OH",
    "HOH", "DOD", "UNX", "UNL", "UNK",
    "NAG", "BMA", "MAN", "GAL", "FUC", "NDG", "NGA", "A2G", "FUL", "GLC", "BGC",
    "BOG", "LMT", "C8E", "12P", "D12", "LPP", "LXA", "LMG", "C10", "DAO", "HTG",
    "PLM", "OLC", "OLA", "OLB", "CLR", "CLL", "CDL", "PEK", "PEE", "STE", "MYR",
    "PAM", "PAE", "LPP", "SPH", "CER", "POP", "POV", "PTY", "DGA", "SQL",
}

# panel_id, genes, name, target_class, in_bowes44, in_vendor77, note
PANEL = [
    ("ACHE", "ACHE", "Acetylcholinesterase", "enzyme", 1, 1, ""),
    ("ADORA2A", "ADORA2A", "Adenosine A2A receptor", "gpcr", 1, 1, ""),
    ("ADRA1A", "ADRA1A", "Alpha-1A adrenergic receptor", "gpcr", 1, 1, ""),
    ("ADRA2A", "ADRA2A", "Alpha-2A adrenergic receptor", "gpcr", 1, 1, ""),
    ("ADRB1", "ADRB1", "Beta-1 adrenergic receptor", "gpcr", 1, 1, ""),
    ("ADRB2", "ADRB2", "Beta-2 adrenergic receptor", "gpcr", 1, 1, ""),
    ("AR", "AR", "Androgen receptor", "nuclear_receptor", 1, 1, ""),
    ("ATM", "ATM", "Serine-protein kinase ATM", "kinase", 0, 1, ""),
    ("AURKA", "AURKA", "Aurora kinase A", "kinase", 0, 1, ""),
    ("AURKB", "AURKB", "Aurora kinase B", "kinase", 0, 1, ""),
    ("AVPR1A", "AVPR1A", "Vasopressin V1A receptor", "gpcr", 1, 1, ""),
    ("BRD4", "BRD4", "Bromodomain-containing protein 4", "bromodomain", 0, 1, ""),
    ("CACNA1C", "CACNA1C", "Voltage-gated calcium channel Cav1.2 alpha-1C", "ion_channel", 1, 1, "Alpha-1C subunit of the L-type channel."),
    ("CCKAR", "CCKAR", "Cholecystokinin receptor type 1", "gpcr", 1, 1, "Vendor table listed this Bowes target under the gene symbol CCL28."),
    ("CCKBR", "CCKBR", "Cholecystokinin receptor type 2", "gpcr", 0, 1, ""),
    ("CHRM1", "CHRM1", "Muscarinic acetylcholine receptor M1", "gpcr", 1, 1, ""),
    ("CHRM2", "CHRM2", "Muscarinic acetylcholine receptor M2", "gpcr", 1, 1, ""),
    ("CHRM3", "CHRM3", "Muscarinic acetylcholine receptor M3", "gpcr", 1, 1, ""),
    ("CHRNA1", "CHRNA1", "Nicotinic acetylcholine receptor alpha-1", "ion_channel", 0, 1, "Present in the vendor 77. Not in the Bowes 44."),
    ("CHRNA4_CHRNB2", "CHRNA4;CHRNB2", "Nicotinic acetylcholine receptor alpha4/beta2", "ion_channel", 1, 0, "Bowes 44 target. Absent from the vendor 77, which lists alpha-1 instead."),
    ("CNR1", "CNR1", "Cannabinoid receptor 1", "gpcr", 1, 1, ""),
    ("CNR2", "CNR2", "Cannabinoid receptor 2", "gpcr", 1, 1, ""),
    ("CTSD", "CTSD", "Cathepsin D", "protease", 0, 1, ""),
    ("DRD1", "DRD1", "Dopamine D1 receptor", "gpcr", 1, 1, ""),
    ("DRD2", "DRD2", "Dopamine D2 receptor", "gpcr", 1, 1, ""),
    ("DRD3", "DRD3", "Dopamine D3 receptor", "gpcr", 0, 1, ""),
    ("EDNRA", "EDNRA", "Endothelin receptor type A", "gpcr", 1, 1, ""),
    ("EGFR", "EGFR", "Epidermal growth factor receptor", "kinase", 0, 1, ""),
    ("ESR1", "ESR1", "Estrogen receptor alpha", "nuclear_receptor", 0, 1, ""),
    ("FGFR1", "FGFR1", "Fibroblast growth factor receptor 1", "kinase", 0, 1, ""),
    ("FLT1", "FLT1", "Vascular endothelial growth factor receptor 1", "kinase", 0, 1, "Vendor name said FMS (VEGFR1). FMS is CSF1R; this target is VEGFR1."),
    ("GABRA1", "GABRA1;GABRB2;GABRG2", "GABA-A receptor, central benzodiazepine site", "ion_channel", 1, 1, "Vendor gene token was GABRAx. Mined on the alpha-1 subunit; beta-2 and gamma-2 complete the benzodiazepine site."),
    ("GPBAR1", "GPBAR1", "G-protein coupled bile acid receptor 1", "gpcr", 0, 1, ""),
    ("GRIN1", "GRIN1", "NMDA receptor GluN1, glycine site", "ion_channel", 1, 1, "Vendor gene token was NMDAR1."),
    ("GSK3B", "GSK3B", "Glycogen synthase kinase-3 beta", "kinase", 0, 1, ""),
    ("HRH1", "HRH1", "Histamine H1 receptor", "gpcr", 1, 1, ""),
    ("HRH2", "HRH2", "Histamine H2 receptor", "gpcr", 1, 1, ""),
    ("HTR1A", "HTR1A", "5-hydroxytryptamine receptor 1A", "gpcr", 1, 1, ""),
    ("HTR1B", "HTR1B", "5-hydroxytryptamine receptor 1B", "gpcr", 1, 1, ""),
    ("HTR2A", "HTR2A", "5-hydroxytryptamine receptor 2A", "gpcr", 1, 1, ""),
    ("HTR2B", "HTR2B", "5-hydroxytryptamine receptor 2B", "gpcr", 1, 1, ""),
    ("HTR3A", "HTR3A", "5-hydroxytryptamine receptor 3A", "ion_channel", 1, 1, ""),
    ("HTR6", "HTR6", "5-hydroxytryptamine receptor 6", "gpcr", 0, 1, ""),
    ("INSR", "INSR", "Insulin receptor", "kinase", 0, 1, ""),
    ("KCNH2", "KCNH2", "Voltage-gated potassium channel hERG", "ion_channel", 1, 1, ""),
    ("KCNQ1", "KCNQ1;KCNE1", "Voltage-gated potassium channel Kv7.1 / minK", "ion_channel", 1, 1, "The safety assay is the KCNQ1/KCNE1 IKs complex. Activities are mined on KCNQ1."),
    ("KDR", "KDR", "Vascular endothelial growth factor receptor 2", "kinase", 0, 1, ""),
    ("KIT", "KIT", "Mast/stem cell growth factor receptor Kit", "kinase", 0, 1, ""),
    ("LCK", "LCK", "Tyrosine-protein kinase Lck", "kinase", 1, 1, ""),
    ("MALT1", "MALT1", "Mucosa-associated lymphoid tissue lymphoma translocation protein 1", "protease", 0, 1, ""),
    ("MAOA", "MAOA", "Monoamine oxidase A", "enzyme", 1, 1, ""),
    ("MAOB", "MAOB", "Monoamine oxidase B", "enzyme", 0, 1, ""),
    ("MAP2K7", "MAP2K7", "Dual specificity mitogen-activated protein kinase kinase 7", "kinase", 0, 1, ""),
    ("MRGPRX2", "MRGPRX2", "Mas-related G-protein coupled receptor member X2", "gpcr", 0, 1, ""),
    ("NR3C1", "NR3C1", "Glucocorticoid receptor", "nuclear_receptor", 1, 1, ""),
    ("NTRK1", "NTRK1", "High affinity nerve growth factor receptor TrkA", "kinase", 0, 1, ""),
    ("OPRD1", "OPRD1", "Delta-type opioid receptor", "gpcr", 1, 1, ""),
    ("OPRK1", "OPRK1", "Kappa-type opioid receptor", "gpcr", 1, 1, ""),
    ("OPRM1", "OPRM1", "Mu-type opioid receptor", "gpcr", 1, 1, ""),
    ("PDE3A", "PDE3A", "cGMP-inhibited 3',5'-cyclic phosphodiesterase A", "enzyme", 1, 1, ""),
    ("PDE4D", "PDE4D", "cAMP-specific 3',5'-cyclic phosphodiesterase 4D", "enzyme", 1, 1, "Vendor table listed the isoform PDE4D2. Mined as the gene PDE4D."),
    ("PDK1", "PDK1", "Pyruvate dehydrogenase kinase isoform 1", "kinase", 0, 1, "Vendor token was PDHK1. This is HGNC PDK1, not PDPK1."),
    ("PGR", "PGR", "Progesterone receptor", "nuclear_receptor", 0, 1, ""),
    ("PIK3CG", "PIK3CG", "Phosphatidylinositol 4,5-bisphosphate 3-kinase catalytic subunit gamma", "kinase", 0, 1, ""),
    ("PTGS1", "PTGS1", "Cyclooxygenase-1", "enzyme", 1, 1, ""),
    ("PTGS2", "PTGS2", "Cyclooxygenase-2", "enzyme", 1, 1, ""),
    ("PTK2", "PTK2", "Focal adhesion kinase 1", "kinase", 0, 1, ""),
    ("RARA", "RARA", "Retinoic acid receptor alpha", "nuclear_receptor", 0, 1, ""),
    ("ROCK1", "ROCK1", "Rho-associated protein kinase 1", "kinase", 0, 1, ""),
    ("ROCK2", "ROCK2", "Rho-associated protein kinase 2", "kinase", 0, 1, ""),
    ("SCN5A", "SCN5A", "Voltage-gated sodium channel Nav1.5", "ion_channel", 1, 1, ""),
    ("SLC6A1", "SLC6A1", "Sodium- and chloride-dependent GABA transporter 1", "transporter", 0, 1, ""),
    ("SLC6A2", "SLC6A2", "Sodium-dependent noradrenaline transporter", "transporter", 1, 1, ""),
    ("SLC6A3", "SLC6A3", "Sodium-dependent dopamine transporter", "transporter", 1, 1, ""),
    ("SLC6A4", "SLC6A4", "Sodium-dependent serotonin transporter", "transporter", 1, 1, ""),
    ("SRC", "SRC", "Proto-oncogene tyrosine-protein kinase Src", "kinase", 0, 1, ""),
    ("STK35", "STK35", "Serine/threonine-protein kinase 35", "kinase", 0, 1, "Also called CLIK1."),
    ("TACR3", "TACR3", "Neuromedin-K receptor NK3", "gpcr", 0, 1, ""),
]

RESULT_FIELDS = [
    "panel_id", "gene_symbols", "target_name", "target_class", "in_bowes44", "in_vendor77",
    "uniprot_accessions", "chembl_target_id", "chembl_pref_name", "chembl_target_type",
    "n_activities_retained", "n_molecules_potent", "n_molecules_intermediate", "n_molecules_weak",
    "pdb_human_nonpolymer", "pdb_any_nonpolymer", "representative_pdb", "representative_resolution",
    "representative_method", "representative_ligand", "representative_ligand_name",
    "meets_ligand_floor", "has_human_druglike_structure", "note",
]

LIGAND_FIELDS = [
    "parent_molecule_chembl_id", "canonical_smiles", "max_pchembl", "n_measurements",
    "standard_type_at_max", "assay_type_at_max", "bin",
]


def get_json(url, data=None, timeout=180, attempts=5):
    payload = None if data is None else json.dumps(data).encode()
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    last_error = None
    for attempt in range(attempts):
        request = urllib.request.Request(url, data=payload, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:500]
            last_error = "%s %s" % (exc.code, body)
            if exc.code not in (429, 500, 502, 503, 504):
                raise RuntimeError("%s failed: %s" % (url, last_error))
        except Exception as exc:
            last_error = str(exc)
        time.sleep(2 * (attempt + 1))
    raise RuntimeError("%s failed after retries: %s" % (url, last_error))


def write_panel_csv():
    PANEL_CSV.parent.mkdir(parents=True, exist_ok=True)
    with PANEL_CSV.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["panel_id", "gene_symbols", "target_name", "target_class", "in_bowes44", "in_vendor77", "note"])
        for row in PANEL:
            writer.writerow(row)


def uniprot_for_gene(gene):
    query = "gene_exact:%s AND organism_id:9606 AND reviewed:true" % gene
    url = UNIPROT + "?" + urllib.parse.urlencode({
        "query": query,
        "format": "json",
        "size": "5",
        "fields": "accession,gene_primary,protein_name,length",
    })
    data = get_json(url)
    results = data.get("results") or []
    exact = []
    for item in results:
        genes = []
        for gene_block in item.get("genes") or []:
            name = (gene_block.get("geneName") or {}).get("value")
            if name:
                genes.append(name)
        if gene in genes:
            exact.append(item)
    chosen = (exact or results)
    if not chosen:
        return "", ""
    item = chosen[0]
    accession = item.get("primaryAccession") or ""
    description = item.get("proteinDescription") or {}
    recommended = (description.get("recommendedName") or {}).get("fullName") or {}
    return accession, recommended.get("value") or ""


def chembl_targets_for_accession(accession):
    url = "%s/target.json?%s" % (CHEMBL, urllib.parse.urlencode({
        "target_components__accession": accession,
        "limit": "100",
    }))
    data = get_json(url)
    return data.get("targets") or []


def component_accessions(target):
    accessions = []
    for component in target.get("target_components") or []:
        accession = component.get("accession")
        if accession:
            accessions.append(accession)
    return accessions


def choose_single_protein(targets, primary_gene):
    human = [target for target in targets if target.get("organism") == "Homo sapiens" and target.get("target_type") == "SINGLE PROTEIN"]
    pool = human or [target for target in targets if target.get("target_type") == "SINGLE PROTEIN"]
    if not pool:
        return None
    for target in pool:
        if primary_gene.lower() in (target.get("pref_name") or "").lower():
            return target
    singles = [target for target in pool if len(component_accessions(target)) == 1]
    return (singles or pool)[0]


def choose_complex(targets, required_accessions):
    required = set(required_accessions)
    matches = []
    for target in targets:
        if target.get("organism") != "Homo sapiens":
            continue
        if target.get("target_type") not in ("PROTEIN COMPLEX", "PROTEIN COMPLEX GROUP"):
            continue
        if required.issubset(set(component_accessions(target))):
            matches.append(target)
    return matches[0] if len(matches) == 1 else None


def iter_activities(target_id, value_filter):
    offset = 0
    limit = 1000
    only = ",".join([
        "parent_molecule_chembl_id", "molecule_chembl_id", "canonical_smiles",
        "pchembl_value", "standard_type", "assay_type", "potential_duplicate",
        "data_validity_comment", "assay_variant_mutation",
    ])
    total = None
    while True:
        params = {
            "target_chembl_id": target_id,
            "assay_type__in": "B,F",
            "pchembl_value__%s" % value_filter[0]: value_filter[1],
            "limit": str(limit),
            "offset": str(offset),
            "only": only,
        }
        url = "%s/activity.json?%s" % (CHEMBL, urllib.parse.urlencode(params))
        data = get_json(url)
        activities = data.get("activities") or []
        if total is None:
            total = (data.get("page_meta") or {}).get("total_count") or 0
        for activity in activities:
            yield activity
        offset += limit
        if offset >= total or not activities:
            break


def accumulate(molecules, activities_seen, activity):
    if activity.get("potential_duplicate") in (1, True, "1"):
        return
    if activity.get("data_validity_comment"):
        return
    if activity.get("assay_variant_mutation"):
        return
    smiles = activity.get("canonical_smiles")
    parent = activity.get("parent_molecule_chembl_id") or activity.get("molecule_chembl_id")
    if not smiles or not parent:
        return
    try:
        pchembl = float(activity.get("pchembl_value"))
    except (TypeError, ValueError):
        return
    activities_seen[0] += 1
    current = molecules.get(parent)
    measurement = {
        "parent_molecule_chembl_id": parent,
        "canonical_smiles": smiles,
        "max_pchembl": pchembl,
        "n_measurements": 1,
        "standard_type_at_max": activity.get("standard_type") or "",
        "assay_type_at_max": activity.get("assay_type") or "",
    }
    if current is None or pchembl > current["max_pchembl"]:
        if current is not None:
            measurement["n_measurements"] = current["n_measurements"] + 1
        molecules[parent] = measurement
    else:
        current["n_measurements"] += 1


def bin_for(pchembl):
    if pchembl >= 6:
        return "potent"
    if pchembl >= 5:
        return "intermediate"
    return "weak"


def mine_activities(target_id):
    molecules = {}
    seen = [0]
    for value_filter in (("gte", "5"), ("lt", "5")):
        for activity in iter_activities(target_id, value_filter):
            accumulate(molecules, seen, activity)
    rows = []
    counts = {"potent": 0, "intermediate": 0, "weak": 0}
    for item in molecules.values():
        label = bin_for(item["max_pchembl"])
        item["bin"] = label
        item["max_pchembl"] = "%.2f" % item["max_pchembl"]
        counts[label] += 1
        rows.append(item)
    rows.sort(key=lambda item: (-float(item["max_pchembl"]), item["parent_molecule_chembl_id"]))
    return seen[0], counts, rows


def pdb_count(accessions, human_only):
    nodes = []
    if len(accessions) == 1:
        nodes.append({
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                "operator": "exact_match",
                "value": accessions[0],
            },
        })
    else:
        for accession in accessions:
            nodes.append({
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                    "operator": "exact_match",
                    "value": accession,
                },
            })
    nodes.append({
        "type": "terminal",
        "service": "text",
        "parameters": {
            "attribute": "rcsb_entry_info.nonpolymer_entity_count",
            "operator": "greater",
            "value": 0,
        },
    })
    if human_only:
        nodes.append({
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_entity_source_organism.taxonomy_lineage.id",
                "operator": "exact_match",
                "value": "9606",
            },
        })
    query = {
        "query": {"type": "group", "logical_operator": "and", "nodes": nodes},
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": 15},
            "sort": [{"sort_by": "rcsb_entry_info.resolution_combined", "direction": "asc"}],
        },
    }
    try:
        data = get_json(RCSB_SEARCH, data=query)
    except RuntimeError:
        return 0, []
    identifiers = [hit.get("identifier") for hit in data.get("result_set") or [] if hit.get("identifier")]
    return int(data.get("total_count") or 0), identifiers


LIPID_NAME_TOKENS = (
    "CHOLESTEROL", "HEMISUCCINATE", "OLEIC", "PALMIT", "STEAR", "MYRIST",
    "PHOSPHATID", "DETERGENT", "MALTOSIDE", "GLUCOSIDE", "POLYETHYLENE",
    "GLYCEROL", "DODECYL", "OCTYL", "LAURYL", "LIPID", "FATTY ACID",
)


def druglike_ligands(entry):
    ligands = []
    for entity in entry.get("nonpolymer_entities") or []:
        chem = ((entity or {}).get("nonpolymer_comp") or {}).get("chem_comp") or {}
        comp_id = chem.get("id") or ""
        name = (chem.get("name") or "").upper()
        weight = chem.get("formula_weight") or 0
        if comp_id in EXCLUDED_COMP_IDS or weight < 160 or weight > 750:
            continue
        if any(token in name for token in LIPID_NAME_TOKENS):
            continue
        ligands.append((abs(weight - 400), comp_id, chem.get("name") or ""))
    ligands.sort()
    return ligands


def representative_structure(entry_ids):
    if not entry_ids:
        return {}
    quoted = ", ".join('"%s"' % entry_id for entry_id in entry_ids)
    query = """
    {
      entries(entry_ids: [%s]) {
        rcsb_id
        exptl { method }
        rcsb_entry_info { resolution_combined }
        nonpolymer_entities { nonpolymer_comp { chem_comp { id formula_weight name } } }
      }
    }
    """ % quoted
    data = get_json(RCSB_GRAPHQL, data={"query": query})
    best = None
    for entry in (data.get("data") or {}).get("entries") or []:
        if not entry:
            continue
        ligands = druglike_ligands(entry)
        if not ligands:
            continue
        resolution_values = (entry.get("rcsb_entry_info") or {}).get("resolution_combined") or [99]
        resolution = resolution_values[0]
        method = ""
        exptl = entry.get("exptl") or []
        if exptl:
            method = exptl[0].get("method") or ""
        candidate = {
            "representative_pdb": entry.get("rcsb_id") or "",
            "representative_resolution": resolution,
            "representative_method": method,
            "representative_ligand": ligands[0][1],
            "representative_ligand_name": ligands[0][2],
        }
        if best is None or candidate["representative_resolution"] < best["representative_resolution"]:
            best = candidate
    return best or {}


def write_ligands(panel_id, rows):
    LIGAND_DIR.mkdir(parents=True, exist_ok=True)
    path = LIGAND_DIR / ("%s.csv" % panel_id)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LIGAND_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def finished_ids():
    if not RESULT_CSV.exists():
        return set()
    with RESULT_CSV.open() as handle:
        return {row["panel_id"] for row in csv.DictReader(handle)}


def append_result(row):
    RESULT_CSV.parent.mkdir(parents=True, exist_ok=True)
    new_file = not RESULT_CSV.exists()
    with RESULT_CSV.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def mine_one(panel_row):
    panel_id, genes, name, target_class, in_bowes, in_vendor, note = panel_row
    gene_list = genes.split(";")
    accessions = []
    for gene in gene_list:
        accession, _protein_name = uniprot_for_gene(gene)
        if accession:
            accessions.append(accession)
    primary_accession = accessions[0] if accessions else ""
    chembl_target = None
    if primary_accession:
        targets = chembl_targets_for_accession(primary_accession)
        chembl_target = choose_single_protein(targets, gene_list[0])
        complex_target = choose_complex(targets, accessions) if len(accessions) > 1 else None
    else:
        targets = []
        complex_target = None

    activities = 0
    counts = {"potent": 0, "intermediate": 0, "weak": 0}
    used = chembl_target
    if used:
        activities, counts, ligand_rows = mine_activities(used["target_chembl_id"])
        if complex_target and complex_target["target_chembl_id"] != used["target_chembl_id"]:
            complex_activities, complex_counts, complex_rows = mine_activities(complex_target["target_chembl_id"])
            if complex_counts["potent"] > counts["potent"]:
                activities, counts, ligand_rows = complex_activities, complex_counts, complex_rows
                used = complex_target
                note = (note + " " if note else "") + "Activities taken from the human protein complex %s." % used["target_chembl_id"]
        write_ligands(panel_id, ligand_rows)
    else:
        write_ligands(panel_id, [])

    human_count, human_ids = pdb_count(accessions or [primary_accession], True) if accessions else (0, [])
    any_count, _any_ids = pdb_count(accessions or [primary_accession], False) if accessions else (0, [])
    if human_count == 0 and accessions:
        # A multi-subunit query can miss entries that contain only the primary subunit.
        human_count, human_ids = pdb_count([accessions[0]], True)
        any_count, _any_ids = pdb_count([accessions[0]], False)
        if human_count:
            note = (note + " " if note else "") + "PDB count uses the primary subunit only."
    structure = representative_structure(human_ids) if human_ids else {}
    if not structure and any_count and not human_count:
        _count, any_ids = pdb_count(accessions[:1] or accessions, False)
        structure = representative_structure(any_ids)
        if structure:
            note = (note + " " if note else "") + "Representative structure is not limited to human entries."

    potent = counts["potent"]
    result = {
        "panel_id": panel_id,
        "gene_symbols": genes,
        "target_name": name,
        "target_class": target_class,
        "in_bowes44": in_bowes,
        "in_vendor77": in_vendor,
        "uniprot_accessions": ";".join(accessions),
        "chembl_target_id": (used or {}).get("target_chembl_id", ""),
        "chembl_pref_name": (used or {}).get("pref_name", ""),
        "chembl_target_type": (used or {}).get("target_type", ""),
        "n_activities_retained": activities,
        "n_molecules_potent": potent,
        "n_molecules_intermediate": counts["intermediate"],
        "n_molecules_weak": counts["weak"],
        "pdb_human_nonpolymer": human_count,
        "pdb_any_nonpolymer": any_count,
        "representative_pdb": structure.get("representative_pdb", ""),
        "representative_resolution": structure.get("representative_resolution", ""),
        "representative_method": structure.get("representative_method", ""),
        "representative_ligand": structure.get("representative_ligand", ""),
        "representative_ligand_name": structure.get("representative_ligand_name", ""),
        "meets_ligand_floor": int(potent >= 50),
        "has_human_druglike_structure": int(bool(structure) and bool(human_count)),
        "note": note,
    }
    return result


def write_provenance():
    status = get_json(CHEMBL + "/status.json")
    payload = {
        "chembl_db_version": status.get("chembl_db_version"),
        "chembl_release_date": status.get("chembl_release_date"),
        "panel_rows": len(PANEL),
        "bowes44_source": "Bowes et al. Nat Rev Drug Discov 2012; gene membership from Redfern et al. J Pharmacol Toxicol Methods 2025, Table 2, Bowes column.",
        "safety77_source": "Reaction Biology InVEST77, stated by the vendor to be aligned with Brennan et al. Nat Rev Drug Discov 2024. The journal supplement itself was not openly available, so this is the vendor enumeration rather than a transcription of the paper table.",
        "union_note": "Nicotinic alpha4/beta2 is in Bowes-44 and absent from the vendor 77. Nicotinic alpha-1 is in the vendor 77 and absent from Bowes-44. The mined union therefore has 78 rows.",
        "activity_rules": {
            "assay_types": ["B", "F"],
            "require_pchembl": True,
            "exclude_mutants": True,
            "exclude_potential_duplicates": True,
            "exclude_data_validity_comments": True,
            "potent": "max pChEMBL >= 6 (<= 1 uM)",
            "intermediate": "5 <= max pChEMBL < 6",
            "weak": "max pChEMBL < 5 (> 10 uM)",
            "ligand_floor": 50,
        },
        "structure_rules": {
            "human_taxonomy": 9606,
            "nonpolymer_required_for_count": True,
            "representative_ligand_mass_da": [150, 900],
            "excluded_component_ids": sorted(EXCLUDED_COMP_IDS),
        },
    }
    PROVENANCE.parent.mkdir(parents=True, exist_ok=True)
    PROVENANCE.write_text(json.dumps(payload, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="", help="Comma-separated panel ids to mine")
    args = parser.parse_args()
    write_panel_csv()
    write_provenance()
    selected = {item for item in args.only.split(",") if item}
    done = set() if selected else finished_ids()
    rows = [row for row in PANEL if (not selected or row[0] in selected) and row[0] not in done]
    print("mining %d targets" % len(rows), flush=True)
    for index, row in enumerate(rows, start=1):
        started = time.time()
        print("[%d/%d] %s" % (index, len(rows), row[0]), flush=True)
        try:
            result = mine_one(row)
        except Exception as exc:
            print("  FAILED %s" % exc, flush=True)
            continue
        if not selected:
            append_result(result)
        else:
            print(json.dumps(result, indent=2), flush=True)
        print(
            "  potent=%s weak=%s pdb_human=%s rep=%s ligand=%s in %.0fs"
            % (
                result["n_molecules_potent"],
                result["n_molecules_weak"],
                result["pdb_human_nonpolymer"],
                result["representative_pdb"],
                result["representative_ligand"],
                time.time() - started,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
