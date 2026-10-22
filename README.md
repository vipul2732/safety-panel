# Safety panel

Command-line prediction of secondary pharmacology against a panel of 71 human safety targets. Given a SMILES string, the program returns a ranked target list and, for each listed target, an estimate of whether an interaction at that target would be potent or weak.

## Method

Two quantities are reported, and they answer different questions.

**Target ranking.** For each panel target the score is the maximum Tanimoto similarity between the query and the potent ligands of that target in the training set. Fingerprints are ECFP4 (Morgan fingerprints, radius 2, 2048 bits). The panel score is the maximum of these 71 similarities. On a Bemis–Murcko scaffold-split test set, a measured panel target falls within the top five predictions for 96% of molecules and at rank one for 87%.

**Potent-interaction probability.** A ligand-only graph model estimates, for a nominated target, the probability that a measured interaction would be potent (pChEMBL ≥ 6) rather than weak (pChEMBL < 5). The model logit was converted to a probability by Platt scaling on validation pairs only. The probability is reported for every ranked target and, separately, for the top-ranked target. It does not alter the fingerprint ranking. A value of 0.80 means that validation pairs assigned a probability in that range were potent in about 80% of cases. Because most measured pairs in the training distribution are potent, high probabilities are common and should be read together with the fingerprint score.

## Panel

The targets are safety, or anti-target, proteins used to assess off-target liability rather than the intended therapeutic target. The panel is the union of the Bowes consensus set of 44 targets (Bowes et al., *Nat. Rev. Drug Discov.* 2012) and a 77-target safety screen. Membership of the latter follows the Reaction Biology InVEST77 enumeration, which the vendor states is aligned with Brennan et al. (*Nat. Rev. Drug Discov.* 2024). A target was retained when ChEMBL contained sufficient potent human measurements and a human ligand-bound structure was available. Seven targets in the union did not meet these criteria.

The resulting panel comprises 71 proteins: 27 G protein-coupled receptors, 19 kinases, 7 ion channels, 6 enzymes, 5 nuclear receptors, 4 transporters, 2 proteases, and 1 bromodomain. The set includes hERG (KCNH2), the muscarinic, adrenergic, and opioid receptors, the cyclooxygenases, and the monoamine transporters.

Ligands were taken from ChEMBL binding and functional assays with a reported pChEMBL value. The maximum value across retained measurements defines the class: potent at pChEMBL ≥ 6 (≤ 1 µM), weak at pChEMBL < 5 (> 10 µM). Intermediate measurements (5 ≤ pChEMBL < 6) are withheld from both classes. An unmeasured pair is not treated as inactive.

## Interpretation of the call

The call is assigned from the panel score and from the distance of the query to the training chemistry. Thresholds were fixed on the validation split and were not adjusted on the test set. Distance is defined as one minus the Tanimoto similarity to the nearest training molecule, whether or not that molecule is potent. A query identical to a training molecule therefore has distance zero.

| Call | Operating rule | Interpretation |
|---|---|---|
| Likely | Panel score ≥ 0.547 | The query is close to a known potent ligand. On the held-out test set, 95% of molecules receiving this call interact with at least one panel target. |
| Uncertain | 0.421 ≤ panel score < 0.547 | The score lies between the high-precision and high-sensitivity thresholds. Neither a positive nor a negative conclusion is supported. |
| Very unlikely | Panel score < 0.421 | The query is more distant from known potent ligands than all but 5% of validation molecules that do hit the panel. |
| Unfamiliar | Panel score < 0.421 and distance > 0.694 | The panel score is low and the query lies outside the training chemistry (beyond the 99th percentile of validation nearest-neighbour distance). The ranking should not be treated as informative. |

## Applicability

The program is intended as a computational screen during compound design, in advance of an in vitro safety-panel assay. It indicates which of the 71 liabilities a compound resembles, and whether that resemblance is characteristic of a potent ligand or of a weak analogue. It does not replace experimental profiling.

Performance is lower for chemistry distant from the training set. For approved drugs withheld from training, a true panel target is recovered in the top five predictions for 65% of panel-positive drugs, and for 48% of those with a panel score below 0.40.

## Installation

```bash
conda create -n safety-panel -c conda-forge python=3.10 rdkit numpy pytorch
conda activate safety-panel
```

## Usage

Run from this directory. A single query:

```bash
python src/predict_panel.py --smiles "CC(=O)Oc1ccccc1C(=O)O"
```

A batch file, with one SMILES per line or an identifier followed by a SMILES:

```bash
python src/predict_panel.py --input molecules.smi --top 10 --output calls.csv
```

The printed table gives rank, Tanimoto similarity, potent-interaction probability, gene symbol, protein class, and target name. The CSV contains the same fields together with the call, the panel score, and the distance to the training set.

The fingerprint reference is `data/models/fingerprint_cutoff/reference.npz`. The ligand model is `data/models/ligand_only/ligand_only.pt`, and the Platt parameters fitted on the validation pairs are in `data/models/ligand_only/potent_calibration.npz`.
