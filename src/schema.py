"""Shared feature layout for the pocket and ligand encoders."""

ELEMENTS = (6, 7, 8, 9, 15, 16, 17, 35, 53)
ELEMENT_OTHER = len(ELEMENTS)

AMINO_ACIDS = (
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
)
AA_ALIASES = {
    "MSE": "MET", "SEP": "SER", "TPO": "THR", "PTR": "TYR", "CYX": "CYS",
    "HID": "HIS", "HIE": "HIS", "HIP": "HIS",
}
ATOM_CLASSES = ("N", "CA", "C", "O", "CB")
BACKBONE = {"N", "CA", "C", "O"}

HYBRIDIZATIONS = ("SP", "SP2", "SP3", "SP3D", "OTHER")

LIGAND_DIM = 10 + 7 + 5 + 1 + len(HYBRIDIZATIONS) + 5 + 1
POCKET_DIM = 10 + (len(AMINO_ACIDS) + 1) + (len(ATOM_CLASSES) + 1) + 1

POCKET_RADIUS = 5.0
LIGAND_RADIUS = 4.5
EMBED_DIM = 128
N_LAYERS = 4
