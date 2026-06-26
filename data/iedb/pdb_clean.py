#!/usr/bin/env python3
"""
pdb_clean.py

For each unique pdb_key in ml_dataset.csv:
  1. Loads the PDB from pdb_cache/
  2. Keeps only the chains in the key (H, L, Ag order), first model only
  3. Strips HETATM (waters, ligands) — keeps standard amino acids only
  4. Renames chains sequentially: A, B, C  (nanobody: A, B)
  5. Sets segid = chain id on every ATOM record
  6. Saves to pdb_clean/{pdb_key}.pdb

Output files are named by pdb_key so downstream scripts can parse chain
assignments directly from the filename, e.g.:
  1TPX_H_L_A.pdb  →  cleaned, chain A=heavy, B=light, C=antigen
"""

import pandas as pd
from pathlib import Path
from Bio.PDB import PDBParser, PDBIO, Select


# -------------------------
# Config
# -------------------------
INPUT_CSV     = "/home/bri/pymol/data/sabdab/sabdab_dataset.csv"
PDB_CACHE_DIR = Path("/home/bri/pymol/data/sabdab/pdb_cache")
OUTPUT_DIR    = Path("/home/bri/pymol/data/sabdab/pdb_clean")


# -------------------------
# Biopython helpers
# -------------------------
class ChainSelect(Select):
    """Keep first model only, specified chains only, ATOM records only (no HETATM/water)."""
    def __init__(self, chain_ids):
        self.chain_ids = set(chain_ids)

    def accept_model(self, model):
        return model.id == 0    # first model only (serial_num is None for X-ray structures)

    def accept_chain(self, chain):
        return chain.id in self.chain_ids

    def accept_residue(self, residue):
        return residue.id[0] == " "     # ' ' = ATOM; 'W' = water; 'H_*' = HETATM ligand


def rename_chains(structure, src_chains: list, tgt_chains: list):
    """
    Rename chains src_chains → tgt_chains in first model.
    Two-pass via numeric temp names to avoid conflicts.
    """
    model = next(iter(structure))
    existing = {c.id for c in model}

    # Pass 1: src → temp (digits, unlikely to clash with letter chain IDs)
    for i, src in enumerate(src_chains):
        if src in existing:
            model[src].id = str(i)

    # Pass 2: temp → final
    for i, tgt in enumerate(tgt_chains):
        tmp = str(i)
        if tmp in {c.id for c in model}:
            model[tmp].id = tgt


def set_segids(pdb_path: Path):
    """
    Post-process PDB text: set segid field (cols 73-76, 0-indexed 72-76)
    to the chain id (col 22, 0-indexed 21) for every ATOM/HETATM line.
    """
    lines = pdb_path.read_text().splitlines()
    out = []
    for line in lines:
        if line.startswith(("ATOM  ", "HETATM")) and len(line) >= 22:
            chain_id = line[21]
            line = line.ljust(76)
            line = line[:72] + chain_id.ljust(4) + line[76:]
        out.append(line)
    pdb_path.write_text("\n".join(out) + "\n")


# -------------------------
# Main
# -------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    keys = sorted(df["pdb_key"].dropna().unique())
    print(f"Unique pdb_keys to process: {len(keys)}\n")

    parser = PDBParser(QUIET=True)
    io = PDBIO()
    skipped = []

    for key in keys:
        parts = key.split("_")
        pdb_id     = parts[0].lower()
        src_chains = parts[1:]                              # original chain IDs from PDB
        tgt_chains = [chr(65 + i) for i in range(len(src_chains))]  # A, B, C, ...

        pdb_path = PDB_CACHE_DIR / f"{pdb_id}.pdb"
        if not pdb_path.exists():
            print(f"SKIP  {key}  — not in cache")
            skipped.append(key)
            continue

        out_path = OUTPUT_DIR / f"{key}.pdb"
        if out_path.exists():
            print(f"EXISTS {key}  — skipping")
            continue

        structure = parser.get_structure(pdb_id, pdb_path)

        # Select chains + drop HETATM
        io.set_structure(structure)
        io.save(str(out_path), ChainSelect(src_chains))

        # Reload, rename chains, re-save
        structure = parser.get_structure(pdb_id, out_path)
        rename_chains(structure, src_chains, tgt_chains)
        io.set_structure(structure)
        io.save(str(out_path))

        # Set segids via text post-processing
        set_segids(out_path)

        print(f"{key}  {src_chains} → {tgt_chains}  →  {out_path.name}")

    print(f"\nDone.  Written: {len(keys) - len(skipped)}  /  Skipped: {len(skipped)}")
    if skipped:
        print("Skipped (not in cache):", skipped)


if __name__ == "__main__":
    main()