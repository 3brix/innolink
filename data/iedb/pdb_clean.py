#!/usr/bin/env python3
"""
pdb_clean.py

For each unique pdb_key in the dataset CSV:
  1. Load the PDB from PDB_CACHE_DIR (first model only)
  2. Keep only the chains named in the key, in H, (L,) Ag order
  3. Strip HETATM (waters, ligands) -> standard amino-acid ATOM records only
  4. Rename chains sequentially A, B, C  (nanobody, no VL: A, B)
  5. Set segid = chain id on every ATOM record
  6. Save to OUTPUT_DIR/{pdb_key}.pdb

Key format (auto-detected by number of chain tokens after the PDB id):
  Antibody : PDB_H_L_Ag   ->  A=heavy, B=light, C=antigen
  Nanobody : PDB_H_Ag     ->  A=heavy, B=antigen           (no VL)

The chain mapping is purely positional, so nanobodies need no special case:
a missing VL is simply one fewer chain token, and the antigen slides from
C to B automatically.

Robustness: any key that references a chain absent from the PDB coordinates
(e.g. an antibody whose light chain was never resolved/deposited) is skipped
and reported, so a mislabeled file is never written.

Performance: each PDB is parsed once and written once (the previous version
parsed twice and wrote three times). Structures are cached per PDB id and
deep-copied per key, so PDBs shared by several keys are read from disk once.
"""

import copy
import pandas as pd
from pathlib import Path
from Bio.PDB import PDBParser, PDBIO, Select


# -------------------------
# Config
# -------------------------
INPUT_CSV     = "/home/bri/pymol/data/sabdab/sabdab_dataset.csv"
PDB_CACHE_DIR = Path("/home/bri/pymol/data/sabdab/pdb_nb")
OUTPUT_DIR    = Path("/home/bri/pymol/data/sabdab/pdb_nb_clean2")


# -------------------------
# Biopython helpers
# -------------------------
class ChainSelect(Select):
    """Keep first model only, specified chains only, standard ATOM residues
    only (drops waters and HETATM ligands)."""
    def __init__(self, chain_ids):
        self.chain_ids = set(chain_ids)

    def accept_model(self, model):
        return model.id == 0            # first model only

    def accept_chain(self, chain):
        return chain.id in self.chain_ids

    def accept_residue(self, residue):
        return residue.id[0] == " "     # ' ' = ATOM; 'W' = water; 'H_*' = HETATM


def first_model(structure):
    return next(iter(structure))


def prune_to_chains(structure, keep_chains):
    """Keep only `keep_chains` in the first model and drop every other model.
    Removing unrelated chains BEFORE renaming is what prevents a collision with
    a chain that already happens to be named A/B/C."""
    model = first_model(structure)
    for m in list(structure):
        if m.id != model.id:
            structure.detach_child(m.id)
    keep = set(keep_chains)
    for cid in [c.id for c in model if c.id not in keep]:
        model.detach_child(cid)


def rename_chains(structure, src_chains, tgt_chains):
    """Rename src_chains -> tgt_chains in the first model. Two-pass via unique
    multi-char temp ids so that (a) a source id equal to a target id cannot
    clash, and (b) the temp ids cannot collide with any 1-char PDB chain id.
    The temp ids are transient in memory and never written to disk.
    Assumes the model has already been pruned to exactly src_chains."""
    model = first_model(structure)
    for i, src in enumerate(src_chains):
        model[src].id = f"__tmp{i}__"
    for i, tgt in enumerate(tgt_chains):
        model[f"__tmp{i}__"].id = tgt


def set_segids(pdb_path):
    """Post-process PDB text: set the segid field (cols 73-76) to the chain id
    (col 22) for every ATOM/HETATM line."""
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
    struct_cache = {}       # pdb_id -> parsed structure (parse each PDB once)
    written, skipped = 0, []

    for key in keys:
        parts = key.split("_")
        pdb_id = parts[0].lower()
        src_chains = parts[1:]                                   # e.g. [H,L,Ag] or [H,Ag]
        tgt_chains = [chr(65 + i) for i in range(len(src_chains))]

        pdb_path = PDB_CACHE_DIR / f"{pdb_id}.pdb"
        if not pdb_path.exists():
            print(f"SKIP  {key}  - not in cache")
            skipped.append((key, "not in cache"))
            continue

        out_path = OUTPUT_DIR / f"{key}.pdb"
        if out_path.exists():
            print(f"EXISTS {key}  - skipping")
            continue

        # parse once per pdb_id, deep-copy so per-key renaming stays independent
        if pdb_id not in struct_cache:
            struct_cache[pdb_id] = parser.get_structure(pdb_id, pdb_path)
        structure = copy.deepcopy(struct_cache[pdb_id])

        # verify every requested chain actually exists (catches an Ab whose
        # VL is missing from the coordinates -> would otherwise mislabel).
        present = {c.id for c in first_model(structure)}
        missing = [c for c in src_chains if c not in present]
        if missing:
            role = "Ab" if len(src_chains) == 3 else "Nb" if len(src_chains) == 2 else "?"
            print(f"SKIP  {key}  - chains {missing} absent from coords ({role})")
            skipped.append((key, f"missing chains {missing}"))
            continue

        # prune to the requested chains FIRST (drops any unrelated A/B/C chains
        # and extra models), then rename, then a single save (+ HETATM strip)
        prune_to_chains(structure, src_chains)
        rename_chains(structure, src_chains, tgt_chains)
        io.set_structure(structure)
        io.save(str(out_path), ChainSelect(tgt_chains))
        set_segids(out_path)

        kind = "Nb" if len(src_chains) == 2 else "Ab"
        print(f"{key}  {src_chains} -> {tgt_chains}  ({kind})  ->  {out_path.name}")
        written += 1

    print(f"\nDone.  Written: {written}  /  Skipped: {len(skipped)}")
    if skipped:
        print("Skipped:")
        for key, why in skipped:
            print(f"  {key}: {why}")


if __name__ == "__main__":
    main()