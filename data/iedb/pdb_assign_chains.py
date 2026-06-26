#!/usr/bin/env python3
"""
pdb_assign_chains.py

For each row in ml_dataset.csv that has a pdb_id, downloads the PDB structure,
identifies the heavy (H), light (L), and antigen (Ag) chains by sequence alignment
against the IEDB sequences, and writes a pdb_key column:

  Conventional antibody:  {PDB_ID}_{H_chain}_{L_chain}_{Ag_chain}  e.g. 1TPX_H_L_A
  Nanobody:               {PDB_ID}_{H_chain}_{Ag_chain}             e.g. 5M00_A_B

Rows without a pdb_id, or where alignment fails, get pdb_key = NaN.

Output: ml_dataset.csv is updated in place (pdb_key column added/replaced).
A summary of failed assignments is printed at the end.
"""

import re
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from Bio.PDB import PDBParser, PPBuilder
from Bio.Align import PairwiseAligner


# -------------------------
# Config — edit paths here
# -------------------------
INPUT_CSV  = "/home/bri/pymol/data/iedb/iedb_dataset.csv"
OUTPUT_CSV = "/home/bri/pymol/data/iedb/iedb_dataset.csv"   # overwrite in place
PDB_CACHE_DIR = Path("/home/bri/pymol/data/iedb/pdb_cache")


# -------------------------
# PDB download & parsing
# -------------------------
def download_pdb(pdb_id: str, cache_dir: Path) -> Path | None:
    pdb_id = pdb_id.lower()
    path = cache_dir / f"{pdb_id}.pdb"
    if path.exists():
        return path
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        r = requests.get(url, timeout=30)
    except requests.RequestException as e:
        print(f"  WARNING: network error for {pdb_id}: {e}")
        return None
    if r.status_code != 200:
        print(f"  WARNING: could not download {pdb_id} (HTTP {r.status_code})")
        return None
    path.write_text(r.text)
    return path


def get_chain_sequences(pdb_path: Path) -> dict:
    """Return {chain_id: amino_acid_sequence} for first model only."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, pdb_path)
    ppbuilder = PPBuilder()
    chains = {}
    model = next(iter(structure))
    for chain in model:
        segs = ppbuilder.build_peptides(chain)
        seq = "".join(str(pp.get_sequence()) for pp in segs)
        if seq:
            chains[chain.id] = seq
    return chains


# -------------------------
# Sequence alignment
# -------------------------
def make_aligner(local: bool = False) -> PairwiseAligner:
    aligner = PairwiseAligner()
    aligner.mode = "local" if local else "global"
    aligner.match_score = 1
    aligner.mismatch_score = 0
    aligner.open_gap_score = -1
    aligner.extend_gap_score = -0.1
    return aligner


def best_match(query: str, chains: dict, aligner: PairwiseAligner,
               exclude: set = None) -> tuple:
    """
    Find the chain whose sequence best aligns to query.
    identity = alignment_score / len(query)
    Always returns the best-matching chain — PDB chain naming is too inconsistent
    to rely on thresholds. Score is returned for logging/diagnostics.
    Returns (chain_id, identity_score) or (None, 0.0) if no chains available.
    """
    if not query or not chains:
        return None, 0.0
    best_id, best_score = None, 0.0
    for chain_id, chain_seq in chains.items():
        if exclude and chain_id in exclude:
            continue
        try:
            score = next(iter(aligner.align(query, chain_seq))).score
        except StopIteration:
            continue
        identity = score / len(query)
        if identity > best_score:
            best_score = identity
            best_id = chain_id
    return best_id, best_score


def strip_modifications(epitope: str) -> str:
    """
    Remove IEDB modification annotations and keep only the AA sequence.
    e.g. "CGADSYEMEEDGVRKC + OX(C1, C16)"  ->  "CGADSYEMEEDGVRKC"
    """
    return re.sub(r"\s*\+.*", "", str(epitope)).strip()


# -------------------------
# Main
# -------------------------
def main():
    PDB_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    df["pdb_key"] = pd.Series(dtype=object)  # object dtype — avoids LossySetitemError when assigning strings

    ab_aligner = make_aligner(local=False)
    ag_aligner = make_aligner(local=True)   # epitope is a short fragment → local search

    has_pdb = df["pdb_id"].notna()
    print(f"Rows with pdb_id: {has_pdb.sum()} / {len(df)}\n")

    pdb_chain_cache = {}   # avoid re-parsing the same PDB for multiple rows
    failed = []

    for idx, row in df[has_pdb].iterrows():
        pdb_id = str(row["pdb_id"]).strip()
        print(f"[{idx}] {pdb_id}", end="  ")

        # Load chains (download once, cache in memory)
        if pdb_id not in pdb_chain_cache:
            pdb_path = download_pdb(pdb_id, PDB_CACHE_DIR)
            pdb_chain_cache[pdb_id] = get_chain_sequences(pdb_path) if pdb_path else {}
        chains = pdb_chain_cache[pdb_id]

        if not chains:
            print("→ SKIP (no chains parsed)")
            failed.append((idx, pdb_id, "download/parse failed"))
            continue

        assigned = set()

        # --- Heavy chain ---
        h_chain, h_score = best_match(row["heavy_seq"], chains, ab_aligner)
        if h_chain:
            assigned.add(h_chain)

        # --- Light chain (skipped for nanobodies) ---
        l_chain, l_score = None, 0.0
        if not row["is_nanobody"]:
            light_seq = row["light_seq"] if row["light_seq"] else ""
            l_chain, l_score = best_match(light_seq, chains, ab_aligner,
                                          exclude=assigned)
            if l_chain:
                assigned.add(l_chain)

        # --- Antigen chain ---
        epitope_clean = strip_modifications(row["epitope_sequence"])
        ag_chain, ag_score = best_match(epitope_clean, chains, ag_aligner,
                                        exclude=assigned)

        # --- Build key ---
        if row["is_nanobody"]:
            parts = [pdb_id.upper(), h_chain, ag_chain]
        else:
            parts = [pdb_id.upper(), h_chain, l_chain, ag_chain]

        if any(p is None for p in parts[1:]):
            # Only happens when chains dict was empty or epitope/sequence was blank
            key = None
            failed.append((idx, pdb_id,
                           f"no chains matched: H={h_chain}({h_score:.2f}) L={l_chain}({l_score:.2f}) Ag={ag_chain}({ag_score:.2f})"))
        else:
            key = "_".join(parts)

        df.at[idx, "pdb_key"] = key
        print(f"H={h_chain}({h_score:.2f})  L={l_chain}({l_score:.2f})  Ag={ag_chain}({ag_score:.2f})  →  {key}")

    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n{'='*60}")
    print(f"pdb_key assigned : {df['pdb_key'].notna().sum()} / {has_pdb.sum()}")
    print(f"Failed / partial : {len(failed)}")
    if failed:
        print("\nFailed rows:")
        for idx, pdb_id, reason in failed:
            print(f"  row {idx:>5}  {pdb_id}  —  {reason}")
    print(f"\nSaved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()