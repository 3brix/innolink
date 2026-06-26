"""
renumber_batch.py  —  run with: pymol -c renumber_batch.py

For every .pdb in PDB_CLEAN_DIR:
  - loads the file
  - renumbers each chain from 1 using the connectivity-based renumber()
  - saves to PDB_RENUM_DIR under the same filename
"""

import glob
import os
from pymol import cmd

# -------------------------
# Config
# -------------------------
PDB_CLEAN_DIR = "/home/bri/pymol/data/iedb/pdb_clean"
PDB_RENUM_DIR = "/home/bri/pymol/data/iedb/pdb_renumbered"

RENUMBER_PY   = "/home/bri/pymol/data/renumber.py"   # path to your renumber.py


# -------------------------
# Load renumber plugin
# -------------------------
cmd.run(RENUMBER_PY)   # registers renumber() into cmd


# -------------------------
# Batch
# -------------------------
os.makedirs(PDB_RENUM_DIR, exist_ok=True)

pdb_files = sorted(glob.glob(os.path.join(PDB_CLEAN_DIR, "*.pdb")))
print(f"Files to process: {len(pdb_files)}")

for pdb_file in pdb_files:
    fname = os.path.basename(pdb_file)
    obj   = fname.replace(".pdb", "")

    out_path = os.path.join(PDB_RENUM_DIR, fname)
    if os.path.exists(out_path):
        print(f"EXISTS {fname} — skipping")
        continue

    cmd.load(pdb_file, obj)

    # Renumber each chain independently starting from 1
    chains = cmd.get_chains(obj)
    for chain in chains:
        cmd.do(f"renumber {obj} and chain {chain}, start=1")

    cmd.save(out_path, obj)
    cmd.delete(obj)
    print(f"Done  {fname}  chains={chains}")

print(f"\nFinished. Output in: {PDB_RENUM_DIR}")
cmd.quit()