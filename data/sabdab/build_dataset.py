#!/usr/bin/env python3
"""
Build a flat dataset from SAbDab peptide-antigen co-crystal structures.

Output columns:
    assay_id, epitope_sequence, epitope_name, epitope_start, epitope_end,
    pdb_id, label, quant_value, receptor_name, heavy_seq, light_seq

Notes:
- label=1 for all rows (all SAbDab entries are co-crystal complexes).
- quant_value prefers 'affinity', falls back to 'delta_g'.
- Multi-antigen rows ("B | I") produce one row per antigen chain.
- Nanobodies (Lchain=NaN) are included; light_seq is None for them.
- Each PDB is parsed only once (cached).
- Deduplication: rows with identical (pdb_id, heavy_seq, light_seq, epitope_sequence)
  are collapsed to one. This removes redundant crystallographic copies while keeping
  genuinely different antibodies that happen to bind the same antigen chain.
- assay_id = {PDB_ID}_{H_chain}_{antigen_chain}, unique after deduplication.
"""
import glob, os, sys
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PDB_DIR    = os.path.join(SCRIPT_DIR, "sabdab_ds")
OUT_PATH   = os.path.join(SCRIPT_DIR, "sabdab_peptide_dataset.csv")

tsv_candidates = glob.glob(os.path.join(PDB_DIR, "*.tsv"))
if not tsv_candidates:
    sys.exit(f"ERROR: No .tsv found in {PDB_DIR}")
TSV_PATH = tsv_candidates[0]

AA3 = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C',
    'GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
    'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P',
    'SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V',
    'MSE':'M','SEC':'U','PYL':'O',
}

def parse_pdb_fast(pdb_path):
    """Parse SEQRES and ATOM records with plain text scanning.
    Returns:
      seqs:   {chain_id: one-letter sequence}
      ranges: {chain_id: (first_resnum, last_resnum)}
    """
    seqres = {}
    atom_res = {}
    with open(pdb_path) as fh:
        for line in fh:
            rec = line[:6].strip()
            if rec == "SEQRES":
                chain = line[11]
                residues = line[19:].split()
                seqres.setdefault(chain, []).extend(residues)
            elif rec == "ATOM":
                chain = line[21]
                try:
                    resnum = int(line[22:26])
                except ValueError:
                    continue
                if chain not in atom_res:
                    atom_res[chain] = [resnum, resnum]
                else:
                    if resnum < atom_res[chain][0]: atom_res[chain][0] = resnum
                    if resnum > atom_res[chain][1]: atom_res[chain][1] = resnum
    seqs   = {c: "".join(AA3.get(aa, "X") for aa in codes) for c, codes in seqres.items()}
    ranges = {c: tuple(v) for c, v in atom_res.items()}
    return seqs, ranges


def pick_quant(row):
    for col in ("affinity", "delta_g"):
        val = row.get(col)
        if pd.notna(val):
            return val
    return None


df_meta = pd.read_csv(TSV_PATH, sep="\t")
print(f"TSV: {len(df_meta)} rows, {df_meta['pdb'].nunique()} unique PDBs")

unique_pdbs = df_meta["pdb"].str.lower().str.strip().unique()
print(f"Parsing {len(unique_pdbs)} PDB files ...")

pdb_cache = {}
for i, pdb_id in enumerate(unique_pdbs, 1):
    pdb_path = os.path.join(PDB_DIR, f"{pdb_id}.pdb")
    if os.path.exists(pdb_path):
        pdb_cache[pdb_id] = parse_pdb_fast(pdb_path)
    if i % 200 == 0:
        print(f"  {i}/{len(unique_pdbs)} ...")

print(f"Parsed {len(pdb_cache)}. Missing: {len(unique_pdbs) - len(pdb_cache)}")

records = []
skipped = []

for _, row in df_meta.iterrows():
    pdb_id      = str(row["pdb"]).lower().strip()
    h_chain     = str(row["Hchain"]).strip()
    l_raw       = row["Lchain"]
    is_nanobody = pd.isna(l_raw)
    l_chain     = str(l_raw).strip()

    if pdb_id not in pdb_cache:
        skipped.append((pdb_id, "PDB missing")); continue

    seqs, ranges = pdb_cache[pdb_id]
    heavy_seq = seqs.get(h_chain)
    light_seq = None if is_nanobody else seqs.get(l_chain)

    if not heavy_seq:
        skipped.append((pdb_id, f"No SEQRES for H-chain '{h_chain}'")); continue
    if not is_nanobody and not light_seq:
        skipped.append((pdb_id, f"No SEQRES for L-chain '{l_chain}'")); continue

    ag_chains = [c.strip() for c in str(row["antigen_chain"]).split("|") if c.strip() and c.strip() != "nan"]
    ag_names  = [n.strip() for n in str(row["antigen_name"]).split("|")]

    if not ag_chains:
        skipped.append((pdb_id, "No antigen chain")); continue

    for i, ag_chain in enumerate(ag_chains):
        epi_seq = seqs.get(ag_chain)
        if not epi_seq:
            skipped.append((pdb_id, f"No SEQRES for antigen chain '{ag_chain}'")); continue
        epi_start, epi_end = ranges.get(ag_chain, (None, None))
        epi_name = ag_names[i] if i < len(ag_names) else ag_names[0]
        records.append({
            "assay_id":         f"{pdb_id.upper()}_{h_chain}_{ag_chain}" if is_nanobody else f"{pdb_id.upper()}_{h_chain}_{l_chain}_{ag_chain}",
            "epitope_sequence": epi_seq,
            "epitope_name":     epi_name,
            "epitope_start":    epi_start,
            "epitope_end":      epi_end,
            "pdb_id":           pdb_id.upper(),
            "label":            1,
            "quant_value":      pick_quant(row),
            "receptor_name":    str(row["compound"]).strip(),
            "heavy_seq":        heavy_seq,
            "light_seq":        light_seq,
        })

cols = [
    "assay_id","epitope_sequence","epitope_name","epitope_start","epitope_end",
    "pdb_id","label","quant_value","receptor_name","heavy_seq","light_seq",
]
out_df = pd.DataFrame(records, columns=cols)

# Deduplicate on sequence identity: same heavy + light + epitope sequences
# regardless of which PDB they came from. Keeps the first occurrence.
n_before = len(out_df)
out_df = out_df.drop_duplicates(subset=["heavy_seq","light_seq","epitope_sequence"])
n_after = len(out_df)

out_df.to_csv(OUT_PATH, index=False)
print(f"\nDone: {n_after} rows ({n_before - n_after} redundant copies removed) -> {OUT_PATH}")
if skipped:
    print(f"\nSkipped {len(skipped)}:")
    for pdb, reason in skipped:
        print(f"  {pdb}: {reason}")