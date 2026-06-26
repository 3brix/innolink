import pandas as pd

# -----------------------------
# Read files
# -----------------------------
bcell = pd.read_excel("/home/bri/pymol/data/iedb/bcell_table_export_sh.xlsx")
receptor = pd.read_excel("/home/bri/pymol/data/iedb/receptor_table_export_sh.xlsx")


# -----------------------------
# Clean column names
# -----------------------------
def clean_cols(df):
    df.columns = (
        df.columns
        .astype(str)
        .str.replace("\xa0", " ")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return df

bcell = clean_cols(bcell)
receptor = clean_cols(receptor)


# -----------------------------
# Extract assay_id (bcell)
# -----------------------------
iri_col = [c for c in bcell.columns if "Assay" in c and "IRI" in c]
iri_col = iri_col[0] if iri_col else None

id_col = [c for c in bcell.columns if "Assay" in c and "ID" in c and "IRI" not in c]
id_col = id_col[0] if id_col else None

if id_col:
    bcell["assay_id"] = pd.to_numeric(bcell[id_col], errors="coerce")
elif iri_col:
    bcell["assay_id"] = (
        bcell[iri_col]
        .astype(str)
        .str.extract(r"/assay/(\d+)")[0]
    )
    bcell["assay_id"] = pd.to_numeric(bcell["assay_id"], errors="coerce")
else:
    raise ValueError("No assay ID column found in bcell")


# -----------------------------
# Label (MANDATORY)
# -----------------------------
qual_col = [c for c in bcell.columns if "qual" in c.lower()][0]

bcell["label_raw"] = bcell[qual_col].astype(str).str.lower()

bcell["label"] = None
bcell.loc[bcell["label_raw"].str.contains("positive"), "label"] = 1
bcell.loc[bcell["label_raw"].str.contains("negative"), "label"] = 0

bcell = bcell.dropna(subset=["label"])


# -----------------------------
# Quantitative measurement (NEW)
# -----------------------------
quant_col = [c for c in bcell.columns if "Quantitative" in c]

if quant_col:
    quant_col = quant_col[0]

    bcell["quant_raw"] = bcell[quant_col]

    bcell["quant_value"] = (
        bcell[quant_col]
        .astype(str)
        .str.extract(r"([0-9]+\.?[0-9]*)")[0]
    )

    bcell["quant_value"] = pd.to_numeric(
        bcell["quant_value"],
        errors="coerce"
    )
else:
    bcell["quant_raw"] = None
    bcell["quant_value"] = None


# -----------------------------
# bcell subset
# -----------------------------
bcell_small = bcell[
    [
        "assay_id",
        "Epitope - Name",
        "Epitope - Reference Name",
        "Epitope - Starting Position",
        "Epitope - Ending Position",
        "Complex - PDB ID",
        "label",
        "quant_value"
    ]
]


# -----------------------------
# Receptor assay_id
# The "Assay - IEDB IDs" column contains comma-separated lists (e.g. "310, 2014388").
# str.extract() only captures the first ID, silently dropping all others and
# causing negatives whose IDs appear later in the list to vanish after the merge.
# Fix: split on commas and explode so every ID gets its own row.
# -----------------------------
rec_id_col = [c for c in receptor.columns if "Assay" in c and "ID" in c][0]

receptor["assay_id"] = receptor[rec_id_col].astype(str).str.split(r",\s*")
receptor = receptor.explode("assay_id")
receptor["assay_id"] = pd.to_numeric(receptor["assay_id"], errors="coerce")
receptor = receptor.dropna(subset=["assay_id"])


# -----------------------------
# Receptor subset
# -----------------------------
receptor_small = receptor.rename(
    columns={
        "Chain 1 - Protein Sequence": "heavy_seq",
        "Chain 2 - Protein Sequence": "light_seq",
        "Receptor - Reference Name": "receptor_name"
    }
)[
    [
        "assay_id",
        "receptor_name",
        "heavy_seq",
        "light_seq"
    ]
]

print("BEFORE merge:")
print(bcell["label"].value_counts(dropna=False))


# -----------------------------
# MERGE
# -----------------------------
dataset = bcell_small.merge(
    receptor_small,
    on="assay_id",
    how="inner"
)

print("AFTER merge:")
print(dataset["label"].value_counts(dropna=False))


# -----------------------------
# ML FILTER
# -----------------------------
# light_seq is optional — nanobodies (VHH) have only a single chain.
mandatory_cols = [
    "Epitope - Name",
    "receptor_name",
    "heavy_seq",
    "label"
]

dataset = dataset.dropna(subset=mandatory_cols)

# Flag nanobodies (single-chain receptors)
dataset["is_nanobody"] = dataset["light_seq"].isna().astype(int)


# -----------------------------
# STANDARDIZE COLUMN NAMES
# -----------------------------
dataset = dataset.rename(columns={

    # epitope
    "Epitope - Name": "epitope_sequence",
    "Epitope - Reference Name": "epitope_name",
    "Epitope - Starting Position": "epitope_start",
    "Epitope - Ending Position": "epitope_end",

    # structure
    "Complex - PDB ID": "pdb_id"
})


# -----------------------------
# DEDUP: flag label conflicts, then deduplicate
# Same (epitope, heavy_seq, light_seq) can appear in multiple assays.
# Pairs with conflicting labels across assays are saved separately for review.
# For nanobodies light_seq is NaN — fill with "" so groupby works correctly.
# -----------------------------
dataset["light_seq"] = dataset["light_seq"].fillna("")
bio_key = ["epitope_sequence", "heavy_seq", "light_seq"]

label_counts = dataset.groupby(bio_key)["label"].nunique()
conflict_keys = label_counts[label_counts > 1].reset_index()[bio_key]

conflicts = dataset.merge(conflict_keys, on=bio_key, how="inner")
clean = dataset.merge(conflict_keys, on=bio_key, how="left", indicator=True)
clean = clean[clean["_merge"] == "left_only"].drop(columns="_merge")

# Deduplicate clean pairs — keep the row with highest quant_value (most informative),
# falling back to first occurrence.
clean = clean.sort_values("quant_value", ascending=False)
clean = clean.drop_duplicates(subset=bio_key, keep="first")

print(f"Conflicting pairs (saved separately): {conflicts[bio_key].drop_duplicates().shape[0]}")
print(f"Clean unique pairs: {len(clean)}")
print("Clean label counts:")
print(clean["label"].value_counts())

dataset = clean


# -----------------------------
# COLUMN ORDER
# -----------------------------
dataset = dataset[
    [
        "assay_id",
        "epitope_sequence",
        "epitope_name",
        "epitope_start",
        "epitope_end",
        "pdb_id",
        "label",
        "quant_value",
        "receptor_name",
        "heavy_seq",
        "light_seq",
        "is_nanobody"
    ]
]

conflicts = conflicts[
    [
        "assay_id",
        "epitope_sequence",
        "epitope_name",
        "epitope_start",
        "epitope_end",
        "pdb_id",
        "label",
        "quant_value",
        "receptor_name",
        "heavy_seq",
        "light_seq",
        "is_nanobody"
    ]
]


# -----------------------------
# SAVE OUTPUT
# -----------------------------
out_path = "/home/bri/pymol/data/iedb/iedb_dataset.csv"
conflicts_path = "/home/bri/pymol/data/iedb/iedb_dataset_conflicts.csv"
dataset.to_csv(out_path, index=False)
conflicts.to_csv(conflicts_path, index=False)


# -----------------------------
# DEBUG
# -----------------------------
print(dataset.head())
print("Final rows:", len(dataset))
print("Unique assays:", dataset["assay_id"].nunique())
print("Saved to:", out_path)