# Data to FASTA Script

A CLI tool for merging biological data with target information and generating CSV and FASTA outputs for sequence analysis.

## Overview

This script processes CSV files containing biological data (e.g., protein sequences) and target information, merges them based on a common key, and generates:

- A merged CSV file with combined data
- A FASTA file with sequences formatted for bioinformatics tools

It's designed for preparing data for tasks like protein-ligand interaction prediction or sequence alignment.

## Requirements

- Python 3.x
- pandas
- argparse (built-in)

Install dependencies:
```bash
pip install pandas
```

## Usage

### Basic Syntax
```bash
python data_to_fasta.py <input_data_file> <target_file> [options]
```

### Arguments
- `input_data_file`: Path to input data CSV file (e.g., germinal.csv or nb_scaffold_paper_examples.csv)
- `target_file`: Path to targets CSV file (e.g., targets.txt)

### Options
- `--merge_key`: Column name to merge on (default: 'target')
- `--name_col`: Column name for sequence names (default: 'name')
- `--target_col`: Column name for targets (default: 'target')
- `--prot_seq_col`: Column name for protein sequences (default: 'protein_sequence')
- `--targ_seq_col`: Column name for target sequences (default: 'sequence')

## Output

- **CSV**: `{input_data_name}_targets.csv` - Merged data with all columns
- **FASTA**: `{input_data_name}_targets.fasta` - Sequences in FASTA format with headers like `>name+target`

## Examples

### Basic Usage
```bash
python data_to_fasta.py germinal.csv targets.txt
```
Merges `germinal.csv` with `targets.txt` on 'target' column, creates `germinal_targets.csv` and `germinal_targets.fasta`.

### Custom Merge Key
```bash
python data_to_fasta.py data.csv targets.csv --merge_key custom_key
```

### Custom Column Names
```bash
python data_to_fasta.py input.csv targets.csv --name_col seq_name --prot_seq_col prot_seq --targ_seq_col targ_seq
```

## FASTA Format

The FASTA file contains entries like:
```
>name+target
protein_sequence:target_sequence
```

## Error Handling

- Checks for file existence
- Validates merge key presence in both files
- Ensures required columns exist
- Provides clear error messages

## Use Cases

- Preparing data for molecular docking simulations
- Generating training data for ML models predicting protein-target interactions
- Creating input files for sequence alignment tools
- Formatting data for bioinformatics pipelines

## Notes

- Target file columns are standardized to lowercase
- Input data columns are used as-is
- FASTA headers combine name and target with '+' separator
- Sequences are concatenated with ':' separator