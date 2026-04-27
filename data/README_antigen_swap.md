# Antigen Swap Script

A CLI tool for shuffling or augmenting target assignments in CSV datasets to create negative samples for machine learning.

## Overview

This script processes CSV files containing biological data (e.g., protein sequences with targets) and creates negative samples by assigning different targets to each row. It can either:

- **Shuffle mode**: Assign one random different target per row
- **Augment mode**: Create all possible negative combinations for each row (data augmentation)

The assigned targets are guaranteed to exist in the dataset and be different from the original target.

## Requirements

- Python 3.x
- pandas
- numpy
- argparse (built-in)

Install dependencies:
```bash
pip install pandas numpy
```

## Usage

### Basic Syntax
```bash
python antigen_swap.py <input_file> [options]
```

### Options
- `input_file`: Path to input CSV file (required)
- `--output_file`: Path to output CSV file (optional, defaults to `{input_dir}/{input_name}_target_swap.csv` or `{input_name}_augmented.csv`)
- `--target_col`: Column name for targets (default: 'target')
- `--label_col`: Column name for labels (default: 'label')
- `--negative_label`: Value for negative labels (default: 0)
- `--augment`: Enable augmentation mode to create all possible negative combinations

### Examples

#### Standard Shuffling
```bash
python antigen_swap.py data.csv
```
Creates `data_target_swap.csv` with one random different target per row.

#### Augmentation
```bash
python antigen_swap.py data.csv --augment
```
Creates `data_augmented.csv` with all possible negative combinations (e.g., if each row has 5 alternative targets, output has 5x rows).

#### Custom Output and Columns
```bash
python antigen_swap.py input.csv --output_file output.csv --target_col my_target --label_col my_label --negative_label -1
```

#### Full Path Example
```bash
python antigen_swap.py /path/to/input.csv --augment --output_file /path/to/output.csv
```

## Output

- **Standard mode**: Same number of rows as input, each with a randomly assigned different target and label=0
- **Augmentation mode**: Multiple rows per input row (one for each possible alternative target), significantly expanding the dataset

## Error Handling

- Checks for file existence
- Validates column presence
- Ensures alternative targets are available
- Provides clear error messages

## Use Cases

- Creating negative samples for binary classification in drug-target interaction prediction
- Data augmentation for imbalanced datasets
- Generating diverse training examples for machine learning models

## Notes

- The script assumes the input CSV has a column with target identifiers
- All assigned targets exist in the original dataset
- No rows are removed; all data is preserved with modified targets