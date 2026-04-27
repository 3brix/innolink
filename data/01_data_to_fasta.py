import pandas as pd
import argparse
import os

def process_data(input_data_file, target_file, merge_key='target'):
    """
    Process input data by merging with targets and generating CSV and FASTA outputs.
    
    Args:
        input_data_file (str): Path to the input data CSV (e.g., germinal.csv or nb_scaffold_paper_examples.csv)
        target_file (str): Path to the targets CSV (e.g., targets.txt)
        merge_key (str): Column name to merge on (default: 'target')
    """
    # Check if files exist
    if not os.path.exists(input_data_file):
        raise FileNotFoundError(f"Input data file not found: {input_data_file}")
    if not os.path.exists(target_file):
        raise FileNotFoundError(f"Target file not found: {target_file}")
    
    # Read data
    input_df = pd.read_csv(input_data_file)
    targets_df = pd.read_csv(target_file)
    
    # Standardize target columns to lowercase
    targets_df.columns = targets_df.columns.str.lower()
    
    # Ensure merge key exists
    if merge_key not in input_df.columns:
        raise ValueError(f"Merge key '{merge_key}' not found in input data columns: {list(input_df.columns)}")
    if merge_key not in targets_df.columns:
        raise ValueError(f"Merge key '{merge_key}' not found in target columns: {list(targets_df.columns)}")
    
    # Merge
    merged_df = pd.merge(input_df, targets_df, on=merge_key)
    
    # Generate output filenames
    base_name = os.path.splitext(os.path.basename(input_data_file))[0]
    csv_output = f"{base_name}_targets.csv"
    fasta_output = f"{base_name}_targets.fasta"
    
    # Save CSV
    merged_df.to_csv(csv_output, index=False)
    print(f"Saved merged data to {csv_output}")
    
    # Generate FASTA
    generate_fasta(merged_df, fasta_output)
    
    return merged_df

def generate_fasta(df, filename, name_col='name', target_col='target', prot_seq_col='protein_sequence', targ_seq_col='target_sequence'):
    """
    Generate FASTA file from DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with sequence data
        filename (str): Output FASTA filename
        name_col (str): Column name for name
        target_col (str): Column name for target
        prot_seq_col (str): Column name for protein sequence
        targ_seq_col (str): Column name for target sequence
    """
    with open(filename, 'w') as f:
        for _, row in df.iterrows():
            header = f">{row[name_col]}{row[target_col]}"
            seq = f"{row[prot_seq_col]}:{row[targ_seq_col]}"
            f.write(f"{header}\n{seq}\n")
    print(f"FASTA file generated: {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge input data with targets and generate CSV and FASTA outputs.")
    parser.add_argument("input_data_file", help="Path to the input data CSV file (e.g., germinal.csv)")
    parser.add_argument("target_file", help="Path to the targets CSV file (e.g., targets.txt)")
    parser.add_argument("--merge_key", default="target", help="Column name to merge on (default: target)")
    
    args = parser.parse_args()
    
    try:
        process_data(args.input_data_file, args.target_file, args.merge_key)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)