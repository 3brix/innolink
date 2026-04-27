import pandas as pd
import argparse
import os

def process_data(master_file):
    """
    Process master data file containing both input data and targets, generating CSV and FASTA outputs.
    
    Args:
        master_file (str): Path to the master CSV file with combined input data and targets
    """
    # Check if file exists
    if not os.path.exists(master_file):
        raise FileNotFoundError(f"Master file not found: {master_file}")
    
    # Read data
    df = pd.read_csv(master_file, sep=';')
    
    # Required columns
    required_cols = ['vhh', 'target', 'vhh_sequence', 'target_sequence']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in master file: {missing_cols}")
    
    # Generate output filenames
    base_name = os.path.splitext(os.path.basename(master_file))[0]
    csv_output = f"{base_name}_processed.csv"
    fasta_output = f"{base_name}_processed.fasta"
    
    # Save CSV (optional, since it's already processed)
    df.to_csv(csv_output, index=False)
    print(f"Saved processed data to {csv_output}")
    
    # Generate FASTA
    generate_fasta(df, fasta_output)
    
    return df

def generate_fasta(df, filename, name_col='vhh', target_col='target', prot_seq_col='vhh_sequence', targ_seq_col='target_sequence'):
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
            header = f">{row[name_col]}_{row[target_col]}"
            seq = f"{row[prot_seq_col]}:{row[targ_seq_col]}"
            f.write(f"{header}\n{seq}\n")
    print(f"FASTA file generated: {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process master data file with combined input data and targets, generating CSV and FASTA outputs.")
    parser.add_argument("master_file", help="Path to the master CSV file with combined data")
    
    args = parser.parse_args()
    
    try:
        process_data(args.master_file)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)