

import pandas as pd
import numpy as np
import argparse
import os

def shuffle_targets(input_file, output_file=None, target_col='target', label_col='label', negative_label=None, augment=False):
    """
    Assign each row a random target different from its original target, or augment by creating all possible combinations.
    The assigned target must exist in the dataset.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file (optional, auto-generated if None)
        target_col (str): Column name for targets (default: 'target')
        label_col (str): Column name for labels (default: 'label')
        negative_label (int): Value for negative labels (default: 0)
        augment (bool): If True, create all possible negative combinations for augmentation
    """
    # Check if input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Read data
    df = pd.read_csv(input_file, sep= ";")
    
    # Check if target column exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in columns: {list(df.columns)}")
    
    # Generate output file if not provided
    if output_file is None:
        input_dir = os.path.dirname(input_file)
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        suffix = "_augmented" if augment else "_target_swap"
        output_file = os.path.join(input_dir, f"{base_name}{suffix}.csv")
    
    unique_targets = df[target_col].unique()
    
    if augment:
        # Augment: create all possible negative combinations
        augmented_rows = []
        for _, row in df.iterrows():
            orig = row[target_col]
            possible = [t for t in unique_targets if t != orig]
            for alt_target in possible:
                new_row = row.copy()
                new_row[target_col] = alt_target
                new_row[label_col] = negative_label
                augmented_rows.append(new_row)
        df_neg = pd.DataFrame(augmented_rows)
    else:
        # Standard: assign one random different target per row
        df_neg = df.copy()
        original_targets = df[target_col].values
        
        new_targets = []
        for orig in original_targets:
            possible = [t for t in unique_targets if t != orig]
            if not possible:
                raise ValueError(f"No alternative targets available for original target '{orig}'. Dataset may have only one unique target.")
            new_targets.append(np.random.choice(possible))
        
        df_neg[target_col] = new_targets
        df_neg[label_col] = negative_label
    
    # Save to output
    df_neg.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")
    print(f"Original rows: {len(df)}, Output rows: {len(df_neg)}")
    
    return df_neg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assign random targets different from originals, or augment with all combinations.")
    parser.add_argument("input_file", help="Path to input CSV file")
    parser.add_argument("--output_file", help="Path to output CSV file (optional, auto-generated)")
    parser.add_argument("--target_col", default="target", help="Column name for targets (default: target)")
    parser.add_argument("--label_col", default="label", help="Column name for labels (default: label)")
    parser.add_argument("--negative_label", type=int, default=0, help="Value for negative labels (default: None)")
    parser.add_argument("--augment", action="store_true", help="Augment dataset by creating all possible negative combinations")
    
    args = parser.parse_args()
    
    try:
        shuffle_targets(args.input_file, args.output_file, args.target_col, args.label_col, args.negative_label, args.augment)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)