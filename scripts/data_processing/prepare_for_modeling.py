from pathlib import Path
import pandas as pd
import numpy as np
from Bio import SeqIO
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from datasets import Dataset


def get_full_sequence(
        mut_string: str,
        wt_seq: str,
) -> str:
    '''
    Expands point mutation strings into full sequences.

    Args:
        mut_string (str): Mutation string (e.g., "L24P A50G")
        wt_seq (str): Wild-type sequence to mutate.

    Returns:
        str: Mutated sequence.
    '''
    if pd.isna(mut_string) or mut_string == "":
        return wt_seq
    
    seq_list = list(wt_seq)
    
    # Parse "L24P A50G"
    for mut in mut_string.split():
        if len(mut) < 3 or not mut[1:-1].isdigit(): continue
        idx = int(mut[1:-1]) - 1
        new_aa = mut[-1]
        if 0 <= idx < len(seq_list):
            seq_list[idx] = new_aa
    return "".join(seq_list)


def expand_mut_to_seq(
        df: pd.DataFrame,
        mut_col: str,
        wt_seq_path: str,
        new_col_name: str = "full_sequence"
):
    '''
    Expands all point mutation strings in a DataFrame into their full sequences.

    Args:
        df (pd.DataFrame): DataFrame containing mutant data.
        mut_col (str): Name of column containing mutation strings (e.g., "L24P A50G")
        wt_seq_path (str): Path to FASTA file containing wild-type sequence to mutate.
        new_col_name (str): Name of new column with expanded sequences.

    Returns:
        pd.DataFrame: Input DataFrame with added new_col_name column.
    '''

    # Load WT sequence
    wt_record = SeqIO.read(wt_seq_path, "fasta")
    wt_seq = str(wt_record.seq)

    # Generate Sequences
    df[new_col_name] = df[mut_col].apply(lambda x: get_full_sequence(x, wt_seq))

    return df


import pandas as pd
import numpy as np

def add_weights(
    variant_df: pd.DataFrame,
    overdispersion_df: pd.DataFrame | None = None, # <--- Now Optional
    data_type: str = 'scores',
    overdispersion_sample_col: str = 'sample',
    overdispersion_metric_col: str = 'pooled_variance_median',
    variant_sample_col: str | None = None,
    value_cols: list | str | None = None
) -> pd.DataFrame:
    """
    Adds weight columns to the variant dataframe for weighted loss functions.
    If overdispersion_df is provided, weights include the system noise/dispersion.
    If None, defaults to pure Inverse Variance (scores) or Weight=1 (counts).
    
    Args:
        variant_df (pd.DataFrame): Dataframe with variant data.
        overdispersion_df (pd.DataFrame, optional): Summary table from calculate_pooled_overdispersion.
        data_type (str): 'scores' or 'counts'.
        overdispersion_sample_col (str): Column in overdispersion_df with sample names.
        overdispersion_metric_col (str): Column in overdispersion_df with the noise metric.
        variant_sample_col (str, optional): [Long Only] Column in variant_df with sample names.
        value_cols (list | str, optional): 
            - Wide: List of columns (e.g. ['var_A1', 'var_A2']).
            - Long: Single column name containing the variance/count data (e.g. 'var').

    Returns:
        pd.DataFrame: variant_df with added weight column(s).
    """
    df = variant_df.copy()
    
    # Prepare Dispersion Map (if provided)
    dispersion_map = {}
    if overdispersion_df is not None:
        if overdispersion_sample_col not in overdispersion_df.columns:
            raise ValueError(f"Column '{overdispersion_sample_col}' not found in overdispersion table.")
            
        dispersion_map = pd.Series(
            overdispersion_df[overdispersion_metric_col].values,
            index=overdispersion_df[overdispersion_sample_col]
        ).to_dict()
        
        print(f"Adding weights using {data_type} approach with overdispersion correction...")
    else:
        print(f"Adding weights using {data_type} approach (Pure Inverse Variance, no overdispersion)...")

    # Case A: Long format
    if variant_sample_col:
        # Resolve value column
        if value_cols is None:
            val_col = 'var' if data_type == 'scores' else 'count'
        else:
            val_col = value_cols if isinstance(value_cols, str) else value_cols[0]

        # Determine Noise/Dispersion values
        if overdispersion_df is not None:
            # Map specific noise metrics to rows
            noise_series = df[variant_sample_col].map(dispersion_map)
            
            # Check for unmapped samples
            if noise_series.isna().any():
                missing = df.loc[noise_series.isna(), variant_sample_col].unique()
                print(f"  Warning: Samples {missing} have no dispersion metric. Weight set to 0.")
                noise_series = noise_series.fillna(np.inf) # Results in weight 0
        else:
            # Default values if no table provided
            # Scores: Add 0.0 noise
            # Counts: Divide by 1.0 (Standard Poisson)
            default_val = 0.0 if data_type == 'scores' else 1.0
            noise_series = default_val

        # Calculate Weights
        if data_type == 'scores':
            # Weight = 1 / (Row_Variance + System_Noise)
            # If noise is 0 (no table), this is pure inverse variance
            total_var = df[val_col] + noise_series
            df['weight'] = 1.0 / (total_var + 1e-8)
            
        elif data_type == 'counts':
            # Weight = 1 / Dispersion_Index
            # If D=1 (no table), weight is 1.0
            if isinstance(noise_series, pd.Series):
                noise_series = noise_series.clip(lower=1.0)
            df['weight'] = 1.0 / noise_series

    # Case B: Wide format
    else:        
        # Resolve columns
        prefix = 'var_' if data_type == 'scores' else 'count_'
        if value_cols is None:
            target_cols = [c for c in df.columns if c.startswith(prefix)]
        else:
            target_cols = value_cols if isinstance(value_cols, list) else [value_cols]

        for col in target_cols:
            sample_name = col.replace(prefix, '')
            weight_col = f"weight_{sample_name}"
            
            # Get metric value
            if overdispersion_df is not None:
                if sample_name not in dispersion_map:
                    print(f"  Warning: No dispersion metric found for '{sample_name}'. Skipping.")
                    continue
                metric_val = dispersion_map[sample_name]
            else:
                # Defaults
                metric_val = 0.0 if data_type == 'scores' else 1.0

            # Calculate Weight
            if data_type == 'scores':
                total_var = df[col] + metric_val
                df[weight_col] = 1.0 / (total_var + 1e-8)
            
            elif data_type == 'counts':
                effective_d = max(1.0, metric_val)
                df[weight_col] = 1.0 / effective_d
            
    return df


def train_val_test_split(
        df: pd.DataFrame, 
        data_split_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''
    Split dataframe into train/val/test (80/10/10) and save as CSV files.

    Args:
        df (pd.DataFrame): DataFrame to split.
        data_split_dir (Path): Directory where `train.csv`, `val.csv`, and `test.csv` will be written.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train_df, val_df, test_df)
    '''

    # Split data, 80-10-10
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    # Save split data
    train_df.to_csv(str(data_split_dir / "train.csv"), index=False)
    val_df.to_csv(str(data_split_dir / "val.csv"), index=False)
    test_df.to_csv(str(data_split_dir / "test.csv"), index=False)

    return train_df, val_df, test_df


def create_hf_dataset(
        df: pd.DataFrame, 
        tokenizer, 
        target_config: list[dict], 
        seq_col: str,
        wt_aa_seq: str,
        mut_col: str | None = None                      
) -> Dataset:
    '''
    Create a HuggingFace Dataset with tokenized sequences, labels/weights, and mutation masks if desired. 

    Args:
        df (pd.DataFrame): DataFrame containing sequences and target columns.
        tokenizer: HuggingFace tokenizer for sequence encoding.
        target_config (dict): Configuration list with 'value_col' and 'weight_col' for each task.
        seq_col (str): Column name containing amino acid sequences.
        wt_aa_seq (str): Wild-type sequence (used to set max_length for tokenization).
        mut_col (str): Name of the column containing mutation strings.

    Returns:
        Dataset: HuggingFace Dataset with input_ids, attention_mask, labels, and sample_weights.
    '''
    ds = Dataset.from_pandas(df)
    
    def tokenize_and_mask(sequences):
        tokenized = tokenizer(
            sequences[seq_col], 
            padding="max_length", 
            truncation=True, 
            max_length=len(wt_aa_seq)+2
        )
        
        # Generate mask if mutations provided
        if mut_col:
            batch_masks = []
            
            for mut_str in sequences[mut_col]:
                # Initialize zero-mask matching the tokenized length
                seq_len = len(tokenized["input_ids"][0]) 
                mask = [0] * seq_len
                
                # Check for empty/NaN mutation strings
                if mut_str is not None and not pd.isna(mut_str) and mut_str != "":
                    # Parse string: "L24P A50G"
                    for mutation in mut_str.split():
                        if len(mutation) < 2: continue
                        
                        # Extract position (Bio Index 1-based)
                        pos_str = "".join(filter(str.isdigit, mutation))
                        
                        if pos_str:
                            # Convert: Bio_Pos -> Token_Index (+1 for CLS)
                            token_idx = int(pos_str) 
                            
                            if token_idx < seq_len:
                                mask[token_idx] = 1
                
                batch_masks.append(mask)

            tokenized["mutation_mask"] = batch_masks

        # Pack Targets into a single Matrix [Batch, Num_Tasks]
        labels = []
        weights = []
        for t in target_config:
            labels.append(sequences[t['value_col']])
            weights.append(sequences[t['weight_col']])
        
        # Transpose lists to match batch format
        tokenized["labels"] = list(zip(*labels))       # -> [[y1, y2, y3], [y1, y2, y3]...]
        tokenized["sample_weights"] = list(zip(*weights)) 
        
        return tokenized

    ds = ds.map(tokenize_and_mask, batched=True)

    # Set columns, depending on mutations
    torch_cols = ["input_ids", "attention_mask", "labels", "sample_weights"]
    
    if mut_col:
        torch_cols.append("mutation_mask")

    ds.set_format("torch", columns=torch_cols)
    return ds


def create_split_datasets(
        model_checkpoint: str, 
        target_config: list[dict], 
        wt_aa_seq: str, 
        seq_col: str, 
        data_split_dir: Path | None = None, 
        train_df: pd.DataFrame | None = None, 
        val_df: pd.DataFrame | None = None, 
        test_df: pd.DataFrame | None = None, 
        save_to_disk: bool = False,
        mut_col: str | None = None
) -> tuple[Dataset, Dataset, Dataset]:
    '''
    Create HuggingFace Datasets for train/val/test splits with tokenized sequences and labels.

    Args:
        model_checkpoint (str): Model checkpoint name for initializing tokenizer.
        target_config (list[dict]): Configuration list with 'value_col' and 'weight_col' for each task.
        wt_aa_seq (str): Wild-type sequence (used for tokenization max_length).
        seq_col (str): Column name containing amino acid sequences.
        data_split_dir (Path): Directory containing split CSV files or where datasets will be saved.
        train_df (pd.DataFrame, optional): Provided training DataFrame. If None, loads from data_split_dir/train.csv.
        val_df (pd.DataFrame, optional): Provided validation DataFrame. If None, loads from data_split_dir/val.csv.
        test_df (pd.DataFrame, optional): Provided test DataFrame. If None, loads from data_split_dir/test.csv.
        save_to_disk (bool): If True, saves datasets to data_split_dir/{train_ds, val_ds, test_ds}.
        mut_col (str): Name of the column containing mutation strings.
        
    Returns:
        tuple[Dataset, Dataset, Dataset]: (train_ds, val_ds, test_ds)
    '''
    if not data_split_dir:
        data_split_dir = Path('')

    # Load in tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # Load in data if df not provided
    if train_df is None:
        train_df = pd.read_csv(str(data_split_dir / "train.csv"))
    if val_df is None:
        val_df = pd.read_csv(str(data_split_dir / "val.csv"))
    if test_df is None:
        test_df = pd.read_csv(str(data_split_dir / "test.csv"))

    train_ds = create_hf_dataset(train_df, tokenizer, target_config, seq_col, wt_aa_seq, mut_col)
    val_ds = create_hf_dataset(val_df, tokenizer, target_config, seq_col, wt_aa_seq, mut_col)
    test_ds = create_hf_dataset(test_df, tokenizer, target_config, seq_col, wt_aa_seq, mut_col)

    if save_to_disk:    
        train_ds.save_to_disk(str(data_split_dir / "train_ds"))
        val_ds.save_to_disk(str(data_split_dir / "val_ds"))
        test_ds.save_to_disk(str(data_split_dir / "test_ds"))

    return train_ds, val_ds, test_ds