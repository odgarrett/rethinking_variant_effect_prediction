import pandas as pd
import numpy as np

def get_simple_counts_df(
        df: pd.DataFrame,
        group_by: str,
        cols_to_keep: list | None = None,
        samples_to_keep: list | None = None
) -> pd.DataFrame:
    '''
    Simplify a variant count dataframe.
    Aggregates counts by the specified group, fills missing samples/variant pairs with 0,
    and returns a clean, long-format DataFrame.

    Args:
        df (pd.DataFrame): Input dataframe containing 'sample', 'count', and the group_by column.
        group_by (str): The column to aggregate by (e.g., 'barcode', 'aa_substitutions').
        cols_to_keep: Additional columns to keep (e.g. ['aa_substitutions'])
        samples_to_keep (list, optional): List of sample names to include. 

    Returns:
        pd.DataFrame: Long-format dataframe with columns ['sample', group_by, 'count'].
    '''
    if cols_to_keep is None:
        cols_to_keep = []
    
    # Aggregate counts by sample and specified group
    agg_df = df.groupby(['sample', group_by] + cols_to_keep, observed=True)['count'].sum().reset_index()

    # Fill missing sample/group pairs with 0 counts
    # Use pivot to move samples to columns and fill in values
    pivot_index = [group_by] + cols_to_keep

    wide_df = agg_df.pivot_table(
        index=pivot_index,
        columns='sample',
        values='count',
        fill_value=0,
        observed=True
    ).reset_index()

    # If samples_to_keep is specified, filter columns
    if samples_to_keep is not None:
        # Only keep columns that are in the requested list (plus the index column)
        valid_samples = [s for s in samples_to_keep if s in wide_df.columns]
        
        # Check for missing samples to help debugging
        missing = set(samples_to_keep) - set(wide_df.columns)
        if missing:
            print(f"Warning: The following requested samples were not found: {missing}")
            
        wide_df = wide_df[pivot_index + valid_samples]

    # Melt back to long format
    long_df = wide_df.melt(
        id_vars=pivot_index,
        var_name='sample',
        value_name='count'
    )
    
    return long_df


def calculate_functional_scores(
    df: pd.DataFrame, 
    preselection_sample: str, 
    postselection_sample: str | list | None = None, 
    group_col: str = 'aa_substitutions', 
    cols_to_keep: list | None = None,
    sequence_type: str = 'aa_sub', 
    wt_sequence: str | None = None,
    pseudocount: float = 0.5,
    return_format: str = 'wide'
) -> pd.DataFrame:
    """
    Calculates enrichment scores (log-ratios) with flexible grouping and robust memory handling.

    Args:
        df (pd.DataFrame): Long-format dataframe containing ['sample', 'count', group_col].
        preselection_sample (str): Name of the input/library sample (denominator).
        postselection_sample (str or list, optional): Name(s) of output sample(s). 
                                                      If None, calculates for ALL other samples.
        group_col (str): The column name to index variants by (e.g., 'barcode', 'aa_substitutions').
        cols_to_keep (list, optional): Additional metadata columns to preserve (e.g., ['aa_substitutions']).
        sequence_type (str): The nature of the group_col data. Used for WT normalization logic.
                             Options: 'aa_sub' (e.g., "A52G"), 'dna' (sequences), 'aa_seq' (sequences).
        wt_sequence (str, optional): The specific WT sequence string. Required if sequence_type is 
                                     'dna' or 'aa_seq'.
        pseudocount (float): Added to counts before log.
        return_format (str): 'wide' (default) or 'long'. 
                             - Wide: Columns are score_A, var_A, count_A...
                             - Long: Rows are (Variant, Sample), columns are score, var, count, pre_count.

    Returns:
        pd.DataFrame: Wide or Long format dataframe with scores.
    """
    print(f"Calculating scores grouping by '{group_col}' (type: {sequence_type})...")
    
    if cols_to_keep is None: cols_to_keep = []

    # Aggregate data, summing counts for any duplicate entries of (sample + group_col + metadata)
    agg_df = df.groupby(['sample', group_col] + cols_to_keep, observed=True)['count'].sum().reset_index()

    # Pivot table, aligning Pre and Post samples side-by-side. 
    # Index becomes a MultiIndex if cols_to_keep is used.
    wide_df = agg_df.pivot_table(
        index=[group_col] + cols_to_keep, 
        columns='sample', 
        values='count', 
        fill_value=0,
        observed=True 
    )
    
    # Identify Wild Type (WT) for Normalization
    # Since index might be MultiIndex, we extract the specific group_col level for checking
    group_index = wide_df.index.get_level_values(group_col)
    wt_mask = np.zeros(len(wide_df), dtype=bool)
    
    if sequence_type == 'aa_sub':
        index_lower = group_index.astype(str).str.lower()
        wt_mask = (group_index == '') | \
                  (index_lower == 'synonymous') | \
                  (index_lower == 'wildtype') | \
                  (index_lower == 'wt')
                  
    elif sequence_type in ['dna', 'aa_seq']:
        if wt_sequence is None:
            print(f"Warning: sequence_type='{sequence_type}' but no `wt_sequence` provided.")
            print("Scores will NOT be normalized by WT change (assumes WT ratio = 1).")
        else:
            wt_mask = (group_index == wt_sequence)
            if not wt_mask.any():
                print(f"Warning: WT sequence '{wt_sequence}' not found in table.")

    # Calculate WT Constants (Pre-selection)
    if wt_mask.any():
        wt_pre_count = wide_df.loc[wt_mask, preselection_sample].sum()
        if wt_pre_count == 0:
            print("Warning: Wild Type has 0 reads in pre-selection. Normalization may be unstable.")
    else:
        wt_pre_count = 0 
    
    wt_pre_p = wt_pre_count + pseudocount
    n_pre_p = wide_df[preselection_sample] + pseudocount

    # Determine Target Samples
    if postselection_sample:
        if isinstance(postselection_sample, str):
            targets = [postselection_sample]
        else:
            targets = postselection_sample
    else:
        targets = [c for c in wide_df.columns if c != preselection_sample]

    # Calculate Scores & Variances
    results = pd.DataFrame(index=wide_df.index)
    
    # We store the pre-selection count. 
    pre_count_col_name = f'count_{preselection_sample}'
    results[pre_count_col_name] = wide_df[preselection_sample]

    for target in targets:
        if target not in wide_df.columns:
            print(f"Warning: Sample '{target}' not found in data. Skipping.")
            continue

        # WT stats for this target
        if wt_mask.any():
            wt_post_count = wide_df.loc[wt_mask, target].sum()
        else:
            wt_post_count = 0
            
        wt_post_p = wt_post_count + pseudocount
        n_post_p = wide_df[target] + pseudocount

        # --- The Score Formula ---
        ratio = (n_post_p / n_pre_p) / (wt_post_p / wt_pre_p)
        results[f'score_{target}'] = np.log2(ratio)

        # --- Variance Formula ---
        term_var = (1/n_post_p) + (1/n_pre_p) + (1/wt_post_p) + (1/wt_pre_p)
        results[f'var_{target}'] = term_var / (np.log(2)**2)
        
        # Keep raw count
        results[f'count_{target}'] = wide_df[target]

    # Handle Return Formats
    results = results.reset_index()

    if return_format == 'long':
        # Rename the pre-selection count so it doesn't get melted into the generic 'count' column
        results = results.rename(columns={pre_count_col_name: 'pre_count'})
        
        # Reshape: Converts score_A, var_A, count_A -> [A, score, var, count]
        # We include cols_to_keep in 'i' so they are preserved as identifier columns
        long_df = pd.wide_to_long(
            results,
            stubnames=['score', 'var', 'count'],
            i=[group_col, 'pre_count'] + cols_to_keep,
            j='post_sample',
            sep='_',
            suffix='.+' # Catch-all suffix
        ).reset_index()
        
        return long_df

    return results


def clean_variant_data(
        df: pd.DataFrame,
        count_columns: list,
        read_count_threshold: int,
        aa_column: str = 'aa_substitutions',
        variants_to_remove: dict | None = None,
        pivot_on: str | None = None,
        return_format: str = 'wide'
) -> pd.DataFrame:
    """
    Cleans variant data by filtering low counts, stop codons, and custom denylists.
    Optionally pivots long data to wide before filtering.

    Args:
        df (pd.DataFrame): The input dataframe.
        count_columns (list): List of columns to sum for thresholding. 
                              If pivoting, this should be the single value column (e.g. ['count']).
        read_count_threshold (int): Minimum summed count required across count_columns.
        aa_column (str): Name of the column containing amino acid info. Rows with '*' dropped.
        variants_to_remove (dict, optional): Dictionary of {col: [values_to_drop]}.
        pivot_on (str, optional): Column to pivot to wide format (e.g., 'sample').
        return_format (str): 'wide' or 'long'. Determines output shape.

    Returns:
        pd.DataFrame: The cleaned dataframe.
    """
    initial_len = len(df)
    
    # Filter stop codons
    if aa_column in df.columns:
        df = df[~df[aa_column].astype(str).str.contains('*', regex=False)]

    # Filter denylists (Do before pivoting to ensure columns exist)
    if variants_to_remove:
        for col, bad_values in variants_to_remove.items():
            if not isinstance(bad_values, list): bad_values = [bad_values]
            if col in df.columns:
                df = df[~df[col].isin(bad_values)]

    # Pivot if requested (Assumes input is Long)
    index_cols = None
    if pivot_on:
        # Infer index columns (all cols except pivot key and value)
        index_cols = [c for c in df.columns if c != pivot_on and c not in count_columns]
        
        # Pivot using the first count_column as values
        df = df.pivot_table(
            index=index_cols, 
            columns=pivot_on, 
            values=count_columns[0], 
            fill_value=0, 
            observed=True
        ).reset_index()
        df.columns.name = None
        
        # Update count_columns to be the new numeric columns (samples) for thresholding
        count_columns = [c for c in df.columns if c not in index_cols]

    # Filter by read depth
    if count_columns:
        total_counts = df[count_columns].sum(axis=1)
        df = df[total_counts >= read_count_threshold]

    # Reshape return format
    if return_format == 'long' and pivot_on:
        # Melt back if we pivoted but user wants long output
        df = df.melt(id_vars=index_cols, var_name=pivot_on, value_name='count')

    print(f"Cleaning complete. {len(df)} variants remaining (from {initial_len}).")
    return df