import pandas as pd
import numpy as np

def calculate_pooled_overdispersion(
    df: pd.DataFrame,
    group_by: list | str,
    value_cols: list | str | None = None,
    data_type: str = 'scores',
    min_replicates: int = 2
) -> pd.DataFrame:
    """
    Calculates global overdispersion (experimental noise) by pooling variance across replicates.
    Automatically detects Wide vs. Long logic based on the inputs provided.

    Args:
        df (pd.DataFrame): Input dataframe.
        group_by (list | str): Column(s) defining the replicate groups.
                               - Wide: e.g. 'aa_substitutions'
                               - Long: e.g. ['aa_substitutions', 'sample_name']
                               (Note: For Long data, the Sample column must be the LAST in the list).
        value_cols (list | str | None): The column(s) containing the data to analyze.
                                        - If list: Treated as multiple samples (Wide).
                                        - If str: Treated as a single value column (Long or Single-Sample Wide).
                                        - If None: Auto-detects columns starting with 'score_' or 'count_'.
        data_type (str): 'scores' (Excess Variance) or 'counts' (Dispersion Index).
        min_replicates (int): Minimum size of a group to include in calculation.

    Returns:
        pd.DataFrame: Summary of noise metrics per sample.
    """
    # Normalize Arguments
    if isinstance(group_by, str): 
        group_by = [group_by]
    
    # Resolve value_cols
    if value_cols is None:
        # Auto-detect standard naming conventions (implies Wide format usually)
        prefix = 'score_' if data_type == 'scores' else 'count_'
        target_cols = [c for c in df.columns if c.startswith(prefix)]
        if not target_cols:
            raise ValueError(f"No columns found starting with '{prefix}'. Please specify value_cols.")
    elif isinstance(value_cols, str):
        target_cols = [value_cols]
    else:
        target_cols = value_cols

    print(f"Calculating overdispersion for {data_type}...")
    
    # Aggregate
    # Group by the provided keys and calculate stats for all target columns
    # observed=True is critical for memory safety
    stats = df.groupby(group_by, observed=True)[target_cols].agg(['var', 'mean', 'count'])

    results = []

    # Case A: Multiple Columns (Wide Format)
    # The columns themselves represent different samples.
    if len(target_cols) > 1 or len(group_by) == 1:
        for col in target_cols:
            sub_stats = stats[col]
            valid = sub_stats[sub_stats['count'] >= min_replicates]
            
            sample_name = col.replace('score_', '').replace('count_', '')
            res = {'sample': sample_name, 'n_variants_used': len(valid)}

            if data_type == 'scores':
                res['pooled_variance_mean'] = valid['var'].mean()
                res['pooled_variance_median'] = valid['var'].median()
            else:
                # Dispersion Index logic
                with np.errstate(divide='ignore', invalid='ignore'):
                    d_index = valid['var'] / valid['mean']
                d_index = d_index[np.isfinite(d_index)]
                
                res['dispersion_index_mean'] = d_index.mean()
                res['dispersion_index_median'] = d_index.median()
            
            results.append(res)

    # Case B: Single Column + Multiple Groups (Long Format)
    # The sample distinction is hidden in the grouping index (e.g. Variant, Sample).
    else:
        col = target_cols[0]
        sub_stats = stats[col]
        valid = sub_stats[sub_stats['count'] >= min_replicates]

        # Calculate raw metrics for every variant/sample pair
        if data_type == 'scores':
            raw_metrics = valid['var']
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                raw_metrics = valid['var'] / valid['mean']
            raw_metrics = raw_metrics[np.isfinite(raw_metrics)]

        # Aggregate BY SAMPLE (The last level of the group_by list)
        sample_level = group_by[-1]
        
        # Group the metrics by sample and summarize
        summary = raw_metrics.groupby(level=sample_level).agg(['mean', 'median', 'count'])
        
        for sample_name, row in summary.iterrows():
            res = {'sample': sample_name, 'n_variants_used': int(row['count'])}
            
            if data_type == 'scores':
                res['pooled_variance_mean'] = row['mean']
                res['pooled_variance_median'] = row['median']
            else:
                res['dispersion_index_mean'] = row['mean']
                res['dispersion_index_median'] = row['median']
            results.append(res)

    return pd.DataFrame(results)