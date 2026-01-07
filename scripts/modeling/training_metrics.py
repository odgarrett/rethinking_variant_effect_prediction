from scipy.stats import pearsonr, spearmanr
import numpy as np

class MultiTaskMetrics:
    def __init__(self, target_config, prediction_index=0):
        """
        Args:
            target_config (list[dict]): List of task definitions (e.g. from config file).
            prediction_index (int): Index to grab from predictions tuple. 
                                    Default is 1 (based on your script), 
                                    but standard HF models often use 0 for logits.
        """
        self.target_config = target_config
        self.prediction_index = prediction_index

    def __call__(self, eval_pred):
        predictions, labels = eval_pred

        # Handle tuple outputs (common in HF models returning loss + logits)
        if isinstance(predictions, tuple):
            predictions = predictions[self.prediction_index]

        results = {}

        # Loop through each task defined in your config
        for i, target in enumerate(self.target_config):
            name = target['value_col']
            
            # Safety check: Ensure we don't go out of bounds if config doesn't match model head
            if i >= predictions.shape[1]:
                print(f"Warning: Task '{name}' index {i} is out of bounds for prediction shape {predictions.shape}")
                continue

            pred_col = predictions[:, i]
            label_col = labels[:, i]

            # In multi-task datasets, labels are often padded with -100 or NaNs
            # We must filter these out, or pearsonr will return NaN.
            mask = ~np.isnan(label_col) & (label_col != -100)
            
            clean_preds = pred_col[mask]
            clean_labels = label_col[mask]

            # Only compute if we have enough data points (need >1 for correlation)
            if len(clean_labels) > 1:
                p_corr = pearsonr(clean_preds, clean_labels)[0]
                s_corr = spearmanr(clean_preds, clean_labels)[0]
            else:
                p_corr = 0.0
                s_corr = 0.0

            results[f"pearson_{name}"] = p_corr
            results[f"spearman_{name}"] = s_corr

        # Average Spearman (handling cases where we might have no valid spearman keys)
        spearman_vals = [v for k, v in results.items() if "spearman" in k]
        results["avg_spearman"] = np.mean(spearman_vals) if spearman_vals else 0.0

        return results