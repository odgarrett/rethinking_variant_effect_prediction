# TODO:
    # calculate alpha
    # linear factor model
    # global epistasis model
    # apply model
from typing import Callable
import pandas as pd
import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit

def calculate_alpha(
        L_low: float,
        L_high: float,
        Kd: float,
) -> float:
    # Hill-Langmuir equation, L = ligand concentration
    def p_bound(L, Kd):
        return L / (L + Kd)

    # Calculate Sensitivity: d(P_bound) / d(ln Kd)
    # This measures how much P_bound changes for a fold-change in affinity.
    # Derivative of L/(L+Kd) wrt ln(Kd) is - L*Kd / (L+Kd)^2 = - P * (1-P)
    # The "Signal Strength" is proportional to this derivative.

    def sensitivity(L, Kd):
        P = p_bound(L, Kd)
        return P * (1 - P)

    sens_low = sensitivity(L_high, Kd) # Low sensistivity when ligand conc is high
    sens_high = sensitivity(L_low, Kd) # High sensistivity when ligand conc is low

    # Calculate Alpha (ratio of sensitivities)
    # Alpha = Signal_High / Signal_Low
    alpha = sens_high / sens_low
    return alpha


def solve_linear_factor_model(
        variables: list,
        targets: list[float],
        weights: list[float],
        coefficients: np.ndarray,
) -> pd.Series:
    """Solve a weighted linear regression model using the normal equation.
    
    Fits a linear model y = X*beta to weighted data, computing regression
    coefficients, their variances, and model fit quality (R²).
    
    Args:
        variables: List of variable/feature names for output labeling.
        targets: Target values (y) to fit, should have same length as coefficients rows.
        weights: Sample weights for weighted least squares, same length as targets.
        coefficients: Design matrix (X) of shape (n_samples, n_features).
    
    Returns:
        pd.Series with indices: [var_1, var_2, ..., var_var_1, var_var_2, ..., R2]
            - First section: regression coefficients (beta values)
            - Second section: variances of each coefficient
            - Last element: R² goodness of fit statistic
   """
    
    # Define output schema
    # [variable_1, variable_2, ...], [variance_1, variance 2, ...], [R2]
    index_names = variables + [f'var_{v}' for v in variables] + ['R2']

    # Rename variables to match linear algebra
    X = coefficients
    y = np.array(targets)
    w = np.array(weights)

    # Mask missing data
    mask = np.isfinite(y) & np.isfinite(w)
    if mask.sum() <= len(variables):
        print("Not enough data points to solve. Ensure missing data is filtered out. Returning NaNs.")
        return pd.Series([np.nan] * len(index_names), index=index_names)
    
    X = X[mask]
    y = y[mask]
    w = w[mask]

    # Weighted normal equation
    W = np.diag(w)
    XtWX = X.T @ W @ X
    XtWy = X.T @ W @ y

    # Solve for beta
    XtWX_inv = np.linalg.pinv(XtWX)
    beta = XtWX_inv @ XtWy

    # Calculate goodness of fit
    y_pred = X @ beta
    y_mean = np.average(y, weights=w)
    ss_tot = np.sum(w * (y - y_mean)**2)
    ss_res = np.sum(w * (y - y_pred)**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Calculate variance beta
    dof = len(y) - len(variables)
    sigma_sq = ss_res / dof
    beta_vars = np.diag(XtWX_inv * sigma_sq)

    # Concatenate for reporting
    results = np.concatenate([beta, beta_vars, [r2]])

    return pd.Series(results, index=index_names)


def hill_langmuir(
        features: np.ndarray,
        ddGs: np.ndarray,
        wt_dGs: np.ndarray,
        RT: float = 0.593
) -> np.ndarray:
    '''
    Hill-Langmuir model.

    Args:
        features: Shape (N_samples, N_ligands) in Molar.
        ddGs: Latent variables [ddG_fold, ddG_bind_1...] (Deltas).
        dG_WTs: Fixed baselines [dG_fold_WT, dG_bind_1_WT...] (Absolute).
        RT: Thermal energy factor (default 0.593 kcal/mol).
    '''
    # Separate params
    dG_fold = wt_dGs[0] + ddGs[0]
    binding_dGs = wt_dGs[1:] + ddGs[1:]

    # Stability
    # dG_fold sign is negative as K_fold and dG_fold represent the same direction
    #K_fold = np.exp(-dG_fold / RT)
    #p_active = K_fold / (1.0 + K_fold)
    p_active = expit(-dG_fold / RT)

    # Binding
    Kds = np.exp(binding_dGs / RT)
    p_bound = features / (features + Kds + 1e-20)


    # Final pred
    p_final = p_active * p_bound.sum(axis=1)

    return p_final


def predict_log2_fold_change(
        features: np.ndarray,
        ddGs: np.ndarray,
        wt_dGs: np.ndarray,
        RT: float = 0.593
) -> np.ndarray:
    """
    Wraps hill_langmuir to predict Log2 Fold Change relative to WT.
    """
    # Calculate mutant probability
    # Uses the ddGs provided by the solver
    p_mut = hill_langmuir(features, ddGs, wt_dGs, RT)
    
    # Calculate WT probability
    # Uses 0.0 for all ddGs (definition of Wild Type)
    ddGs_wt = np.zeros_like(ddGs)
    p_wt = hill_langmuir(features, ddGs_wt, wt_dGs, RT)
    
    # Calculate Log2 Ratio
    # Add epsilon to prevent log(0)
    epsilon = 1e-20
    return np.log2((p_mut + epsilon) / (p_wt + epsilon))


def solve_global_epistasis_model(
        variables: list,
        targets: list[float],
        weights: list[float],
        features: np.ndarray,
        wt_energies: list[float],
        model_func: Callable = predict_log2_fold_change,
        bounds: tuple | None = None,
        alpha: float = 0.1       
) -> pd.Series:
    '''
    Fits the Hill-Langmuir thermodynamic model to weighted data. Specifically, fits ddG values for each input
    variable with a weight and a bias to explain the log-fold change in fraction bound relative to wild-type.
    Uses L2 regularization to constrain the ddG values to be more realistic.

    Args:
        variables (list): Names of the latent variables (e.g., ['ddG_fold', 'ddG_bind_C', 'ddG_bind_F']),
                   Note, ddG_fold must be first.
        targets (list[float]): Observed enrichment scores.
        weights (list[float]): Sample weights (inverse variance)
        features: Matrix of ligand concentrations for each sample. Shape: (N_samples, N_conc).
        wt_dGs (list[float]): Fixed baselines [dG_fold_WT, dG_bind_1_WT...] (Absolute).
        model_func (callable): The physics function (defaults to hill_langmuir)
        bounds (tuple): Bounds for predicted ddGs. (lower_bound, upper_bound). 
        alpha (float): Regularization parameter to prevent exploding predictions.

    Returns:
        pd.Series: [beta for each ddG, variance for each ddG, R2]
    '''
    # Schema for output
    param_names = variables + ['scale_factor', 'intercept']
    index_names = param_names + [f'var_{v}' for v in param_names] + ['R2']

    y = np.array(targets)
    w = np.array(weights)
    X = features
    wt_dGs = np.array(wt_energies)

    # Mask missing data
    mask = np.isfinite(y) & np.isfinite(w)
    if mask.sum() <= len(variables):
        print("Not enough data points to solve. Ensure missing data is filtered out. Returning NaNs.")
        return pd.Series([np.nan] * len(index_names), index=index_names)
    
    X = X[mask]
    y = y[mask]
    w = w[mask]

    # If user didn't provide bounds, enforce +/- 20 kcal/mol
    if bounds is None:
        limit = 20.0
        # Lower bound, Upper bound for each variable
        # +-20 for ddG, +-10 factor signal amplification, +-5 score offset
        lb = [-limit] * len(variables) + [0.1, -5.0]
        ub = [limit] * len(variables) + [10.0, 5.0]
        bounds = (lb, ub)

    # Define weighted residuals callback function for minimization
    # sum(weight * (observed - predicted)^2)
    def residuals(params):
        # Unpack: [Physics_Params, Scale, Intercept]
        beta_physics = params[:-2]
        A = params[-2]
        B = params[-1]
        
        # Pure Physics Prediction
        raw_pred = model_func(X, beta_physics, wt_dGs)
        
        # Scaled Prediction
        y_pred = A * raw_pred + B
        
        # Error + Regularization (only penalize physics betas, not scale/bias)
        err = np.sqrt(w) * (y - y_pred)
        penalty = np.sqrt(alpha) * beta_physics
        return np.concatenate([err, penalty])
    
    # Solve
    initial_guess = np.concatenate([np.zeros(len(variables)), [1.0, 0.0]])

    try:
        # Fit model
        res = least_squares(residuals, initial_guess, bounds=bounds, method='trf')
        beta_all = res.x

        # Unpack
        beta_phys = beta_all[:-2]
        A, B = beta_all[-2:]

        # Get prediction using parameters
        pred = A * model_func(X, beta_phys, wt_dGs) + B

        # Calculate pure residuals for R2
        pure_res = (y - pred) * np.sqrt(w)

        # Calculate Statistics
        ss_res = np.sum(pure_res**2)
        y_mean = np.average(y, weights=w)
        ss_tot = np.sum(w * (y - y_mean)**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Estimate Variance (Covariance Matrix)
        # Cov = (J.T * J)^-1 * MSE
        dof = len(y) - len(variables)
        mse = ss_res / dof if dof > 0 else ss_res
        
        J = res.jac
        try:
            cov_matrix = np.linalg.pinv(J.T @ J) * mse
            beta_vars = np.diag(cov_matrix)
        except:
            beta_vars = np.zeros(len(beta_all))

        results = np.concatenate([beta_all, beta_vars, [r2]])
        return pd.Series(results, index=index_names)

    except Exception as e:
        return pd.Series([np.nan] * len(index_names), index=index_names)


def apply_biophysical_model(
    df: pd.DataFrame,
    model_type: str,
    variables: list,
    sample_cols: list,
    design_info: np.ndarray,
    wt_energies: list[float] | None = None,
    alpha: float | None = None
) -> pd.DataFrame:
    """
    Applies a biophysical solver (Linear or Global Epistasis) variant-by-variant.

    Args:
        df: Dataframe containing variant data. Must have columns f'score_{sample}' 
            and f'var_{sample}' for each sample in sample_cols.
        model_type: 'linear_factor' or 'global_epistasis'.
        variables: Names of the latent variables to solve for.
        sample_cols: List of sample names (e.g., ['A1', 'A2', ...]) matching dataframe columns.
        design_info: 
            - If linear: Design matrix (X) of shape (n_samples, n_variables).
            - If epistasis: Concentration matrix of shape (n_samples, n_ligands).
        wt_energies: (Required for global_epistasis) List of [dG_fold_WT, dG_bind_WT...].
        alpha: (Required for global_epistasis) Regularization strength.

    Returns:
        pd.DataFrame: Original df with new columns for betas, variances, and R2.
    """
    print(f"Fitting {model_type} model to {len(df)} variants...")

    # Validate Inputs
    score_cols = [f'score_{s}' for s in sample_cols]
    var_cols = [f'var_{s}' for s in sample_cols]
    
    missing_cols = [c for c in score_cols + var_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in dataframe: {missing_cols}")

    if model_type == 'global_epistasis' and wt_energies is None:
        raise ValueError("Must provide `wt_energies` for global epistasis model.")

    # Define Row Processor
    def _fit_row(row):
        # Extract Targets
        targets = row[score_cols].values.astype(float)
        
        # Extract Variances and convert to Weights (1/sigma^2)
        variances = row[var_cols].values.astype(float)
        
        # Handle zero or infinite variance safely
        # If variance is 0 (or very close), weight becomes infinite. 
        # We clip variance to a tiny epsilon to keep weights finite but large.
        variances = np.maximum(variances, 1e-12)
        weights = 1.0 / variances
        
        # Dispatch to Solver
        if model_type == 'linear_factor':
            return solve_linear_factor_model(
                variables=variables,
                targets=targets,
                weights=weights,
                coefficients=design_info
            )
            
        elif model_type == 'global_epistasis':
            return solve_global_epistasis_model(
                variables=variables,
                targets=targets,
                weights=weights,
                features=design_info,
                wt_energies=wt_energies, # type: ignore
                model_func=predict_log2_fold_change,
                alpha=alpha # type: ignore
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    # Apply to DataFrame
    # axis=1 applies the function to each row
    results_df = df.apply(_fit_row, axis=1)

    # Merge Results
    # This concatenates the new columns (beta_*, var_*, R2) horizontally
    return pd.concat([df, results_df], axis=1)