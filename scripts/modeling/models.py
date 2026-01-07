from transformers import AutoModel
import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """Projection head mapping sequence embeddings to task outputs.

    Can be a single `nn.Linear` (when `hidden_dim` is None or 0) or a small
    MLP with one hidden layer. Intended as the final mapping from an ESM
    embedding to one or more scalar predictions.

    Args:
        input_dim (int): Input embedding size (ESM hidden size).
        output_dim (int): Number of outputs (tasks or physics params).
        hidden_dim (int | None): If None or 0, uses a single linear layer.
        dropout (float): Dropout probability for the MLP hidden layer.

    Attributes:
        net (nn.Module): The mapping module (Linear or Sequential MLP).
    """
    def __init__(self, input_dim, output_dim, hidden_dim=None, dropout=0.0):
        super().__init__()
        
        if hidden_dim is None or hidden_dim == 0:
            # Simple Linear Regressor
            self.net = nn.Linear(input_dim, output_dim)
        else:
            # MLP with one hidden layer
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim)
            )

    def forward(self, x):
        return self.net(x)


class EsmClsNHeads(nn.Module):
    """ESM model predicting multiple task outputs from the CLS embedding.

    Wraps a pretrained ESM backbone and a `ProjectionHead` that regresses
    from the CLS token embedding (position 0) to `num_tasks` scalar outputs.

    Args:
        model_name (str): HuggingFace checkpoint name for `AutoModel.from_pretrained`.
        num_tasks (int): Number of regression targets.
        mlp_hidden_dim (int | None): Hidden dim for the projection MLP.
        dropout (float): Dropout applied in the projection head.
    """
    def __init__(self, model_name, num_tasks, mlp_hidden_dim=None, dropout=0.0):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size

        # One linear output per task, standard linear regressor if mlp_hidden_size=0
        self.regressor = ProjectionHead(
            input_dim=hidden_size,
            output_dim=num_tasks,
            hidden_dim=mlp_hidden_dim,
            dropout=dropout,
        )

        self.loss_fct = nn.MSELoss(reduction='none')

    def forward(self, input_ids, attention_mask, labels=None, sample_weights=None):
        # Backbone
        outputs = self.backbone(input_ids, attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        
        # Heads
        # Shape: [Batch, Num_Tasks]
        logits = self.regressor(cls_emb)
        
        # Compute loss
        loss = None
        if labels is not None:
            # Weighted Loss
            # Raw MSE per element: [Batch, Num_Tasks]
            raw_loss = self.loss_fct(logits, labels)
            
            # Apply weights: [Batch, Num_Tasks] * [Batch, Num_Tasks]
            if sample_weights is not None:
                weighted_loss = raw_loss * sample_weights
            else:
                weighted_loss = raw_loss
                
            # Average over batch AND tasks
            loss = weighted_loss.mean()

        return (loss, logits) if loss is not None else logits
    

class EsmMutNHeads(nn.Module):
    """ESM model predicting task outputs from averaged mutant-position embeddings.

    Instead of using the CLS token, this model averages embeddings at positions
    indicated by `mutation_mask` to produce a mutant-specific embedding which
    is passed through a `ProjectionHead` to predict `num_tasks` outputs.

    Args:
        model_name (str): HuggingFace checkpoint name for `AutoModel.from_pretrained`.
        num_tasks (int): Number of regression targets.
        mlp_hidden_dim (int | None): Hidden dim for the projection MLP.
        dropout (float): Dropout applied in the projection head.
    """
    def __init__(self, model_name, num_tasks, mlp_hidden_dim=None, dropout=0.0):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size

        # One linear output per task, standard linear regressor if mlp_hidden_size=0
        self.regressor = ProjectionHead(
            input_dim=hidden_size,
            output_dim=num_tasks,
            hidden_dim=mlp_hidden_dim,
            dropout=dropout,
        )
        self.loss_fct = nn.MSELoss(reduction='none')

    def forward(self, input_ids, attention_mask, mutation_mask, labels=None, sample_weights=None):
        # Backbone
        outputs = self.backbone(input_ids, attention_mask)
        # Shape: [Batch, Seq_Len, Hidden]
        last_hidden_state = outputs.last_hidden_state
        
        # Expand mask to match hidden dimension
        # Mask Shape: [Batch, Seq_Len] -> [Batch, Seq_Len, 1]
        expanded_mask = mutation_mask.unsqueeze(-1).float()
        
        # Zero out non-mutant positions
        # Result: [Batch, Seq_Len, Hidden]
        masked_embeddings = last_hidden_state * expanded_mask
        
        # Sum embeddings along the sequence dimension
        # Result: [Batch, Hidden]
        sum_embeddings = masked_embeddings.sum(dim=1)
        
        # Count number of mutations per sample to calculate Mean
        # Result: [Batch, 1]
        # (clamp min=1 to avoid divide-by-zero if a WT sequence is passed)
        mutation_counts = expanded_mask.sum(dim=1).clamp(min=1e-9)
        
        # Mean Pool
        mutant_emb = sum_embeddings / mutation_counts
        
        # Heads
        logits = self.regressor(mutant_emb)
        
        # Compute weighted loss
        loss = None
        if labels is not None:
            raw_loss = self.loss_fct(logits, labels)
            
            if sample_weights is not None:
                weighted_loss = raw_loss * sample_weights
            else:
                weighted_loss = raw_loss
                
            loss = weighted_loss.mean()

        return (loss, logits) if loss is not None else logits
    

class ddGDecoderHillLangmuir(nn.Module):
    """Physics-based decoder using a Hill-Langmuir binding model.

    Maps predicted changes in folding energy (`ddG_fold`) and ligand binding
    energies (`ddG_ligands_map`) to per-sort enrichment scores. Models include
    folding probability, ligand binding with Hill cooperativity, and optional
    historical sequencing of sorts.

    Args:
        sort_configs (list): List of dicts defining experimental conditions. Each
            dict should provide `ligands` (concentrations in M) and `masks` (0/1
            indicating if a ligand was present for that sort). Optionally, a
            `history_indices` list can specify previous sorts contributing to a
            cumulative trajectory.
        ligand_names (list): Names of ligands in consistent order (e.g. ['C','F']).
        use_histories (bool): If True, sum log-probabilities according to provided trajectories.
        wt_dGs (dict | None): Initial WT binding energies per ligand (kcal/mol).
        wt_dG_fold (float): Initial WT folding energy (kcal/mol).
        learn_anchor (bool): If True, WT energies are trainable.
        learn_physics (bool): If True, additional physics params (Hill, conc. corr.) are learned.

    Class Attributes:
        R, T, RT: Physical constants used to convert between dG and equilibrium constants.
    """
    # Physical Constants (Standard State)
    R = 0.001987  # Gas constant (kcal / K / mol)
    T = 298.15    # Temperature (Kelvin) -> 25°C
    RT = R * T    # ~0.592 kcal/mol

    def __init__(self, sort_configs, ligand_names, use_histories=False, wt_dGs=None, wt_dG_fold=-4.0, learn_anchor=True, learn_physics=True):
        super().__init__()
        self.ligand_names = ligand_names
        self.num_ligands = len(ligand_names)
        self.num_sorts = len(sort_configs)
        self.use_histories = use_histories
        self.learn_physics = learn_physics

        # define learnable physics parameters
        if self.learn_physics:
            # Hill Coefficient. Initializing at 1.0, which is standard Michaelis-Menten.
            # To constrain as positive and prevent from getting too low, we use the softplus
            # function (ln(1+e^x) in the forward pass. That means to initialize at 1.0 here,
            # we need to have the tensor be x, where softplus(x) + 0.5 ~= 1.0. x here = 0.54
            self.n_hill_param = nn.Parameter(torch.tensor(0.54))

            # Concentration correction per sort. Allows the model to adjust the effective concentration
            # to account for depletion / pipetting errors. Starting at zero, as tanh(0) = 0.
            # To constrain, we'll use tanh (a sigmoid curve) to bound it between sane values.
            # We'll use -2, 2 as bounds, which end up being 0.1 to 7 times the conc. after the log correction.
            self.conc_param = nn.Parameter(torch.zeros(self.num_sorts))

        else:
            self.register_buffer('n_hill_fixed', torch.tensor(1.0))
            self.register_buffer('conc_fixed', torch.zeros(self.num_sorts))

        # define WT energies, configure to be trainable or not
        # nn.ParameterDict() will hold the value for each ligand
        self.wt_dG_ligands = nn.ParameterDict()

        # WT folding energy
        self.wt_dG_fold = nn.Parameter(torch.tensor(float(wt_dG_fold)))
        # gradient enabled by default, turn it off if requested
        if not learn_anchor: self.wt_dG_fold.requires_grad = False

        # Binding energies. dG = RT * ln(Kd)
        if wt_dGs is None: wt_dGs = {}
        for name in ligand_names:
            # retrieve from dict, default to 1 uM Kd
            val = wt_dGs.get(name, self.RT * math.log(1e-6))
            param = nn.Parameter(torch.tensor(float(val)))
            if not learn_anchor: param.requires_grad = False
            self.wt_dG_ligands[name] = param

        # Define experiment condition matrices
        # Shapes: (n_sorts, n_ligands)
        L_matrix = [] # conc, in M
        M_matrix = [] # selection masks

        # Trajectory Matrix: (n_sorts, n_sorts)
        # Defines the sequential history of selection
        trajectory_matrix = torch.zeros(self.num_sorts, self.num_sorts)

        for i, config in enumerate(sort_configs):
            # Extract concentration (add 1e-15 to avoid log(0))
            row_L = [config['ligands'].get(name, 0.0) + 1e-15 for name in ligand_names]
            row_M = [config['masks'].get(name, 0.0) for name in ligand_names]
            L_matrix.append(row_L)
            M_matrix.append(row_M)

            # Extract history from config (defaults to self if missing)
            indices = config.get('history_indices', [i])
            for past_idx in indices:
                trajectory_matrix[i, past_idx] = 1.0

        # Register experimental conditions as buffers, which are non-trainable
        # constants that need to follow along with the model on different devices
        self.register_buffer('L_tensor', torch.tensor(L_matrix, dtype=torch.float32))
        self.register_buffer('M_tensor', torch.tensor(M_matrix, dtype=torch.float32))
        self.register_buffer('traj_tensor', trajectory_matrix)

        # Define learnable linear parameters to map dG to enrichment
        self.scale = nn.Parameter(torch.ones(self.num_sorts))
        self.bias = nn.Parameter(torch.zeros(self.num_sorts))

    def forward(self, ddG_fold, ddG_ligands_map):
        """
        Args:
            ddG_fold: Predicted stability change (kcal/mol). Shape: (Batch, 1)
            ddG_ligands_map: Dict of tensors {name: (Batch, 1)} in kcal/mol.
        """
        # Preprocess learnable physics parameters
        if self.learn_physics:
            # Hill Coefficient must be positive and > 0.5 (which would be strong negative cooperativity)
            n_hill = F.softplus(self.n_hill_param) + 0.5

            # Concentration correction, between ~0.13x to 7.4x
            log_conc_correction = 2.0 * torch.tanh(self.conc_param)
        else:
            n_hill = self.n_hill_fixed
            log_conc_correction = self.conc_fixed

        # Calculate folding probability
        # dG_total = WT_dG + ddG
        dG_fold_total = self.wt_dG_fold + ddG_fold
        # Restrict dG_fold to reasonable range to prevent gradient explosion
        dG_fold_total = torch.clamp(dG_fold_total, min=-20.0, max=20.0)

        # Convert to eq. constant, K: K = e^(-dG / RT)
        K_fold = torch.exp(-dG_fold_total / self.RT)
        p_folded = K_fold / (1.0 + K_fold)

        # Calculate binding probabilities
        # Vectorize WT dG list
        wt_dG_stack = torch.stack([self.wt_dG_ligands[name] for name in self.ligand_names])
        
        # Stack predicted ddG values for vector operations
        ddG_stack = torch.cat([ddG_ligands_map[name] for name in self.ligand_names], dim=1)

        # Total binding energy (kcal/mol)
        dG_bind_total = wt_dG_stack.unsqueeze(0) + ddG_stack

        # Restrict binding energy to reasonable range to prevent gradient explosion
        dG_bind_total = torch.clamp(dG_bind_total, min=-20.0, max=20.0)

        # Convert to Kd (M). Kd = e^(dG/RT)
        Kd_val = torch.exp(dG_bind_total / self.RT)

        # Apply Hill and ligand corrections
        Kd_effective = torch.pow(Kd_val, n_hill) # type: ignore
        conc_mult = torch.exp(log_conc_correction).view(1, -1, 1) # type: ignore
        L_corrected = self.L_tensor.unsqueeze(0) * conc_mult # type: ignore
        L_effective = torch.pow(L_corrected, n_hill) # type: ignore


        # Calculate p_bound using Hill-Langmuir equation
        # Kd: (batch_size, n_ligands) -> (batch_size, 1, n_ligands)
        Kd_broad = Kd_effective.unsqueeze(1)

        # Hill-Langmuir, using broadcasting for efficient computation
        # (1, n_sorts, n_ligands) / (batch_size, 1, n_ligands) = (batch_size, n_sorts, n_ligands)
        # essentially, calculate p_bound for each ligand, for each sort, for each variant in the batch
        p_bound = L_effective / (L_effective + Kd_broad + 1e-15)

        # Combine and apply masking logic
        # p_effective = 0 if mask is 0 (ligand not yet seen), else = p_bound
        p_effective = p_bound * self.M_tensor.unsqueeze(0) + (1.0 - self.M_tensor.unsqueeze(0)) # type: ignore

        # Calculate probability of having preconditions for selection on binding
        # For example, in sort A2, p of making it = p_bind_C * p_bind_F
        p_combined_binding = torch.prod(p_effective, dim=2)

        # Calculate total probability of having preconditions for selection
        p_observable = p_folded * p_combined_binding

        # Calculate probability of independent sorts
        log_prob_independent = torch.log(p_observable + 1e-15)

        # Branch logic for modeling sorting history
        if self.use_histories:
        # Sum log-probs according to history
            # (Batch, Sorts) x (Sorts, Sorts)^T -> (Batch, Sorts)
            # For A4, this sums log_prob(A1) + log_prob(A2) + log_prob(A3) + log_prob(A4)
            log_prob_cumulative = torch.matmul(log_prob_independent, self.traj_tensor.t()) # type: ignore
            return self.scale * log_prob_cumulative + self.bias
        
        else:
            return self.scale * log_prob_independent + self.bias
        

class ddGDecoderLinear(nn.Module):
    """Simple linear decoder mapping ddG features to per-sort scores.

    Implements a masked linear model where the input features are the
    predicted folding ddG and ligand ddGs. The connectivity between inputs
    and sorts is defined by `sort_configs` and enforced via `weight_mask`.

    Args:
        sort_configs (list): List of per-sort configuration dicts containing at least a `masks` key.
        ligand_names (list): Ordered ligand names matching columns in configs.

    Returns:
        A module that maps `(ddG_fold, ddG_ligands_map)` to per-sort scores.
    """
    def __init__(self, sort_configs, ligand_names):
        super().__init__()
        self.ligand_names = ligand_names
        self.num_ligands = len(ligand_names)
        self.num_sorts = len(sort_configs)
        
        # Total inputs = 1 (Folding) + N (Ligands)
        self.num_inputs = 1 + self.num_ligands
        
        # Define Learnable Parameters
        # Weight Matrix shape: (num_sorts, num_inputs)
        # We initialize to -1.0 because positive ddG usually reduces enrichment.
        self.weights = nn.Parameter(torch.full((self.num_sorts, self.num_inputs), -1.0))
        
        # Bias shape: (num_sorts)
        # Initializes to 0.0 (Centered data assumption)
        self.bias = nn.Parameter(torch.zeros(self.num_sorts))
        
        # Build the Connectivity Mask from Config
        # We want to force weights to 0 if the physics says they shouldn't exist.
        # Column 0 is always "Folding" (Input index 0)
        # Columns 1..N are "Binding" (Input indices 1..N)
        mask_matrix = torch.zeros(self.num_sorts, self.num_inputs)
        
        for i, config in enumerate(sort_configs):
            # Folding Term: Always active for every sort (assuming expression matters)
            mask_matrix[i, 0] = 1.0
            
            # Binding Terms: Active only if ligand is in the 'masks' dict
            for j, lig_name in enumerate(ligand_names):
                # 'masks' dict keys are ligand names, values are 1/0
                if config['masks'].get(lig_name, 0) > 0:
                    # Column index = j + 1 (because 0 is fold)
                    mask_matrix[i, j + 1] = 1.0
                    
        # Register as buffer so it saves with state_dict but isn't updated by optimizer
        self.register_buffer('weight_mask', mask_matrix)

    def forward(self, ddG_fold, ddG_ligands_map):
        """
        Args:
            ddG_fold: (Batch, 1)
            ddG_ligands_map: Dict {name: (Batch, 1)}
        Returns:
            scores: (Batch, num_sorts)
        """
        # Stack Inputs into a single Feature Matrix X
        # Shape: (Batch, 1 + num_ligands)
        # Order must match the mask construction: [Fold, Ligand1, Ligand2...]
        ligand_cols = [ddG_ligands_map[name] for name in self.ligand_names]
        X = torch.cat([ddG_fold] + ligand_cols, dim=1)
        
        # Apply Masked Linear Transformation
        # W_effective = W * Mask
        # This ensures gradients for unconnected terms are zeroed out
        effective_weights = self.weights * self.weight_mask # type: ignore
        
        # y = X @ W.T + b
        # (Batch, In) @ (Sorts, In).T -> (Batch, Sorts)
        scores = torch.matmul(X, effective_weights.t()) + self.bias
        
        return scores
    

class PhysicsESM(nn.Module):
    """End-to-end model combining a pretrained ESM backbone with a physics-aware decoder.

    Constructs an ESM feature extractor (`AutoModel.from_pretrained`) -> a
    `ProjectionHead` that regresses from sequence embeddings to latent physics
    variables (stability `ddG_fold` and per-ligand `ddG`s) -> a decoder that
    maps those latent physics to per-sort enrichment scores.

    Supported decoders:
        - 'hill_langmuir': physics-based Hill-Langmuir binding + folding model
          (uses `ddGDecoderHillLangmuir`).
        - 'linear': masked linear mapping from ddG features to scores
          (uses `ddGDecoderLinear`).

    Embedding modes:
        - 'cls' (default): use the CLS token embedding (position 0).
        - 'mutant_mean': mean over positions indicated by `mutation_mask`.

    Args:
        model_checkpoint (str): HF checkpoint for `AutoModel.from_pretrained`.
        sort_configs (list): Per-sort config dicts with `ligands` (conc. in M),
            `masks` (0/1 presence) and optional `history_indices`.
        ligand_names (list): Ordered ligand names matching `sort_configs` columns.
        decoder_type (str): One of {'hill_langmuir', 'linear'}.
        embedding_mode (str): 'cls' or 'mutant_mean'.
        wt_dGs (dict|None): Optional initial WT binding energies (kcal/mol).
        wt_dG_fold (float): Initial WT folding energy (kcal/mol).
        frozen_backbone (bool): Freeze ESM backbone parameters if True.
        learn_anchor (bool): Make WT energies trainable if True.
        learn_physics (bool): Learn additional physics params (Hill, conc. corr.) if True.
        ddG_std (float): Init stddev for final regression layer.
        mlp_hidden_dim (int|None): Hidden dim for `ProjectionHead` MLP (None => single linear).
        dropout (float): Dropout for projection MLP.

    Forward signature:
        forward(input_ids, attention_mask, mutation_mask=None, labels=None, sample_weights=None)

    Returns:
        If `labels` is None:
            (predictions, physics_dict)
        If `labels` is provided:
            (loss, predictions, physics_dict)

        - `predictions`: Tensor (batch_size, num_sorts) of per-sort scores.
        - `physics_dict`: dict with keys `'ddG_fold'` and each ligand name, values shaped (batch_size, 1).
        - `loss`: MSE loss (scalar) when labels are provided (supports optional `sample_weights`).

    Example:
        model = PhysicsESM('esm-model', sort_configs, ['C','F'], decoder_type='hill_langmuir')
        preds, phys = model(input_ids, attention_mask, mutation_mask)
    """
    def __init__(
        self, 
        model_checkpoint, 
        sort_configs, 
        ligand_names, 
        decoder_type,
        embedding_mode='cls', 
        wt_dGs=None, 
        wt_dG_fold=-4.0, 
        frozen_backbone=False, 
        learn_anchor=True, 
        learn_physics=True, 
        ddG_std=0.02,
        mlp_hidden_dim=None,
        dropout=0.0
    ):
        super().__init__()
        self.ligand_names = ligand_names
        self.decoder_type = decoder_type
        self.embedding_mode = embedding_mode

        # Initialize backbone
        self.esm = AutoModel.from_pretrained(model_checkpoint)
        if frozen_backbone:
            for param in self.esm.parameters():
                param.requires_grad = False

        # Initialize physics regressor (ESM -> ddG)
        hidden_size = self.esm.config.hidden_size
        output_dim = 1 + len(ligand_names) # Stability + N_ligands

        self.physics_regressor = ProjectionHead(
            input_dim=hidden_size, 
            output_dim=output_dim, 
            hidden_dim=mlp_hidden_dim, 
            dropout=dropout
        )

        # Custom Initialization for the final layer of the head
        # We want to start near 0.0 (WT behavior).
        # We access the *last* linear layer in the sequential block or the single linear layer.
        if isinstance(self.physics_regressor.net, nn.Sequential):
            final_linear = self.physics_regressor.net[-1]
        else:
            final_linear = self.physics_regressor.net
            
        nn.init.normal_(final_linear.weight, mean=0.0, std=ddG_std) # type: ignore
        nn.init.zeros_(final_linear.bias) # type: ignore

        # Initialize Decoder based on type
        if decoder_type == 'hill_langmuir':
            self.decoder = ddGDecoderHillLangmuir(
                sort_configs, 
                ligand_names, 
                wt_dGs=wt_dGs, 
                wt_dG_fold=wt_dG_fold, 
                learn_anchor=learn_anchor, 
                learn_physics=learn_physics
            )
        elif decoder_type == 'linear':
            self.decoder = ddGDecoderLinear(sort_configs, ligand_names)
        else:
            raise ValueError(f"Unknown decoder_type: {decoder_type}")

    def get_embedding(self, last_hidden_state, mutation_mask=None):
        """
        Helper to extract the sequence embedding based on the selected mode.
        """
        if self.embedding_mode == 'cls':
            # Use the CLS token (first position)
            # Shape: [Batch, Hidden]
            return last_hidden_state[:, 0, :]
        
        elif self.embedding_mode == 'mutant_mean':
            if mutation_mask is None:
                raise ValueError("embedding_mode='mutant_mean' requires 'mutation_mask' in forward()")
            
            # Expand mask: [Batch, Seq_Len] -> [Batch, Seq_Len, 1]
            expanded_mask = mutation_mask.unsqueeze(-1).float()
            
            # Zero out WT positions: [Batch, Seq_Len, Hidden]
            masked_embeddings = last_hidden_state * expanded_mask
            
            # Sum and Divide: [Batch, Hidden]
            sum_embeddings = masked_embeddings.sum(dim=1)
            # clamp min=1 to avoid div-by-zero (if passed a pure WT sequence with 0 mutations)
            mutation_counts = expanded_mask.sum(dim=1).clamp(min=1e-9)
            
            return sum_embeddings / mutation_counts
        
        else:
            raise ValueError(f"Unknown embedding_mode: {self.embedding_mode}")

    def forward(self, input_ids, attention_mask, mutation_mask=None, labels=None, sample_weights=None):
        """
        Args:
            mutation_mask: Boolean/Float mask of shape (Batch, Seq_Len). 
                           1 = Mutant Position, 0 = WT Position.
                           Required only if embedding_mode='mutant_mean'.
        """
        # Run Backbone
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        # Extract Embedding (cls or mutant)
        embedding = self.get_embedding(last_hidden_state, mutation_mask)

        # Predict latent physics variables
        physics = self.physics_regressor(embedding)

        # Unpack ddG
        ddG_fold = physics[:, 0:1]
        ddG_ligands_map = {}
        for i, name in enumerate(self.ligand_names):
            ddG_ligands_map[name] = physics[:, i+1 : i+2]

        # Run through selected decoder
        predictions = self.decoder(ddG_fold, ddG_ligands_map)

        # Calculate loss
        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss(reduction='none')
            loss_unreduced = loss_fct(predictions, labels)

            if sample_weights is not None:
                loss_unreduced = loss_unreduced * sample_weights

            loss = loss_unreduced.mean()

        # Store physics values
        physics_dict = {'ddG_fold': ddG_fold}
        physics_dict.update(ddG_ligands_map)

        if loss is not None:
            return (loss, predictions, physics_dict)
        else:
            return (predictions, physics_dict)
        

