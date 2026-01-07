from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from torch.optim import AdamW

from .training_metrics import MultiTaskMetrics


class DifferentialLearningRateTrainer(Trainer):
    def __init__(self, *args, head_learning_rate=1e-3, use_differential_lr=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.head_learning_rate = head_learning_rate
        self.use_differential_lr = use_differential_lr

    def create_optimizer(self):
        # If the flag is OFF, fall back to the standard Hugging Face behavior
        if not self.use_differential_lr:
            return super().create_optimizer()

        # If ON, create separate groups
        print(f"Using Differential Learning Rates: Backbone={self.args.learning_rate}, Head={self.head_learning_rate}")
        
        # Base LR from TrainingArguments (usually small, e.g. 1e-5)
        backbone_lr = self.args.learning_rate
        
        # Separate Parameters
        backbone_params = []
        head_params = []
        
        for name, param in self.model.named_parameters(): # type: ignore
            if not param.requires_grad:
                continue
            
            # Logic: If it contains "esm" (or "bert", etc), it's backbone.
            # Everything else (decoder, projector, regression heads) is head.
            if "esm" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
        
        # Create Parameter Groups
        optimizer_grouped_parameters = [
            {
                "params": backbone_params,
                "lr": backbone_lr,
                "weight_decay": self.args.weight_decay
            },
            {
                "params": head_params,
                "lr": self.head_learning_rate,
                "weight_decay": self.args.weight_decay 
            }
        ]
        
        # Initialize Optimizer
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
        )
        
        return self.optimizer

def initialize_trainer(model, model_output_dir, target_config, train_ds, val_ds, model_hyperparameters: dict):
    compute_metrics_fn = MultiTaskMetrics(target_config)
    
    args = TrainingArguments(
        output_dir=model_output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=model_hyperparameters['learning_rate'],
        per_device_train_batch_size=model_hyperparameters['per_device_train_batch_size'],
        per_device_eval_batch_size=model_hyperparameters['per_device_eval_batch_size'],
        num_train_epochs=model_hyperparameters['epochs'],
        weight_decay=model_hyperparameters['weight_decay'],
        max_grad_norm=model_hyperparameters["max_grad_norm"],
        load_best_model_at_end=True,
        metric_for_best_model=model_hyperparameters["metric_for_best_model"],
        greater_is_better=model_hyperparameters["greater_is_better"],
        include_for_metrics = target_config,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none"
    )

    trainer = DifferentialLearningRateTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=model_hyperparameters['es_patience'])],
        use_differential_lr=model_hyperparameters['use_differential_lr'],
        head_learning_rate=model_hyperparameters['head_learning_rate']
    )
    return trainer