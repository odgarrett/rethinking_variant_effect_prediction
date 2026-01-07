# rethinking_variant_effect_prediction

Experiments with variant effect prediction in Pik-AVR-Pik DMS data. Refactored version of local "rethinking_de_sequence_analysis" with proper version control and better use of the dms_variants package for analysis.

## Data Processing Pipeline

1. FASTQ data received from Amplicon-EZ.
2. Reads merged using vsearch.
3. Merged reads aligned to wild-type using alignparse package.
4. Sorts are treated as samples from the same library, and input into dms_variants CodonVariantTable.
5. DNA-sequence-wise and AA-sequence-wise scores (enrichment, physics, etc.) are calculated.
6. DNA-level scores can be used to calculate overdispersion for error estimates.
7. Data is split 80-10-10 train-val-test.
8. Modeling dataset made using Dataset class from datasets package with label counts or scores.

## Modeling Approaches

### Enrichment score prediction

Assumes enrichment (functional) scores are directly proportional to phenotype of interest.

- Key variables
    - read count threshold
    - pre and post selection libraries (e.g. A1/lib, A3/lib, A1/A3)
    - inverse-variance weighting
    - model size

### Multi-enrichment score prediction

Same as "enrichment score prediction" but uses multihead prediction to regularlize the model in the hopes of increased generalizability. Combining output enrichment scores in different ways (e.g. averaging, weighted averaging, fitting to physical model) can be tested for better correlation with ground-truth phenotype (e.g. HR response, in vitro binding affinity)

- Key variables
    - read count threshold
    - inverse-variance weighting
    - model size
    - predicted enrichment score processing

### Linear factor modeling

Assumes enrichment scores are a linear combination of latent phenotypes (such as $\Delta \Delta G_{fold}$, $\Delta \Delta G_{bind,C}$, and $\Delta \Delta G_{bind,F}$). Two variants of this model:
1. **Two-step**: Solve linear factor model to obtain these latent phenotype values for each variant, then fit sequence-to-phenotype model to these values.
2. **One-step**: Use linear factor model as a biophysical decoder module on top of sequence-to-phenotype model.

- Key variables
    - read count threshold
    - inverse-variance weighting
    - weights for sort contributions to latent phenotypes
    - model size
    - one-step vs two-step

### Global epistasis modeling

Assumes enrichment scores are the output of a non-linear sampling function of an additive latent phenotype. For the purposes of single cell assays of binding affinity, the non-linear function is essentially the Hill-Langmuir equation, and $\Delta \Delta G$ of folding and binding are the additive phenotypes.

- Key variables
    - read count threshold
    - inverse-variance weighting
    - wild-type $\Delta \Delta G$ guesses (and whether they are trainable)
    - trainability of Hill coefficient and effective concentrations
    - one-step vs two-step

### Likelihood modeling

Assumes raw read counts are stochastic samples drawn from a probability distribution (Negative Binomial) defined by the underlying biophysics and experimental noise. Explicitly models the generative process of the experiment to maximize the probability of the observed data given the sequence.
- Key variables
    - overdispersion parameter ($\alpha$) estimation (e.g. from synonymous variants)
    - generative distribution (Poisson vs. Negative Binomial)
    - wild-type $\Delta \Delta G$ guesses
    - trainability of Hill coefficient and effective concentrations