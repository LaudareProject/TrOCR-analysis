# Analysis Experiments

## Setup

```bash
uv sync
```

## Run Experiments

### Token-level threshold analysis (main experiments)
```bash
uv run python threshold_analysis.py
```
Generates balanced accuracy curves across CER thresholds for ML experiments, GradCAM ablation, and filtered experiments (excluding token_id 0,2). Outputs 3 plots.

### Individual ML experiment with custom threshold
```bash
uv run python ml_experiment.py 0.05
```
Runs ML classification experiments at specified CER threshold.

### Individual GradCAM ablation with custom threshold  
```bash
uv run python gradcam_ablation.py 0.05
```
Runs GradCAM feature ablation experiments at specified CER threshold.

### CER distribution analysis
```bash
uv run python cer_histogram.py
```
Generates token-level and sample-level CER histograms.

### TrOCR Ablation Experiments (Cortonese dataset)
```bash
uv run python trocr_ablation.py
```
Tests different training configurations on the Cortonese dataset using Microsoft TrOCR:

-Full fine-tuning (all layers)

-Baseline (first 10 layers frozen)

-No CLAHE

-No augmentation

-Full freezing

Outputs performance metrics (CER, WER) and comparison plots across configurations.

### Cross-Validation on Public READ-16 Dataset
```bash
uv run python read16_crossval.py
```
Performs cross-validation of the best full fine-tuned Cortonese model on READ-16. Outputs CER/WER metrics.

### Token-Level GradCAM & Attention Analysis (Cortonese dataset)
```bash
uv run python token_gradcam_analysis.py
```
Analyzes token-level attention, GradCAM feature importance, and additional metrics (e.g., Gini coefficient) for the best full fine-tuned model. Outputs GradCAM heatmaps, attention distributions, and token-level performance visualizations.

## Output Files

- `ml_experiment_threshold_analysis.png` - ML experiment curves
- `gradcam_ablation_threshold_analysis.png` - GradCAM ablation curves  
- `filtered_entropy_loss_threshold_analysis.png` - Filtered experiment curves
- `cer_histogram_token_level.png` / `cer_histogram_sample_level.png` - CER distributions
- 'trocr_ablation_metrics.json' - Metrics for each TrOCR training configuration
- 'trocr_ablation_comparison.png' - Plot comparing CER/WER across configurations
- 'read16_crossval_metrics.json' - Cross-validation metrics for READ-16 dataset
- 'token_gradcam_heatmaps/' - GradCAM visualizations per token
- 'token_attention_metrics.json' - Token-level attention and Gini metrics

