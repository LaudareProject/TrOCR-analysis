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

## Output Files

- `ml_experiment_threshold_analysis.png` - ML experiment curves
- `gradcam_ablation_threshold_analysis.png` - GradCAM ablation curves  
- `filtered_entropy_loss_threshold_analysis.png` - Filtered experiment curves
- `cer_histogram_token_level.png` / `cer_histogram_sample_level.png` - CER distributions