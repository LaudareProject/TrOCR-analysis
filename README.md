# TrOCR Analysis Experiments

Reusable scripts analyze Microsoft TrOCR fine-tuning, GradCAM/attention attributions, and READ16/Hüttner hyper-parameter sweeps for historical handwriting datasets. All analyses rely on pre-split COCO-style JSON annotations plus their image folders; download and unpack the datasets before running each script.

## Prerequisites

- Sync the workspace with UV and ensure Python 3.11+ is active: `uv sync`
- Install dependencies from `pyproject.toml`/`uv.lock` via `uv sync` before invoking the analysis scripts.
- Mount or copy the Cortona/OCR image archive and the split annotations (`ocr_train.json`, `ocr_val.json`, `ocr_test.json`); the Colab-derived scripts expect `/content/drive/MyDrive` paths, so adjust the constants inside the scripts if you reproduce locally.

## 1. TrOCR Ablation Study (Cortona dataset)

`ablation_study_experiments.py` replays the TrOCR ablations described in the project: full fine-tuning, freezing variants, CLAHE removal, and augmentation removal. It is designed to run inside Colab with `/content/drive/MyDrive/OMR/images/Dataset cortona` and the JSON splits, but you can rerun it locally by updating the path constants near the top of the file.

1. Place `ocr_train.json`, `ocr_val.json`, `ocr_test.json`, and the Cortona images in a drive-like directory and update `IMG_DIR`, `TRAIN_JSON`, `VAL_JSON`, and `TEST_JSON` accordingly.
2. Set the `EXPERIMENT` variable to one of `baseline`, `no_clahe`, `no_aug`, `freeze_none`, `freeze_all`. Each option sets a tailored learning rate, batch size, gradient accumulation, and whether CLAHE/augmentation is enabled.
3. Run the script: `uv run python ablation_study_experiments.py`. It fine-tunes `microsoft/trocr-base-handwritten`, evaluates CER/WER on `ocr_test.json`, and emits:
   - `trocr_ablation_results/{experiment}_{timestamp}/results.json` (experiment metrics and config)
   - Model checkpoints + processor files for the final epoch inside the same directory
   - Attention visualizations exported next to the metrics for qualitative comparison
4. Archive the resulting `results.json` and plots to reproduce the comparison table between configurations.

## 2. GradCAM and Attention Analysis (Cortona dataset)

`combined_token_level_analysis_for_grad_cam_and_attention.py` loads a fine-tuned TrOCR checkpoint, rebuilds the Cortona test split, and generates token-wise GradCAM vs. encoder-decoder attention metrics.

1. Update the hard-coded paths used by the script: set `IMG_DIR`, `TRAIN_JSON`, `VAL_JSON`, and `TEST_JSON` to your dataset location; set `model_path` to your fine-tuned checkpoint (default `/content/drive/MyDrive/freezenoneupdatedepochpatience_results`); and set `output_dir` to your export folder (default `/content/combined_gradcam_attention_analysis`).
2. Keep the JSON annotations and image root aligned so the `TrOCRDataset` constructor can resolve each image file.
3. Adjust `num_samples` within the script to limit how many examples produce paired visualizations.
4. Run `uv run python combined_token_level_analysis_for_grad_cam_and_attention.py`. Outputs:
   - `combined_token_results.csv` — token-level GradCAM mass, attention mass, Gini, entropy, and loss per token
   - `combined_image_results.csv` — per-sample summaries of attribution strength and CER/WER
   - Visual overlays (GradCAM vs attention) under the configured `output_dir`
5. Use the CSVs and overlays to reproduce plots that compare GradCAM importance with attention distribution and token-level errors.

## 3. READ16 / Hüttner Experiments

`hyperparams_crossval.py` runs the READ16 cross-validation and I-Ct_91 experiments that mimic the Hüttner et al. (2025) training policy using one-cycle schedules and optional custom augmentation.

1. Download the READ16 dataset from [Zenodo DOI 10.5281/zenodo.1297399](https://doi.org/10.5281/zenodo.1297399) and unzip it so `READ16-data/` contains the expected CSV/JSON splits. Keep `READ16.zip` if you prefer to mirror the original directory structure and update the script paths accordingly.
2. Prepare the I-Ct_91 data by placing its directory (`I-Ct_91/`) alongside the repository; it should contain the `train_test_*` JSONs used by the script.
3. Run the experiments with the desired flags. Minimal CLI usage looks like:
   ```bash
   uv run python hyperparams_crossval.py --dataset read16 --custom-aug
   uv run python hyperparams_crossval.py --dataset I-Ct_91 --huttner --pretrain
   ```
   Key flags:
   - `--dataset {read16,I-Ct_91}` (required)
   - `--huttner` to activate the one-cycle Hüttner learning rate policy
   - `--custom-aug` to enable the CLAHE/augmentation stack described in the paper
   - `--pretrain` to warm-start from a CATMuS-trained checkpoint before fine-tuning
   - `--early-stop`, `--huttner_max_lr`, `--huttner_pct_start` to tweak the scheduler
4. The script saves CER/WER metrics plus config details under `bozen_training_output/` or `ict91_training_output/` (depending on the dataset) and exports a `results.json` with summary statistics; checkpoints and processor files are kept in the same folder.
5. Reproduced results reported in this repository are:

| Dataset | Pretrained on CATMuS | Augmentation policy | Hyper-parameters | Command flags | Test CER | Test WER |
| --- | --- | --- | --- | --- | ---: | ---: |
| `I-Ct_91` | Yes | Huttner re-implemented (`HuttnerSupplementAugmentation`, no CLAHE) | Huttner one-cycle | `--huttner --dataset I-Ct_91 --pretrain` | 6.85% | 27.46% |
| `read16` | No | Custom (`ManuscriptAugmentation` + CLAHE) | Huttner one-cycle | `--huttner --custom-aug --dataset read16` | 5.16% | 21.18% |
| `I-Ct_91` | Yes | Custom (`ManuscriptAugmentation` + CLAHE) | Huttner one-cycle | `--huttner --custom-aug --dataset I-Ct_91 --pretrain` | 7.09% | 28.51% |
| `I-Ct_91` | No | Custom (`ManuscriptAugmentation` + CLAHE) | Huttner one-cycle | `--huttner --custom-aug --dataset I-Ct_91` | 7.81% | 30.40% |
| `read16` | No | Custom (`ManuscriptAugmentation` + CLAHE) | Standard (non-Huttner) | `--custom-aug --dataset read16` | 7.06% | 25.69% |
| `I-Ct_91` | No | Custom (`ManuscriptAugmentation` + CLAHE) | Standard (non-Huttner) | `--custom-aug --dataset I-Ct_91` | 9.55% | 34.72% |

Reference baseline: Kraken CATMuS is reported at 7.89% CER and 31.17% WER.

## Script Catalog (all repository scripts)

### Training and attribution pipelines

- `ablation_study_experiments.py`
  - Purpose: runs TrOCR ablations on Cortona pre-split JSON data (`baseline`, `no_clahe`, `no_aug`, `freeze_none`, `freeze_all`).
  - Run: Colab-style execution (script includes `google.colab` setup and `/content` paths); set `EXPERIMENT`, then run `uv run python ablation_study_experiments.py`.
  - Outputs: `trocr_ablation_results/<exp_timestamp>/config.json`, `results.json`, checkpoints, final model, attention maps.
- `combined_token_level_analysis_for_grad_cam_and_attention.py`
  - Purpose: computes token-level GradCAM and cross-attention metrics for a trained TrOCR checkpoint.
  - Run: Colab-style execution; set model/data paths in constants, then run `uv run python combined_token_level_analysis_for_grad_cam_and_attention.py`.
  - Outputs: `combined_token_results.csv`, `combined_image_results.csv`, per-sample visualizations, plots in `combined_gradcam_attention_analysis/`.
- `hyperparams_crossval.py`
  - Purpose: fine-tunes/evaluates TrOCR on `read16` or `I-Ct_91` with Huttner/custom settings and optional CATMuS pretraining.
  - Run: `uv run python hyperparams_crossval.py --dataset read16 --custom-aug` (or any valid flag combination).
  - Outputs: `bozen_training_output/` or `ict91_training_output/` with checkpoints, `results.json`, final model (+ pretrain artifacts if `--pretrain`).

### Analysis scripts on `combined_token_results.csv`

- `threshold_analysis.py`
  - Purpose: sweeps CER thresholds and compares balanced accuracy for ML, GradCAM ablation, and filtered-token variants.
  - Run: `uv run python threshold_analysis.py`.
  - Outputs: `ml_experiment_threshold_analysis.png`, `gradcam_ablation_threshold_analysis.png`, `filtered_token_threshold_analysis.png`.
- `cer_histogram.py`
  - Purpose: plots token-level and sample-level CER histograms and prints threshold-coverage stats.
  - Run: `uv run python cer_histogram.py`.
  - Outputs: `cer_distribution_histogram.png`, `sample_cer_distribution_histogram.png`.
- `correlations.py`
  - Purpose: computes/plots a correlation heatmap for numeric token metrics.
  - Run: `uv run python correlations.py`.
  - Outputs: `seaborn_plot_2_heatmap.png`.
- `scatter_plots.py`
  - Purpose: generates regression scatter plots of `token_loss` vs key GradCAM/attention metrics.
  - Run: `uv run python scatter_plots.py`.
  - Outputs: `seaborn_scatter_token_loss_vs_<metric>.png` (multiple files).

### Visualization utility

- `visualize_heatmaps.py`
  - Purpose: post-processes raw token heatmaps, overlays on input image, and builds a combined heatmap view.
  - Run: `uv run python visualize_heatmaps.py <sample_number> <aspect_ratio> [--opacity 0.5]`.
  - Outputs: `combined_visualizations/sample_<id>/processed_heatmaps/` with overlay images and `full_heatmap_combined.png`.

Notes:
- Most analysis scripts require `combined_token_results.csv` in the current working directory.
- `ablation_study_experiments.py` and `combined_token_level_analysis_for_grad_cam_and_attention.py` are notebook-export scripts with Colab-specific assumptions (`/content`, Drive mount, in-script dependency installs).
