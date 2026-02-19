#!/usr/bin/env python3
"""
Threshold Analysis Using a Global CER Threshold from KMeans
Compares ML Experiment and GradCAM Ablation performance at one data-driven threshold
"""

import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, balanced_accuracy_score
from scipy.stats import chi2
import urllib.parse
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data_ml(csv_path, excluded_token_ids=None):
    """Load CSV data and aggregate token-level features by sample_id for ML experiment"""
    modality_label = "unfiltered" if excluded_token_ids is None else f"filtered token_id NOT IN {excluded_token_ids}"
    print(f"Loading data for ML experiment ({modality_label})...")
    df = pd.read_csv(csv_path)

    # Decode URL-encoded column names
    df.columns = [urllib.parse.unquote_plus(col) for col in df.columns]

    if excluded_token_ids is not None:
        df = df[~df['token_id'].isin(excluded_token_ids)].copy()

    # Define original feature columns for aggregation
    gradcam_features_orig = ['gradcam_gini', 'gradcam_coverage', 'gradcam_entropy']
    attention_features_orig = ['attention_gini', 'attention_coverage', 'attention_entropy']

    # After aggregation, column names will have suffixes: _mean, _min, _max, _std, _median
    gradcam_features = [
        'gradcam_gini_mean', 'gradcam_gini_min', 'gradcam_gini_max',
        'gradcam_gini_std', 'gradcam_gini_median',
        'gradcam_coverage_mean', 'gradcam_coverage_min', 'gradcam_coverage_max',
        'gradcam_coverage_std', 'gradcam_coverage_median',
        'gradcam_entropy_mean', 'gradcam_entropy_min', 'gradcam_entropy_max',
        'gradcam_entropy_std', 'gradcam_entropy_median'
    ]
    attention_features = [
        'attention_gini_mean', 'attention_gini_min', 'attention_gini_max',
        'attention_gini_std', 'attention_gini_median',
        'attention_coverage_mean', 'attention_coverage_min', 'attention_coverage_max',
        'attention_coverage_std', 'attention_coverage_median',
        'attention_entropy_mean', 'attention_entropy_min', 'attention_entropy_max',
        'attention_entropy_std', 'attention_entropy_median'
    ]
    loss_features = ['token_loss_mean', 'token_loss_max', 'token_loss_std', 'token_loss_median']

    # Aggregate features by sample_id using mean, min, max
    agg_dict = {}

    # Mean, min, max, std, median aggregation for gradcam and attention features
    for feature in gradcam_features_orig + attention_features_orig:
        agg_dict[feature] = ['mean', 'min', 'max', 'std', 'median']

    # Multiple aggregations for token loss
    agg_dict['token_loss'] = ['mean', 'max', 'std', 'median']

    # Take first value of image_cer since it should be the same for all tokens in the same sample
    agg_dict['image_cer'] = 'first'

    aggregated_df = df.groupby('sample_id').agg(agg_dict).reset_index()

    # Flatten column names for multi-level aggregations
    new_columns = []
    for col in aggregated_df.columns:
        if isinstance(col, tuple):
            if col[1] == '' or col[1] == 'first':
                new_columns.append(col[0])
            else:
                new_columns.append(f"{col[0]}_{col[1]}")
        else:
            new_columns.append(col)

    aggregated_df.columns = new_columns

    # Avoid NaN from std on groups with a single token
    aggregated_df = aggregated_df.fillna(0)

    return aggregated_df, gradcam_features, attention_features, loss_features

def load_and_preprocess_data_gradcam(csv_path, excluded_token_ids=None):
    """Load CSV data and aggregate token-level features by sample_id for GradCAM ablation"""
    modality_label = "unfiltered" if excluded_token_ids is None else f"filtered token_id NOT IN {excluded_token_ids}"
    print(f"Loading data for GradCAM ablation ({modality_label})...")
    df = pd.read_csv(csv_path)
    
    # Decode URL-encoded column names
    df.columns = [urllib.parse.unquote_plus(col) for col in df.columns]

    if excluded_token_ids is not None:
        df = df[~df['token_id'].isin(excluded_token_ids)].copy()
    # Define original feature columns for aggregation
    gradcam_features_orig = ['gradcam_gini', 'gradcam_coverage', 'gradcam_entropy']

    # Group by sample_id and compute aggregations
    agg_dict = {}
    for feature in gradcam_features_orig:
        agg_dict[feature] = ['mean', 'min', 'max', 'std', 'median']
    
    # Multiple aggregations for token loss
    agg_dict['token_loss'] = ['mean', 'max', 'std', 'median']
    agg_dict['image_cer'] = 'first'

    aggregated_df = df.groupby('sample_id').agg(agg_dict).reset_index()

    # Flatten column names
    new_columns = []
    for col in aggregated_df.columns:
        if isinstance(col, tuple):
            if col[1] == '' or col[1] == 'first':
                new_columns.append(col[0])
            else:
                new_columns.append(f"{col[0]}_{col[1]}")
        else:
            new_columns.append(col)

    aggregated_df.columns = new_columns
    aggregated_df = aggregated_df.fillna(0)
    
    # Group features by metric type
    gini_features = ['gradcam_gini_mean', 'gradcam_gini_min', 'gradcam_gini_max', 'gradcam_gini_std', 'gradcam_gini_median']
    coverage_features = ['gradcam_coverage_mean', 'gradcam_coverage_min', 'gradcam_coverage_max', 'gradcam_coverage_std', 'gradcam_coverage_median']
    entropy_features = ['gradcam_entropy_mean', 'gradcam_entropy_min', 'gradcam_entropy_max', 'gradcam_entropy_std', 'gradcam_entropy_median']
    loss_features = ['token_loss_mean', 'token_loss_max', 'token_loss_std', 'token_loss_median']

    return aggregated_df, gini_features, coverage_features, entropy_features, loss_features

def run_experiment_silent(X, y, feature_names, threshold=0, sample_ids=None, return_predictions=False):
    """Run a single binary classification experiment with Leave-One-Out Cross-Validation (silent version)"""
    # Initialize Leave-One-Out cross-validator
    loo = LeaveOneOut()

    # Store predictions for all folds
    y_true_all = []
    y_pred_all = []

    # Perform Leave-One-Out cross-validation
    for fold, (train_idx, test_idx) in enumerate(loo.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Apply PCA after scaling and retain 90% of variance
        pca = PCA(n_components=0.90)
        X_train_transformed = pca.fit_transform(X_train_scaled)
        X_test_transformed = pca.transform(X_test_scaled)

        # Train Logistic Regression model
        model = LogisticRegression(penalty='elasticnet', C=1.0, l1_ratio=0.5,
                                  solver='saga', random_state=42, max_iter=1000)
        model.fit(X_train_transformed, y_train)

        # Make predictions
        y_pred = model.predict(X_test_transformed)

        # Store results
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

    # Convert to numpy arrays
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    # Calculate metrics
    target_names = [f'Good (CER<={threshold})', f'Bad (CER>{threshold})']
    report_dict = classification_report(y_true_all, y_pred_all, target_names=target_names, output_dict=True)

    results = {
        'accuracy': accuracy_score(y_true_all, y_pred_all),
        'balanced_accuracy': balanced_accuracy_score(y_true_all, y_pred_all),
        'recall': report_dict['macro avg']['recall'],
        'precision': report_dict['macro avg']['precision'],
        'f1_score': report_dict['macro avg']['f1-score'],
        'feature_names': list(feature_names),
    }

    if return_predictions:
        results['y_true'] = y_true_all
        results['y_pred'] = y_pred_all
        if sample_ids is not None:
            results['sample_ids'] = np.asarray(sample_ids)

    return results

def run_ml_experiments_for_threshold(df, gradcam_features, attention_features, loss_features, threshold):
    """Run all ML experiments for a given threshold and return results"""
    # Prepare target variable
    y = (df['image_cer'] > threshold).astype(int).values

    # Skip if all samples have the same class
    if len(np.unique(y)) < 2:
        return None

    results = {}

    # Experiment 1: Only GradCAM metrics
    X_gradcam = df[gradcam_features].values
    results['GradCAM Only'] = run_experiment_silent(X_gradcam, y, gradcam_features, threshold)

    # Experiment 2: Only Attention metrics
    X_attention = df[attention_features].values
    results['Attention Only'] = run_experiment_silent(X_attention, y, attention_features, threshold)

    # Experiment 3: Only Token Loss features
    X_loss = df[loss_features].values
    results['Token Loss Only'] = run_experiment_silent(X_loss, y, loss_features, threshold)

    # Experiment 4: GradCAM + Attention
    gradcam_attention_features = gradcam_features + attention_features
    X_gradcam_attention = df[gradcam_attention_features].values
    results['GradCAM + Attention'] = run_experiment_silent(X_gradcam_attention, y, gradcam_attention_features, threshold)

    # Experiment 5: GradCAM + Token Loss
    gradcam_loss_features = gradcam_features + loss_features
    X_gradcam_loss = df[gradcam_loss_features].values
    results['GradCAM + Token Loss'] = run_experiment_silent(X_gradcam_loss, y, gradcam_loss_features, threshold)

    # Experiment 6: Attention + Token Loss
    attention_loss_features = attention_features + loss_features
    X_attention_loss = df[attention_loss_features].values
    results['Attention + Token Loss'] = run_experiment_silent(X_attention_loss, y, attention_loss_features, threshold)

    # Experiment 7: All features
    all_features = gradcam_features + attention_features + loss_features
    X_all = df[all_features].values
    results['All Features'] = run_experiment_silent(X_all, y, all_features, threshold)

    return results

def run_gradcam_experiments_for_threshold(df, gini_features, coverage_features, entropy_features, loss_features, threshold):
    """Run GradCAM ablation experiments for a given threshold and return results"""
    # Prepare target variable
    y = (df['image_cer'] > threshold).astype(int).values

    # Skip if all samples have the same class
    if len(np.unique(y)) < 2:
        return None

    results = {}
    
    # Create metric groups dictionary
    metric_groups = {
        'gini': gini_features,
        'coverage': coverage_features,
        'entropy': entropy_features
    }

    # Individual metric experiments (each with mean, min, max as a group)
    for metric_name, features in metric_groups.items():
        X_single = df[features].values
        results[f'Single: {metric_name}'] = run_experiment_silent(X_single, y, features, threshold)

    # Pairwise metric experiments
    metric_names = list(metric_groups.keys())
    for i in range(len(metric_names)):
        for j in range(i+1, len(metric_names)):
            metric1 = metric_names[i]
            metric2 = metric_names[j]
            combined_features = metric_groups[metric1] + metric_groups[metric2]
            
            X_pair = df[combined_features].values
            results[f'Pair: {metric1} + {metric2}'] = run_experiment_silent(X_pair, y, combined_features, threshold)
    
    # Each individual metric + ALL loss features
    for metric_name, features in metric_groups.items():
        combined_features = features + loss_features
        
        X_combined = df[combined_features].values
        results[f'Pair: {metric_name} + loss'] = run_experiment_silent(X_combined, y, combined_features, threshold)
    
    # Loss alone (all loss features)
    X_loss = df[loss_features].values
    results['Single: loss'] = run_experiment_silent(X_loss, y, loss_features, threshold)

    # All three metrics
    all_gradcam_features = gini_features + coverage_features + entropy_features
    X_all_gradcam = df[all_gradcam_features].values
    results['All Three GradCAM'] = run_experiment_silent(X_all_gradcam, y, all_gradcam_features, threshold)

    return results


def run_attention_experiments_for_threshold(df, gini_features, coverage_features, entropy_features, loss_features, threshold):
    """Run Attention ablation experiments for a given threshold and return results."""
    y = (df['image_cer'] > threshold).astype(int).values

    if len(np.unique(y)) < 2:
        return None

    results = {}

    metric_groups = {
        'gini': gini_features,
        'coverage': coverage_features,
        'entropy': entropy_features,
    }

    for metric_name, features in metric_groups.items():
        X_single = df[features].values
        results[f'Single: {metric_name}'] = run_experiment_silent(X_single, y, features, threshold)

    metric_names = list(metric_groups.keys())
    for i in range(len(metric_names)):
        for j in range(i + 1, len(metric_names)):
            metric1 = metric_names[i]
            metric2 = metric_names[j]
            combined_features = metric_groups[metric1] + metric_groups[metric2]

            X_pair = df[combined_features].values
            results[f'Pair: {metric1} + {metric2}'] = run_experiment_silent(X_pair, y, combined_features, threshold)

    for metric_name, features in metric_groups.items():
        combined_features = features + loss_features

        X_combined = df[combined_features].values
        results[f'Pair: {metric_name} + loss'] = run_experiment_silent(X_combined, y, combined_features, threshold)

    X_loss = df[loss_features].values
    results['Single: loss'] = run_experiment_silent(X_loss, y, loss_features, threshold)

    all_attention_features = gini_features + coverage_features + entropy_features
    X_all_attention = df[all_attention_features].values
    results['All Three Attention'] = run_experiment_silent(X_all_attention, y, all_attention_features, threshold)

    return results

def compute_kmeans_global_threshold(cer_values):
    """Compute a global threshold from CER values using 2-cluster KMeans initialized at 0 and 1."""
    cer_array = np.asarray(cer_values, dtype=float).reshape(-1, 1)

    kmeans = KMeans(
        n_clusters=2,
        init=np.array([[0.0], [1.0]]),
        n_init=1,
        random_state=42,
    )
    cluster_labels = kmeans.fit_predict(cer_array)
    centers = kmeans.cluster_centers_.flatten()

    low_cluster_idx = int(np.argmin(np.abs(centers - 0.0)))
    high_cluster_idx = int(np.argmin(np.abs(centers - 1.0)))

    if low_cluster_idx == high_cluster_idx:
        low_cluster_idx, high_cluster_idx = np.argsort(centers)

    low_cluster_values = cer_array[cluster_labels == low_cluster_idx].flatten()
    high_cluster_values = cer_array[cluster_labels == high_cluster_idx].flatten()

    if len(low_cluster_values) == 0 or len(high_cluster_values) == 0:
        raise ValueError("KMeans produced an empty cluster; cannot compute global threshold.")

    low_cluster_max = float(np.max(low_cluster_values))
    high_cluster_min = float(np.min(high_cluster_values))
    threshold = (low_cluster_max + high_cluster_min) / 2.0

    low_cluster_count = int(len(low_cluster_values))
    high_cluster_count = int(len(high_cluster_values))
    total_count = low_cluster_count + high_cluster_count

    low_cluster_ratio = low_cluster_count / total_count
    high_cluster_ratio = high_cluster_count / total_count

    return (
        threshold,
        centers,
        low_cluster_max,
        high_cluster_min,
        low_cluster_count,
        high_cluster_count,
        low_cluster_ratio,
        high_cluster_ratio,
    )


def evaluate_model_with_sample_ids(df, feature_names, threshold):
    """Evaluate one feature set with LOOCV and return predictions keyed by sample_id."""
    y = (df['image_cer'] > threshold).astype(int).values
    X = df[feature_names].values
    sample_ids = df['sample_id'].values
    return run_experiment_silent(
        X,
        y,
        feature_names,
        threshold,
        sample_ids=sample_ids,
        return_predictions=True,
    )


def run_model_comparison_tests(selected_models, modality_data, threshold):
    """Run Cochran's Q and pairwise McNemar tests on selected models."""
    evaluated_models = []

    for model in selected_models:
        data = modality_data[model['modality']]
        df = data['df_ml'] if model['dataset'] == 'ml' else data['df_gradcam']
        eval_result = evaluate_model_with_sample_ids(df, model['feature_names'], threshold)
        evaluated_models.append(
            {
                'label': model['label'],
                'sample_ids': eval_result['sample_ids'],
                'y_true': eval_result['y_true'],
                'y_pred': eval_result['y_pred'],
            }
        )

    common_sample_ids = set(evaluated_models[0]['sample_ids'])
    for model_eval in evaluated_models[1:]:
        common_sample_ids &= set(model_eval['sample_ids'])
    common_sample_ids = sorted(common_sample_ids)

    if len(common_sample_ids) == 0:
        raise ValueError("No common sample_id found across selected models for statistical tests.")

    aligned_true = None
    correctness_by_model = {}

    for model_eval in evaluated_models:
        by_id = {
            sid: (yt, yp)
            for sid, yt, yp in zip(model_eval['sample_ids'], model_eval['y_true'], model_eval['y_pred'])
            if sid in common_sample_ids
        }

        model_true = np.array([by_id[sid][0] for sid in common_sample_ids])
        model_pred = np.array([by_id[sid][1] for sid in common_sample_ids])

        if aligned_true is None:
            aligned_true = model_true
        elif not np.array_equal(aligned_true, model_true):
            raise ValueError("Inconsistent y_true across selected models after sample alignment.")

        correctness_by_model[model_eval['label']] = (model_pred == aligned_true).astype(int)

    model_labels = [m['label'] for m in evaluated_models]
    correctness_matrix = np.column_stack([correctness_by_model[label] for label in model_labels])

    n_samples, n_models = correctness_matrix.shape
    col_sums = np.sum(correctness_matrix, axis=0)
    row_sums = np.sum(correctness_matrix, axis=1)
    total_sum = np.sum(col_sums)

    numerator = (n_models - 1) * (n_models * np.sum(col_sums ** 2) - total_sum ** 2)
    denominator = n_models * total_sum - np.sum(row_sums ** 2)
    q_stat = numerator / denominator if denominator != 0 else 0.0
    q_pvalue = 1.0 - chi2.cdf(q_stat, df=n_models - 1)

    pairwise_results = []
    pairs = list(itertools.combinations(model_labels, 2))

    for label_a, label_b in pairs:
        a = correctness_by_model[label_a]
        b = correctness_by_model[label_b]

        b_only = int(np.sum((a == 1) & (b == 0)))
        c_only = int(np.sum((a == 0) & (b == 1)))
        discordant = b_only + c_only

        if discordant == 0:
            chi2_stat = 0.0
            p_value = 1.0
        else:
            chi2_stat = ((abs(b_only - c_only) - 1) ** 2) / discordant
            p_value = 1.0 - chi2.cdf(chi2_stat, df=1)

        pairwise_results.append(
            {
                'model_a': label_a,
                'model_b': label_b,
                'b_only': b_only,
                'c_only': c_only,
                'chi2': chi2_stat,
                'p_value': p_value,
            }
        )

    # Holm-Bonferroni correction across pairwise McNemar tests
    m = len(pairwise_results)
    ranked = sorted(enumerate(pairwise_results), key=lambda x: x[1]['p_value'])
    running_max = 0.0
    for rank, (original_idx, result) in enumerate(ranked, start=1):
        holm_value = (m - rank + 1) * result['p_value']
        running_max = max(running_max, holm_value)
        pairwise_results[original_idx]['p_value_holm'] = min(1.0, running_max)

    return {
        'n_common_samples': len(common_sample_ids),
        'cochran_q': {
            'q_stat': q_stat,
            'df': n_models - 1,
            'p_value': q_pvalue,
        },
        'mcnemar_pairwise': pairwise_results,
    }

def main():
    csv_path = 'results/combined_token_results.csv'

    modalities = {
        'unfiltered': None,
        'filtered_0_2': [0, 2],
    }

    modality_data = {}
    for modality_name, excluded_token_ids in modalities.items():
        df_ml, gradcam_features_ml, attention_features, loss_features = load_and_preprocess_data_ml(
            csv_path,
            excluded_token_ids=excluded_token_ids,
        )
        df_gradcam, gini_features, coverage_features, entropy_features, loss_features_gradcam = load_and_preprocess_data_gradcam(
            csv_path,
            excluded_token_ids=excluded_token_ids,
        )
        modality_data[modality_name] = {
            'df_ml': df_ml,
            'gradcam_features_ml': gradcam_features_ml,
            'attention_features': attention_features,
            'attention_gini_features': [f for f in attention_features if f.startswith('attention_gini_')],
            'attention_coverage_features': [f for f in attention_features if f.startswith('attention_coverage_')],
            'attention_entropy_features': [f for f in attention_features if f.startswith('attention_entropy_')],
            'loss_features': loss_features,
            'df_gradcam': df_gradcam,
            'gini_features': gini_features,
            'coverage_features': coverage_features,
            'entropy_features': entropy_features,
            'loss_features_gradcam': loss_features_gradcam,
        }

    (
        threshold,
        centers,
        low_cluster_max,
        high_cluster_min,
        low_cluster_count,
        high_cluster_count,
        low_cluster_ratio,
        high_cluster_ratio,
    ) = compute_kmeans_global_threshold(modality_data['unfiltered']['df_ml']['image_cer'].values)

    print("Using global threshold from KMeans clustering")
    print(f"  Cluster centers: {np.sort(centers)}")
    print(f"  Max CER in low cluster: {low_cluster_max:.6f}")
    print(f"  Min CER in high cluster: {high_cluster_min:.6f}")
    print(f"  Global threshold: {threshold:.6f}")
    print(
        "  Ratio below threshold cluster: "
        f"{low_cluster_ratio:.4f} ({low_cluster_count} samples)"
    )
    print(
        "  Ratio above threshold cluster: "
        f"{high_cluster_ratio:.4f} ({high_cluster_count} samples)"
    )

    results_by_modality = {
        modality_name: {
            'ml': {},
            'gradcam': {},
            'attention': {},
        }
        for modality_name in modalities
    }

    candidates_by_modality = {
        modality_name: []
        for modality_name in modalities
    }

    for modality_name, data in modality_data.items():
        ml_exp_results = run_ml_experiments_for_threshold(
            data['df_ml'],
            data['gradcam_features_ml'],
            data['attention_features'],
            data['loss_features'],
            threshold,
        )
        if ml_exp_results:
            for exp_name, result in ml_exp_results.items():
                results_by_modality[modality_name]['ml'][exp_name] = result['balanced_accuracy']
                candidates_by_modality[modality_name].append(
                    {
                        'group': 'ml',
                        'dataset': 'ml',
                        'modality': modality_name,
                        'exp_name': exp_name,
                        'balanced_accuracy': result['balanced_accuracy'],
                        'feature_names': result['feature_names'],
                    }
                )

        gradcam_exp_results = run_gradcam_experiments_for_threshold(
            data['df_gradcam'],
            data['gini_features'],
            data['coverage_features'],
            data['entropy_features'],
            data['loss_features_gradcam'],
            threshold,
        )
        if gradcam_exp_results:
            for exp_name, result in gradcam_exp_results.items():
                clean_exp_name = exp_name.replace('Single: ', '').replace('Pair: ', '')
                results_by_modality[modality_name]['gradcam'][clean_exp_name] = result['balanced_accuracy']
                candidates_by_modality[modality_name].append(
                    {
                        'group': 'gradcam',
                        'dataset': 'gradcam',
                        'modality': modality_name,
                        'exp_name': clean_exp_name,
                        'balanced_accuracy': result['balanced_accuracy'],
                        'feature_names': result['feature_names'],
                    }
                )

        attention_exp_results = run_attention_experiments_for_threshold(
            data['df_ml'],
            data['attention_gini_features'],
            data['attention_coverage_features'],
            data['attention_entropy_features'],
            data['loss_features'],
            threshold,
        )
        if attention_exp_results:
            for exp_name, result in attention_exp_results.items():
                clean_exp_name = exp_name.replace('Single: ', '').replace('Pair: ', '')
                results_by_modality[modality_name]['attention'][clean_exp_name] = result['balanced_accuracy']
                candidates_by_modality[modality_name].append(
                    {
                        'group': 'attention',
                        'dataset': 'ml',
                        'modality': modality_name,
                        'exp_name': clean_exp_name,
                        'balanced_accuracy': result['balanced_accuracy'],
                        'feature_names': result['feature_names'],
                    }
                )

    selected_models_for_tests_by_modality = {}
    comparison_tests_by_modality = {}

    for modality_name, modality_candidates in candidates_by_modality.items():
        loss_only_candidates = [
            c for c in modality_candidates
            if c['exp_name'].lower() in {'token loss only', 'loss'}
        ]
        gradcam_candidates = [
            c for c in modality_candidates
            if c['group'] == 'gradcam' and c['exp_name'].lower() != 'loss'
        ]
        attention_candidates = [
            c for c in modality_candidates
            if c['group'] == 'attention' and c['exp_name'].lower() != 'loss'
        ]
        ml_candidates = [
            c for c in modality_candidates
            if c['group'] == 'ml' and c['exp_name'].lower() != 'token loss only'
        ]

        best_loss_only = max(loss_only_candidates, key=lambda x: x['balanced_accuracy'])
        best_gradcam = max(gradcam_candidates, key=lambda x: x['balanced_accuracy'])
        best_attention = max(attention_candidates, key=lambda x: x['balanced_accuracy'])
        best_ml = max(ml_candidates, key=lambda x: x['balanced_accuracy'])

        selected_models_for_tests = [
            {
                **best_loss_only,
                'label': f"Loss-only | {best_loss_only['exp_name']}",
            },
            {
                **best_gradcam,
                'label': f"Top GradCAM | {best_gradcam['exp_name']}",
            },
            {
                **best_attention,
                'label': f"Top Attention | {best_attention['exp_name']}",
            },
            {
                **best_ml,
                'label': f"Top ML | {best_ml['exp_name']}",
            },
        ]

        selected_models_for_tests_by_modality[modality_name] = selected_models_for_tests
        comparison_tests_by_modality[modality_name] = run_model_comparison_tests(
            selected_models_for_tests,
            modality_data,
            threshold,
        )

    # Create plots
    plt.style.use('default')
    sns.set_style("whitegrid")

    def plot_balanced_accuracy_bars(results_dict, title, output_file):
        sorted_results = sorted(results_dict.items(), key=lambda item: item[1], reverse=True)
        labels = [item[0] for item in sorted_results]
        values = [item[1] for item in sorted_results]

        plt.figure(figsize=(10, max(6, 0.35 * len(labels))))
        colors = sns.color_palette("colorblind", n_colors=len(labels))
        plt.barh(labels, values, color=colors)
        plt.gca().invert_yaxis()

        plt.xlabel('Balanced Accuracy', fontsize=12)
        plt.title(title, fontsize=12)
        plt.xlim(0.0, 1.0)
        plt.grid(True, axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()

    for modality_name, modality_results in results_by_modality.items():
        plot_balanced_accuracy_bars(
            modality_results['ml'],
            f"ML Experiments ({modality_name}, threshold = {threshold:.4f})",
            f"results/plots/ml_experiment_threshold_analysis_{modality_name}.png",
        )

        plot_balanced_accuracy_bars(
            modality_results['gradcam'],
            f"GradCAM Ablation ({modality_name}, threshold = {threshold:.4f})",
            f"results/plots/gradcam_ablation_threshold_analysis_{modality_name}.png",
        )

        plot_balanced_accuracy_bars(
            modality_results['attention'],
            f"Attention Ablation ({modality_name}, threshold = {threshold:.4f})",
            f"results/plots/attention_ablation_threshold_analysis_{modality_name}.png",
        )

    # Print summary statistics
    print(f"\n{'='*80}")
    print("GLOBAL THRESHOLD ANALYSIS SUMMARY")
    print(f"{'='*80}")
    print(f"Global threshold used: {threshold:.6f}")

    for modality_name, modality_results in results_by_modality.items():
        print(f"\nModality: {modality_name}")
        print("ML Experiment - Balanced Accuracy by Experiment:")
        for exp_name, balanced_acc in modality_results['ml'].items():
            print(f"  {exp_name:<35}: {balanced_acc:.4f}")

        print("GradCAM Ablation - Balanced Accuracy by Experiment:")
        for exp_name, balanced_acc in modality_results['gradcam'].items():
            print(f"  {exp_name:<35}: {balanced_acc:.4f}")

        print("Attention Ablation - Balanced Accuracy by Experiment:")
        for exp_name, balanced_acc in modality_results['attention'].items():
            print(f"  {exp_name:<35}: {balanced_acc:.4f}")

    for modality_name in modalities:
        selected_models_for_tests = selected_models_for_tests_by_modality[modality_name]
        comparison_tests = comparison_tests_by_modality[modality_name]

        print(f"\nSelected models for significance tests ({modality_name}):")
        for model in selected_models_for_tests:
            print(
                f"  {model['label']:<70}: "
                f"BA={model['balanced_accuracy']:.4f}"
            )

        print("\nCochran's Q test (overall difference across 4 selected models):")
        print(
            f"  n_common_samples={comparison_tests['n_common_samples']}, "
            f"Q={comparison_tests['cochran_q']['q_stat']:.4f}, "
            f"df={comparison_tests['cochran_q']['df']}, "
            f"p={comparison_tests['cochran_q']['p_value']:.6g}"
        )

        print("\nPairwise McNemar tests (Holm-Bonferroni-corrected):")
        for test_res in comparison_tests['mcnemar_pairwise']:
            print(
                f"  {test_res['model_a']}  vs  {test_res['model_b']}\n"
                f"    b_only={test_res['b_only']}, c_only={test_res['c_only']}, "
                f"chi2={test_res['chi2']:.4f}, p={test_res['p_value']:.6g}, "
                f"p_holm={test_res['p_value_holm']:.6g}"
            )

if __name__ == "__main__":
    main()
