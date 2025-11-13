#!/usr/bin/env python3
"""
GradCAM Ablation Study: Individual and Pairwise Feature Analysis
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import sys
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(csv_path):
    """Load CSV data and aggregate token-level features by sample_id"""
    print("Loading data...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows")
    
    # Define original feature columns for aggregation
    gradcam_features_orig = ['gradcam_gini', 'gradcam_coverage', 'gradcam_entropy']
    
    print("Aggregating token-level features by sample_id...")
    
    # Group by sample_id and compute mean, min, max aggregation
    agg_dict = {}
    for feature in gradcam_features_orig:
        agg_dict[feature] = ['mean', 'min', 'max']
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
    print(f"Aggregated to {len(aggregated_df)} samples")
    
    # After aggregation with multiple functions, column names have suffixes
    # Group features by metric type (each metric has mean, min, max)
    gini_features = ['gradcam_gini_mean', 'gradcam_gini_min', 'gradcam_gini_max']
    coverage_features = ['gradcam_coverage_mean', 'gradcam_coverage_min', 'gradcam_coverage_max']
    entropy_features = ['gradcam_entropy_mean', 'gradcam_entropy_min', 'gradcam_entropy_max']
    
    return aggregated_df, gini_features, coverage_features, entropy_features

def run_experiment(X, y, feature_names, experiment_name, threshold=0):
    """Run a single binary classification experiment with Leave-One-Out Cross-Validation"""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"Features used: {feature_names}")
    print(f"{'='*60}")
    
    print(f"Using Leave-One-Out Cross-Validation with {len(X)} samples")
    
    # Initialize Leave-One-Out cross-validator
    loo = LeaveOneOut()
    
    # Store predictions for all folds
    y_true_all = []
    y_pred_all = []
    y_pred_proba_all = []
    feature_coefficients_all = []
    
    # Perform Leave-One-Out cross-validation
    for fold, (train_idx, test_idx) in enumerate(loo.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Logistic Regression model
        model = LogisticRegression(penalty='elasticnet', C=1.0, l1_ratio=0.5, 
                                  solver='saga', random_state=42, max_iter=1000)
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]  # Probability of positive class
        
        # Store results
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
        y_pred_proba_all.extend(y_pred_proba)
        feature_coefficients_all.append(model.coef_[0])
    
    # Convert to numpy arrays
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    y_pred_proba_all = np.array(y_pred_proba_all)
    
    # Calculate overall metrics
    accuracy = accuracy_score(y_true_all, y_pred_all)
    
    print(f"Leave-One-Out CV Performance:")
    print(f"  Accuracy: {accuracy:.4f}")
    
    # Detailed classification report
    print(f"\nClassification Report:")
    target_names = [f'Good (CER<={threshold})', f'Bad (CER>{threshold})']
    report_dict = classification_report(y_true_all, y_pred_all, target_names=target_names, output_dict=True)
    print(classification_report(y_true_all, y_pred_all, target_names=target_names))
    
    # Confusion matrix
    print(f"Confusion Matrix:")
    cm = confusion_matrix(y_true_all, y_pred_all)
    print(f"              Predicted")
    print(f"              Good  Bad")
    print(f"Actual Good   {cm[0,0]:4d}  {cm[0,1]:3d}")
    print(f"       Bad    {cm[1,0]:4d}  {cm[1,1]:3d}")
    
    # Average feature coefficients across all folds
    avg_coefficients = np.mean(feature_coefficients_all, axis=0)
    std_coefficients = np.std(feature_coefficients_all, axis=0)
    
    print(f"\nFeature Coefficients (mean ± std across {len(feature_coefficients_all)} folds):")
    feature_coefficients = {}
    if len(feature_names) == 1:
        coef_mean = avg_coefficients[0]
        coef_std = std_coefficients[0]
        print(f"  {feature_names[0]}: {coef_mean:8.4f} ± {coef_std:6.4f}")
        feature_coefficients = {feature_names[0]: coef_mean}
    else:
        for feature, coef_mean, coef_std in zip(feature_names, avg_coefficients, std_coefficients):
            print(f"  {feature}: {coef_mean:8.4f} ± {coef_std:6.4f}")
            feature_coefficients[feature] = coef_mean
    
    return {
        'accuracy': accuracy,
        'y_true': y_true_all,
        'y_pred': y_pred_all,
        'y_pred_proba': y_pred_proba_all,
        'feature_coefficients': feature_coefficients,
        'feature_coefficients_std': dict(zip(feature_names, std_coefficients)),
        'classification_report': report_dict,
        'f1_score': report_dict['macro avg']['f1-score'],
        'precision': report_dict['macro avg']['precision'],
        'recall': report_dict['macro avg']['recall']
    }

def main():
    # Check if threshold argument is provided
    if len(sys.argv) != 2:
        print("Usage: python gradcam_ablation.py <threshold>")
        print("Example: python gradcam_ablation.py 0.1")
        sys.exit(1)
    
    try:
        threshold = float(sys.argv[1])
    except ValueError:
        print("Error: Threshold must be a number")
        sys.exit(1)
    
    csv_path = 'combined_token_results.csv'
    
    # Load and preprocess data
    df, gini_features, coverage_features, entropy_features = load_and_preprocess_data(csv_path)
    
    # Create a dictionary mapping metric names to their feature sets
    metric_groups = {
        'gini': gini_features,
        'coverage': coverage_features,
        'entropy': entropy_features
    }
    
    # Prepare target variable - binary classification
    # Good samples: CER <= threshold (class 0)
    # Bad samples: CER > threshold (class 1)
    y = (df['image_cer'] > threshold).astype(int).values
    
    print(f"\nUsing threshold: {threshold}")
    print(f"Good samples: CER <= {threshold} (class 0)")
    print(f"Bad samples: CER > {threshold} (class 1)")
    
    print(f"\nClass distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for cls, count in zip(unique, counts):
        class_name = f"Good (CER<={threshold})" if cls == 0 else f"Bad (CER>{threshold})"
        print(f"  {class_name}: {count} samples ({count/len(y)*100:.1f}%)")
    
    # Individual metric experiments (each with mean, min, max)
    individual_results = []
    
    for metric_name, features in metric_groups.items():
        X_single = df[features].values
        result = run_experiment(X_single, y, features, f"Single Metric: {metric_name}", threshold)
        individual_results.append((metric_name, result))
    
    # Pairwise metric experiments
    pairwise_results = []
    
    metric_names = list(metric_groups.keys())
    for i in range(len(metric_names)):
        for j in range(i+1, len(metric_names)):
            metric1 = metric_names[i]
            metric2 = metric_names[j]
            combined_features = metric_groups[metric1] + metric_groups[metric2]
            
            X_pair = df[combined_features].values
            result = run_experiment(X_pair, y, combined_features, f"Pair: {metric1} + {metric2}", threshold)
            pairwise_results.append((f"{metric1} + {metric2}", result))
    
    # All three metrics (for comparison)
    all_features = gini_features + coverage_features + entropy_features
    X_all = df[all_features].values
    result_all = run_experiment(X_all, y, all_features, "All Three: gini + coverage + entropy", threshold)
    
    # Summary comparison
    print(f"\n{'='*80}")
    print("GRADCAM ABLATION STUDY SUMMARY")
    print(f"{'='*80}")
    
    all_experiments = []
    
    # Add individual results
    for name, result in individual_results:
        all_experiments.append((f"Single: {name}", result))
    
    # Add pairwise results
    for name, result in pairwise_results:
        all_experiments.append((f"Pair: {name}", result))
    
    # Add all features result
    all_experiments.append(("All Three", result_all))
    
    print(f"{'Experiment':<35} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 85)
    
    for name, result in all_experiments:
        print(f"{name:<35} {result['accuracy']:<10.4f} {result['precision']:<10.4f} {result['recall']:<10.4f} {result['f1_score']:<10.4f}")
    
    # Find best experiment
    best_exp = max(all_experiments, key=lambda x: x[1]['f1_score'])
    print(f"\nBest performing experiment: {best_exp[0]} (F1-Score = {best_exp[1]['f1_score']:.4f})")
    
    # Feature importance analysis
    print(f"\n{'='*80}")
    print("FEATURE IMPORTANCE ANALYSIS")
    print(f"{'='*80}")
    
    print("\nIndividual Feature Performance:")
    individual_sorted = sorted(individual_results, key=lambda x: x[1]['f1_score'], reverse=True)
    for name, result in individual_sorted:
        coef = list(result['feature_coefficients'].values())[0]
        print(f"  {name:<12}: F1={result['f1_score']:.4f}, Coef={coef:7.4f}")
    
    print(f"\nBest Pairwise Combinations:")
    pairwise_sorted = sorted(pairwise_results, key=lambda x: x[1]['f1_score'], reverse=True)
    for name, result in pairwise_sorted[:3]:  # Top 3
        print(f"  {name:<25}: F1={result['f1_score']:.4f}")

if __name__ == "__main__":
    main()