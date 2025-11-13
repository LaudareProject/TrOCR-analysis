#!/usr/bin/env python3
"""
Machine Learning Experiment: Binary Classification of Image Quality using Attention and GradCAM Metrics
Good samples: CER = 0
Bad samples: CER > 0
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import urllib.parse
import sys
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(csv_path):
    """Load CSV data and aggregate token-level features by sample_id"""
    print("Loading data...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows with columns: {list(df.columns)}")
    
    # Decode URL-encoded column names
    df.columns = [urllib.parse.unquote_plus(col) for col in df.columns]
    print(f"Decoded column names: {list(df.columns)}")
    
    # Define original feature columns for aggregation
    gradcam_features_orig = ['gradcam_gini', 'gradcam_coverage', 'gradcam_entropy']  # excluding gradcam_peak
    attention_features_orig = ['attention_gini', 'attention_coverage', 'attention_entropy']  # excluding attention_peak
    
    # After aggregation, column names will have suffixes: _mean, _min, _max
    gradcam_features = [
        'gradcam_gini_mean', 'gradcam_gini_min', 'gradcam_gini_max',
        'gradcam_coverage_mean', 'gradcam_coverage_min', 'gradcam_coverage_max',
        'gradcam_entropy_mean', 'gradcam_entropy_min', 'gradcam_entropy_max'
    ]
    attention_features = [
        'attention_gini_mean', 'attention_gini_min', 'attention_gini_max',
        'attention_coverage_mean', 'attention_coverage_min', 'attention_coverage_max',
        'attention_entropy_mean', 'attention_entropy_min', 'attention_entropy_max'
    ]
    loss_features = ['token_loss_mean', 'token_loss_max', 'token_loss_std']  # multiple loss aggregations
    
    print(f"GradCAM features: {gradcam_features}")
    print(f"Attention features: {attention_features}")
    print(f"Loss features: {loss_features}")
    
    # Aggregate features by sample_id using mean
    print("Aggregating token-level features by sample_id...")
    
    # Group by sample_id and compute aggregations
    agg_dict = {}
    
    # Mean, min, max aggregation for gradcam and attention features
    for feature in gradcam_features_orig + attention_features_orig:
        agg_dict[feature] = ['mean', 'min', 'max']
    
    # Multiple aggregations for token loss
    agg_dict['token_loss'] = ['mean', 'max', 'std']
    
    # Take first value of image_cer since it should be the same for all tokens in the same sample
    agg_dict['image_cer'] = 'first'
    
    aggregated_df = df.groupby('sample_id').agg(agg_dict).reset_index()
    
    # Flatten column names for multi-level aggregations
    new_columns = []
    for col in aggregated_df.columns:
        if isinstance(col, tuple):
            if col[1] == '' or col[1] == 'first':
                new_columns.append(col[0])  # Keep original name for single aggregations
            else:
                new_columns.append(f"{col[0]}_{col[1]}")  # Create new name for multiple aggregations
        else:
            new_columns.append(col)  # Single-level column names
    
    aggregated_df.columns = new_columns
    
    print(f"Aggregated to {len(aggregated_df)} samples")
    print(f"Target (image_cer) statistics:")
    print(aggregated_df['image_cer'].describe())
    
    return aggregated_df, gradcam_features, attention_features, loss_features

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
    for feature, coef_mean, coef_std in zip(feature_names, avg_coefficients, std_coefficients):
        print(f"  {feature}: {coef_mean:8.4f} ± {coef_std:6.4f}")
    
    return {
        'accuracy': accuracy,
        'y_true': y_true_all,
        'y_pred': y_pred_all,
        'y_pred_proba': y_pred_proba_all,
        'feature_coefficients': dict(zip(feature_names, avg_coefficients)),
        'feature_coefficients_std': dict(zip(feature_names, std_coefficients)),
        'classification_report': classification_report(y_true_all, y_pred_all, target_names=target_names, output_dict=True)
    }

def main():
    # Check if threshold argument is provided
    if len(sys.argv) != 2:
        print("Usage: python ml_experiment.py <threshold>")
        print("Example: python ml_experiment.py 0.1")
        sys.exit(1)
    
    try:
        threshold = float(sys.argv[1])
    except ValueError:
        print("Error: Threshold must be a number")
        sys.exit(1)
    
    csv_path = 'combined_token_results.csv'
    
    # Load and preprocess data
    df, gradcam_features, attention_features, loss_features = load_and_preprocess_data(csv_path)
    
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
    
    # Experiment 1: Only GradCAM metrics (excluding peaks)
    X_gradcam = df[gradcam_features].values
    results_gradcam = run_experiment(X_gradcam, y, gradcam_features, "GradCAM Only", threshold)
    
    # Experiment 2: Only Attention metrics (excluding peaks)
    X_attention = df[attention_features].values
    results_attention = run_experiment(X_attention, y, attention_features, "Attention Only", threshold)
    
    # Experiment 3: Both GradCAM and Attention metrics (excluding peaks)
    combined_features = gradcam_features + attention_features
    X_combined = df[combined_features].values
    results_combined = run_experiment(X_combined, y, combined_features, "GradCAM + Attention", threshold)
    
    # Experiment 4: Only Token Loss features
    X_loss = df[loss_features].values
    results_loss = run_experiment(X_loss, y, loss_features, "Token Loss Only", threshold)
    
    # Experiment 5: GradCAM + Token Loss
    gradcam_loss_features = gradcam_features + loss_features
    X_gradcam_loss = df[gradcam_loss_features].values
    results_gradcam_loss = run_experiment(X_gradcam_loss, y, gradcam_loss_features, "GradCAM + Token Loss", threshold)
    
    # Experiment 6: All features (GradCAM + Attention + Loss)
    all_features = gradcam_features + attention_features + loss_features
    X_all = df[all_features].values
    results_all = run_experiment(X_all, y, all_features, "All Features (GradCAM + Attention + Loss)", threshold)
    
    # Summary comparison
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    
    experiments = [
        ("GradCAM Only", results_gradcam),
        ("Attention Only", results_attention), 
        ("GradCAM + Attention", results_combined),
        ("Token Loss Only", results_loss),
        ("GradCAM + Token Loss", results_gradcam_loss),
        ("All Features", results_all)
    ]
    
    print(f"{'Experiment':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 70)
    for name, results in experiments:
        # Get macro average metrics
        report = results['classification_report']
        precision = report['macro avg']['precision']
        recall = report['macro avg']['recall']
        f1 = report['macro avg']['f1-score']
        
        print(f"{name:<20} {results['accuracy']:<10.4f} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f}")
    
    # Find best experiment by F1-score
    best_exp = max(experiments, key=lambda x: x[1]['classification_report']['macro avg']['f1-score'])
    best_f1 = best_exp[1]['classification_report']['macro avg']['f1-score']
    print(f"\nBest performing experiment: {best_exp[0]} (F1-Score = {best_f1:.4f})")

if __name__ == "__main__":
    main()