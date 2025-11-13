#!/usr/bin/env python3
"""
Filtered Token Experiment: Compare GradCAM (gini + entropy) with Loss
- GradCAM features computed ONLY from tokens where token_id NOT IN (0, 2)
- Loss features computed from ALL tokens
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, balanced_accuracy_score
import urllib.parse
import sys
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data_filtered(csv_path):
    """Load CSV data and aggregate with filtered tokens for GradCAM"""
    print("Loading data...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows")
    
    # Decode URL-encoded column names
    df.columns = [urllib.parse.unquote_plus(col) for col in df.columns]
    
    print(f"Total tokens before filtering: {len(df)}")
    print(f"Token ID distribution:\n{df['token_id'].value_counts().head(10)}")
    
    # Split data for different aggregations
    # For GradCAM: exclude token_id 0 and 2
    df_gradcam_filtered = df[~df['token_id'].isin([0, 2])].copy()
    print(f"\nTokens after filtering (excluding token_id 0 and 2): {len(df_gradcam_filtered)}")
    print(f"Filtered out {len(df) - len(df_gradcam_filtered)} tokens")
    
    # For Loss: use all tokens
    df_loss_all = df.copy()
    
    # Define feature columns
    gradcam_features_orig = ['gradcam_gini', 'gradcam_entropy']  # Only gini and entropy
    
    # Aggregate GradCAM features from filtered data
    print("\nAggregating GradCAM features (filtered tokens)...")
    agg_dict_gradcam = {}
    for feature in gradcam_features_orig:
        agg_dict_gradcam[feature] = ['mean', 'min', 'max']
    agg_dict_gradcam['image_cer'] = 'first'
    
    gradcam_aggregated = df_gradcam_filtered.groupby('sample_id').agg(agg_dict_gradcam).reset_index()
    
    # Flatten column names for GradCAM
    new_columns_gradcam = []
    for col in gradcam_aggregated.columns:
        if isinstance(col, tuple):
            if col[1] == '' or col[1] == 'first':
                new_columns_gradcam.append(col[0])
            else:
                new_columns_gradcam.append(f"{col[0]}_{col[1]}")
        else:
            new_columns_gradcam.append(col)
    
    gradcam_aggregated.columns = new_columns_gradcam
    
    # Aggregate Loss features from all tokens
    print("Aggregating Loss features (all tokens)...")
    agg_dict_loss = {
        'token_loss': ['mean', 'max', 'std'],
        'image_cer': 'first'
    }
    
    loss_aggregated = df_loss_all.groupby('sample_id').agg(agg_dict_loss).reset_index()
    
    # Flatten column names for Loss
    new_columns_loss = []
    for col in loss_aggregated.columns:
        if isinstance(col, tuple):
            if col[1] == '' or col[1] == 'first':
                new_columns_loss.append(col[0])
            else:
                new_columns_loss.append(f"{col[0]}_{col[1]}")
        else:
            new_columns_loss.append(col)
    
    loss_aggregated.columns = new_columns_loss
    
    # Merge the two aggregated dataframes
    print("Merging GradCAM and Loss features...")
    merged_df = pd.merge(gradcam_aggregated, loss_aggregated[['sample_id', 'token_loss_mean', 'token_loss_max', 'token_loss_std']], 
                         on='sample_id', how='inner')
    
    print(f"Final merged dataset: {len(merged_df)} samples")
    
    # Define final feature lists
    gini_features = ['gradcam_gini_mean', 'gradcam_gini_min', 'gradcam_gini_max']
    entropy_features = ['gradcam_entropy_mean', 'gradcam_entropy_min', 'gradcam_entropy_max']
    loss_features = ['token_loss_mean', 'token_loss_max', 'token_loss_std']
    
    print(f"\nGini features: {gini_features}")
    print(f"Entropy features: {entropy_features}")
    print(f"Loss features: {loss_features}")
    
    return merged_df, gini_features, entropy_features, loss_features

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
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
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
    balanced_acc = balanced_accuracy_score(y_true_all, y_pred_all)
    
    print(f"Leave-One-Out CV Performance:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Balanced Accuracy: {balanced_acc:.4f}")
    
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
    for feature, coef_mean, coef_std in zip(feature_names, avg_coefficients, std_coefficients):
        print(f"  {feature}: {coef_mean:8.4f} ± {coef_std:6.4f}")
    
    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'y_true': y_true_all,
        'y_pred': y_pred_all,
        'y_pred_proba': y_pred_proba_all,
        'feature_coefficients': dict(zip(feature_names, avg_coefficients)),
        'feature_coefficients_std': dict(zip(feature_names, std_coefficients)),
        'classification_report': report_dict
    }

def main():
    # Check if threshold argument is provided
    if len(sys.argv) != 2:
        print("Usage: python filtered_token_experiment.py <threshold>")
        print("Example: python filtered_token_experiment.py 0.1")
        sys.exit(1)
    
    try:
        threshold = float(sys.argv[1])
    except ValueError:
        print("Error: Threshold must be a number")
        sys.exit(1)
    
    csv_path = 'combined_token_results.csv'
    
    # Load and preprocess data with filtered tokens
    df, gini_features, entropy_features, loss_features = load_and_preprocess_data_filtered(csv_path)
    
    # Prepare target variable - binary classification
    y = (df['image_cer'] > threshold).astype(int).values
    
    print(f"\nUsing threshold: {threshold}")
    print(f"Good samples: CER <= {threshold} (class 0)")
    print(f"Bad samples: CER > {threshold} (class 1)")
    
    print(f"\nClass distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for cls, count in zip(unique, counts):
        class_name = f"Good (CER<={threshold})" if cls == 0 else f"Bad (CER>{threshold})"
        print(f"  {class_name}: {count} samples ({count/len(y)*100:.1f}%)")
    
    # Experiment 1: Only Gini (filtered tokens)
    X_gini = df[gini_features].values
    results_gini = run_experiment(X_gini, y, gini_features, "Gini Only (filtered)", threshold)
    
    # Experiment 2: Only Entropy (filtered tokens)
    X_entropy = df[entropy_features].values
    results_entropy = run_experiment(X_entropy, y, entropy_features, "Entropy Only (filtered)", threshold)
    
    # Experiment 3: Gini + Entropy (filtered tokens)
    gini_entropy_features = gini_features + entropy_features
    X_gini_entropy = df[gini_entropy_features].values
    results_gini_entropy = run_experiment(X_gini_entropy, y, gini_entropy_features, "Gini + Entropy (filtered)", threshold)
    
    # Experiment 4: Only Loss (all tokens)
    X_loss = df[loss_features].values
    results_loss = run_experiment(X_loss, y, loss_features, "Loss Only (all tokens)", threshold)
    
    # Experiment 5: Gini + Loss
    gini_loss_features = gini_features + loss_features
    X_gini_loss = df[gini_loss_features].values
    results_gini_loss = run_experiment(X_gini_loss, y, gini_loss_features, "Gini (filtered) + Loss (all tokens)", threshold)
    
    # Experiment 6: Entropy + Loss
    entropy_loss_features = entropy_features + loss_features
    X_entropy_loss = df[entropy_loss_features].values
    results_entropy_loss = run_experiment(X_entropy_loss, y, entropy_loss_features, "Entropy (filtered) + Loss (all tokens)", threshold)
    
    # Experiment 7: Gini + Entropy + Loss
    all_features = gini_features + entropy_features + loss_features
    X_all = df[all_features].values
    results_all = run_experiment(X_all, y, all_features, "Gini + Entropy (filtered) + Loss (all tokens)", threshold)
    
    # Summary comparison
    print(f"\n{'='*80}")
    print("FILTERED TOKEN EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    print("NOTE: GradCAM features computed from tokens with token_id NOT IN (0, 2)")
    print("      Loss features computed from ALL tokens")
    print(f"{'='*80}")
    
    experiments = [
        ("Gini Only (filtered)", results_gini),
        ("Entropy Only (filtered)", results_entropy),
        ("Gini + Entropy (filtered)", results_gini_entropy),
        ("Loss Only (all tokens)", results_loss),
        ("Gini + Loss", results_gini_loss),
        ("Entropy + Loss", results_entropy_loss),
        ("All Features", results_all)
    ]
    
    print(f"{'Experiment':<35} {'Accuracy':<10} {'Bal.Acc.':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 95)
    for name, results in experiments:
        report = results['classification_report']
        precision = report['macro avg']['precision']
        recall = report['macro avg']['recall']
        f1 = report['macro avg']['f1-score']
        
        print(f"{name:<35} {results['accuracy']:<10.4f} {results['balanced_accuracy']:<10.4f} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f}")
    
    # Find best experiment by balanced accuracy
    best_exp = max(experiments, key=lambda x: x[1]['balanced_accuracy'])
    best_bal_acc = best_exp[1]['balanced_accuracy']
    print(f"\nBest performing experiment: {best_exp[0]} (Balanced Accuracy = {best_bal_acc:.4f})")

if __name__ == "__main__":
    main()
