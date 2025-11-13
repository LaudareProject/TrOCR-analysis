#!/usr/bin/env python3
"""
Threshold Analysis: Recall Curves for Different CER Thresholds
Compares ML Experiment and GradCAM Ablation performance across varying thresholds
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, balanced_accuracy_score
import urllib.parse
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data_ml(csv_path):
    """Load CSV data and aggregate token-level features by sample_id for ML experiment"""
    print("Loading data for ML experiment...")
    df = pd.read_csv(csv_path)

    # Decode URL-encoded column names
    df.columns = [urllib.parse.unquote_plus(col) for col in df.columns]

    # Define original feature columns for aggregation
    gradcam_features_orig = ['gradcam_gini', 'gradcam_coverage', 'gradcam_entropy']
    attention_features_orig = ['attention_gini', 'attention_coverage', 'attention_entropy']

    # After aggregation, column names will have "_mean" suffix
    gradcam_features = ['gradcam_gini_mean', 'gradcam_coverage_mean', 'gradcam_entropy_mean']
    attention_features = ['attention_gini_mean', 'attention_coverage_mean', 'attention_entropy_mean']
    loss_features = ['token_loss_mean', 'token_loss_max', 'token_loss_std']

    # Aggregate features by sample_id using mean
    agg_dict = {}

    # Mean aggregation for gradcam and attention features
    for feature in gradcam_features_orig + attention_features_orig:
        agg_dict[feature] = 'mean'

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
                new_columns.append(col[0])
            else:
                new_columns.append(f"{col[0]}_{col[1]}")
        else:
            new_columns.append(col)

    aggregated_df.columns = new_columns

    return aggregated_df, gradcam_features, attention_features, loss_features

def load_and_preprocess_data_gradcam(csv_path):
    """Load CSV data and aggregate token-level features by sample_id for GradCAM ablation"""
    print("Loading data for GradCAM ablation...")
    df = pd.read_csv(csv_path)
    
    # Decode URL-encoded column names
    df.columns = [urllib.parse.unquote_plus(col) for col in df.columns]
    # Define original feature columns for aggregation
    gradcam_features_orig = ['gradcam_gini', 'gradcam_coverage', 'gradcam_entropy']

    # Group by sample_id and compute aggregations
    agg_dict = {}
    for feature in gradcam_features_orig:
        agg_dict[feature] = 'mean'
    
    # Multiple aggregations for token loss
    agg_dict['token_loss'] = ['mean', 'max', 'std']
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
    
    gradcam_features = ['gradcam_gini_mean', 'gradcam_coverage_mean', 'gradcam_entropy_mean']
    loss_features = ['token_loss_mean', 'token_loss_max', 'token_loss_std']

    return aggregated_df, gradcam_features, loss_features

def run_experiment_silent(X, y, feature_names, threshold=0):
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

        # Train Logistic Regression model
        model = LogisticRegression(penalty='elasticnet', C=1.0, l1_ratio=0.5,
                                  solver='saga', random_state=42, max_iter=1000)
        model.fit(X_train_scaled, y_train)

        # Make predictions
        y_pred = model.predict(X_test_scaled)

        # Store results
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

    # Convert to numpy arrays
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    # Calculate metrics
    target_names = [f'Good (CER<={threshold})', f'Bad (CER>{threshold})']
    report_dict = classification_report(y_true_all, y_pred_all, target_names=target_names, output_dict=True)

    return {
        'accuracy': accuracy_score(y_true_all, y_pred_all),
        'balanced_accuracy': balanced_accuracy_score(y_true_all, y_pred_all),
        'recall': report_dict['macro avg']['recall'],
        'precision': report_dict['macro avg']['precision'],
        'f1_score': report_dict['macro avg']['f1-score']
    }

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

def run_gradcam_experiments_for_threshold(df, gradcam_features, loss_features, threshold):
    """Run GradCAM ablation experiments for a given threshold and return results"""
    # Prepare target variable
    y = (df['image_cer'] > threshold).astype(int).values

    # Skip if all samples have the same class
    if len(np.unique(y)) < 2:
        return None

    results = {}

    # Individual GradCAM feature experiments
    for feature in gradcam_features:
        feature_name = feature.replace('gradcam_', '').replace('_mean', '')
        X_single = df[[feature]].values
        results[f'Single: {feature_name}'] = run_experiment_silent(X_single, y, [feature], threshold)

    # Pairwise GradCAM feature experiments (original pairs)
    for i in range(len(gradcam_features)):
        for j in range(i+1, len(gradcam_features)):
            feature1 = gradcam_features[i]
            feature2 = gradcam_features[j]
            feature_names = [feature1, feature2]

            name1 = feature1.replace('gradcam_', '').replace('_mean', '')
            name2 = feature2.replace('gradcam_', '').replace('_mean', '')

            X_pair = df[feature_names].values
            results[f'Pair: {name1} + {name2}'] = run_experiment_silent(X_pair, y, feature_names, threshold)
    
    # NEW: Each individual GradCAM feature + ALL loss features
    for gradcam_feature in gradcam_features:
        feature_names = [gradcam_feature] + loss_features
        gradcam_name = gradcam_feature.replace('gradcam_', '').replace('_mean', '')
        
        X_combined = df[feature_names].values
        results[f'Pair: {gradcam_name} + loss'] = run_experiment_silent(X_combined, y, feature_names, threshold)
    
    # NEW: Loss alone (all loss features)
    X_loss = df[loss_features].values
    results['Single: loss'] = run_experiment_silent(X_loss, y, loss_features, threshold)

    # All three GradCAM features
    X_all_gradcam = df[gradcam_features].values
    results['All Three GradCAM'] = run_experiment_silent(X_all_gradcam, y, gradcam_features, threshold)

    return results

def main():
    csv_path = 'combined_token_results.csv'

    # Load data for both experiments
    df_ml, gradcam_features_ml, attention_features, loss_features = load_and_preprocess_data_ml(csv_path)
    df_gradcam, gradcam_features_gradcam, loss_features_gradcam = load_and_preprocess_data_gradcam(csv_path)

    # Define threshold range
    thresholds = np.arange(0.0, 0.175, 0.005)  # 0.0 to 0.17 with step 0.005

    print(f"Running experiments for thresholds: {thresholds}")

    # Store results
    ml_results = {}
    gradcam_results = {}

    # Run experiments for each threshold
    for threshold in thresholds:
        print(f"Processing threshold: {threshold:3f}")

        # ML experiments
        ml_exp_results = run_ml_experiments_for_threshold(df_ml, gradcam_features_ml, attention_features, loss_features, threshold)
        if ml_exp_results:
            for exp_name, result in ml_exp_results.items():
                if exp_name not in ml_results:
                    ml_results[exp_name] = {'thresholds': [], 'balanced_accuracies': []}
                ml_results[exp_name]['thresholds'].append(threshold)
                ml_results[exp_name]['balanced_accuracies'].append(result['balanced_accuracy'])

        # GradCAM experiments
        gradcam_exp_results = run_gradcam_experiments_for_threshold(df_gradcam, gradcam_features_gradcam, loss_features_gradcam, threshold)
        if gradcam_exp_results:
            for exp_name, result in gradcam_exp_results.items():
                # Remove "Single: " and "Pair: " prefixes from experiment names
                clean_exp_name = exp_name.replace('Single: ', '').replace('Pair: ', '')
                if clean_exp_name not in gradcam_results:
                    gradcam_results[clean_exp_name] = {'thresholds': [], 'balanced_accuracies': []}
                gradcam_results[clean_exp_name]['thresholds'].append(threshold)
                gradcam_results[clean_exp_name]['balanced_accuracies'].append(result['balanced_accuracy'])

    # Create plots
    plt.style.use('default')
    sns.set_style("whitegrid")

    # Plot 1: ML Experiment Results
    plt.figure(figsize=(10, 6))
    colors_ml = sns.color_palette("colorblind", n_colors=len(ml_results))

    for i, (exp_name, data) in enumerate(ml_results.items()):
        plt.plot(data['thresholds'], data['balanced_accuracies'],
                linewidth=1, label=exp_name, color=colors_ml[i])

    plt.xlabel('CER Threshold', fontsize=12)
    plt.ylabel('Balanced Accuracy', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add fine grid with 0.01 spacing
    plt.grid(True, alpha=0.3, which='major')
    plt.grid(True, alpha=0.1, which='minor', linewidth=0.5)
    plt.minorticks_on()
    
    # Set minor ticks for fine grid
    ax = plt.gca()
    ax.set_xticks(np.arange(0, 0.18, 0.01), minor=True)
    ax.set_yticks(np.arange(0.4, 0.71, 0.01), minor=True)
    
    # Set major y-axis ticks every 0.1 with thick labels
    ax.set_yticks(np.arange(0.4, 0.71, 0.1), minor=False)
    ax.tick_params(axis='y', which='major', width=2, length=6)
    
    # Rotate x-axis labels by 45 degrees
    plt.xticks(rotation=45)
    
    plt.ylim(0.4, 0.7)

    plt.tight_layout()
    plt.savefig('ml_experiment_threshold_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot 2: GradCAM Ablation Results
    plt.figure(figsize=(10, 6))
    colors_gradcam = sns.color_palette("colorblind", n_colors=len(gradcam_results))

    for i, (exp_name, data) in enumerate(gradcam_results.items()):
        plt.plot(data['thresholds'], data['balanced_accuracies'],
                linewidth=1, label=exp_name, color=colors_gradcam[i])

    plt.xlabel('CER Threshold', fontsize=12)
    plt.ylabel('Balanced Accuracy', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add fine grid with 0.01 spacing
    plt.grid(True, alpha=0.3, which='major')
    plt.grid(True, alpha=0.1, which='minor', linewidth=0.5)
    plt.minorticks_on()
    
    # Set minor ticks for fine grid
    ax = plt.gca()
    ax.set_xticks(np.arange(0, 0.18, 0.01), minor=True)
    ax.set_yticks(np.arange(0.4, 0.71, 0.01), minor=True)
    
    # Set major y-axis ticks every 0.1 with thick labels
    ax.set_yticks(np.arange(0.4, 0.71, 0.1), minor=False)
    ax.tick_params(axis='y', which='major', width=2, length=6)
    
    # Rotate x-axis labels by 45 degrees
    plt.xticks(rotation=45)
    
    plt.ylim(0.4, 0.7)

    plt.tight_layout()
    plt.savefig('gradcam_ablation_threshold_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print summary statistics
    print(f"\n{'='*80}")
    print("THRESHOLD ANALYSIS SUMMARY")
    print(f"{'='*80}")

    print("\nML Experiment - Best Balanced Accuracy by Experiment:")
    for exp_name, data in ml_results.items():
        max_balanced_acc = max(data['balanced_accuracies'])
        best_threshold = data['thresholds'][data['balanced_accuracies'].index(max_balanced_acc)]
        print(f"  {exp_name:<25}: {max_balanced_acc:.4f} at threshold {best_threshold:.2f}")

    print("\nGradCAM Ablation - Best Balanced Accuracy by Experiment:")
    for exp_name, data in gradcam_results.items():
        max_balanced_acc = max(data['balanced_accuracies'])
        best_threshold = data['thresholds'][data['balanced_accuracies'].index(max_balanced_acc)]
        print(f"  {exp_name:<25}: {max_balanced_acc:.4f} at threshold {best_threshold:.2f}")

if __name__ == "__main__":
    main()
