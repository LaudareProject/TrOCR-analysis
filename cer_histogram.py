#!/usr/bin/env python3
"""
CER Distribution Analysis: Histogram of Character Error Rate values for tokens
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import urllib.parse
import warnings
warnings.filterwarnings('ignore')

def load_token_data(csv_path):
    """Load CSV data and return token-level CER values"""
    print("Loading token-level data...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} token-level records")

    # Decode URL-encoded column names if needed
    df.columns = [urllib.parse.unquote_plus(col) for col in df.columns]
    print(f"Columns: {list(df.columns)}")

    return df

def create_cer_histogram(df):
    """Create histogram of CER values"""
    # Check if we have image_cer column
    if 'image_cer' not in df.columns:
        print("Error: 'image_cer' column not found in data")
        return

    # Get CER values
    cer_values = df['image_cer'].values

    # Remove any NaN values
    cer_values = cer_values[~np.isnan(cer_values)]

    print(f"CER Statistics:")
    print(f"  Count: {len(cer_values)}")
    print(f"  Min: {np.min(cer_values):.4f}")
    print(f"  Max: {np.max(cer_values):.4f}")
    print(f"  Mean: {np.mean(cer_values):.4f}")
    print(f"  Median: {np.median(cer_values):.4f}")
    print(f"  Std: {np.std(cer_values):.4f}")

    # Count perfect samples (CER = 0)
    perfect_count = np.sum(cer_values == 0)
    print(f"  Perfect samples (CER = 0): {perfect_count} ({perfect_count/len(cer_values)*100:.1f}%)")

    # Create the plot
    plt.style.use('default')
    sns.set_style("whitegrid")

    plt.figure(figsize=(12, 6))

    # Create histogram with colorblind-friendly color
    colors = sns.color_palette("colorblind")

    # Create bins with width 0.01
    bin_width = 0.02
    bins = np.arange(0, np.max(cer_values) + bin_width, bin_width)
    plt.hist(cer_values, bins=bins, alpha=0.7, color=colors[0], edgecolor='black', linewidth=0.5)

    # Add vertical lines for key statistics
    plt.axvline(np.mean(cer_values), color=colors[1], linestyle='--', linewidth=2, label=f'Mean = {np.mean(cer_values):.3f}')
    plt.axvline(np.median(cer_values), color=colors[2], linestyle='-.', linewidth=2, label=f'Median = {np.median(cer_values):.3f}')

    # # Add vertical lines for threshold examples
    # thresholds = [0.02, 0.04, 0.06, 0.08, 0.10]
    # for i, thresh in enumerate(thresholds):
    #     if thresh <= np.max(cer_values):
    #         count_below = np.sum(cer_values <= thresh)
    #         plt.axvline(thresh, color=colors[3], linestyle=':', alpha=0.6, linewidth=1)
    #         plt.text(thresh, plt.ylim()[1] * 0.8, f'{thresh:.2f}\n({count_below/len(cer_values)*100:.1f}%)',
    #                 rotation=90, ha='right', va='top', fontsize=8, alpha=0.7)

    plt.xlabel('Character Error Rate (CER)', fontsize=12)
    plt.ylabel('Number of Tokens', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Set x-axis ticks every 0.01 and rotate labels
    max_cer = np.max(cer_values)
    x_ticks = np.arange(0, max_cer + 0.02, 0.02)
    plt.xticks(x_ticks, rotation=45)

    # Add text box with statistics
    stats_text = f'Total tokens: {len(cer_values)}\nPerfect (CER=0): {perfect_count} ({perfect_count/len(cer_values)*100:.1f}%)'
    plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=10)

    plt.tight_layout()
    plt.savefig('cer_distribution_histogram.png', dpi=300, bbox_inches='tight')
    plt.show()

    return cer_values

def create_sample_level_cer_histogram(df):
    """Create histogram of sample-level (aggregated) CER values"""
    print("\nCreating sample-level CER histogram...")

    # Aggregate by sample_id to get one CER per sample
    sample_cer = df.groupby('sample_id')['image_cer'].first().values

    print(f"Sample-level CER Statistics:")
    print(f"  Count: {len(sample_cer)}")
    print(f"  Min: {np.min(sample_cer):.4f}")
    print(f"  Max: {np.max(sample_cer):.4f}")
    print(f"  Mean: {np.mean(sample_cer):.4f}")
    print(f"  Median: {np.median(sample_cer):.4f}")
    print(f"  Std: {np.std(sample_cer):.4f}")

    # Count perfect samples (CER = 0)
    perfect_count = np.sum(sample_cer == 0)
    print(f"  Perfect samples (CER = 0): {perfect_count} ({perfect_count/len(sample_cer)*100:.1f}%)")

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Create histogram with colorblind-friendly color
    colors = sns.color_palette("colorblind")

    # Create bins with width 0.01
    bin_width = 0.02
    bins = np.arange(0, np.max(sample_cer) + bin_width, bin_width)
    plt.hist(sample_cer, bins=bins, alpha=0.7, color=colors[0], edgecolor='black', linewidth=0.5)

    # Add vertical lines for key statistics
    plt.axvline(np.mean(sample_cer), color=colors[1], linestyle='--', linewidth=2, label=f'Mean = {np.mean(sample_cer):.3f}')
    plt.axvline(np.median(sample_cer), color=colors[2], linestyle='-.', linewidth=2, label=f'Median = {np.median(sample_cer):.3f}')

    # # Add vertical lines for threshold examples
    # thresholds = [0.02, 0.04, 0.06, 0.08, 0.10]
    # for i, thresh in enumerate(thresholds):
    #     if thresh <= np.max(sample_cer):
    #         count_below = np.sum(sample_cer <= thresh)
    #         plt.axvline(thresh, color=colors[3], linestyle=':', alpha=0.6, linewidth=1)
    #         plt.text(thresh, plt.ylim()[1] * 0.8, f'{thresh:.2f}\n({count_below/len(sample_cer)*100:.1f}%)',
    #                 rotation=90, ha='right', va='top', fontsize=8, alpha=0.7)

    plt.xlabel('Character Error Rate (CER)', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Set x-axis ticks every 0.01 and rotate labels
    max_cer = np.max(sample_cer)
    x_ticks = np.arange(0, max_cer + 0.02, 0.02)
    plt.xticks(x_ticks, rotation=45)

    # Add text box with statistics
    stats_text = f'Total samples: {len(sample_cer)}\nPerfect (CER=0): {perfect_count} ({perfect_count/len(sample_cer)*100:.1f}%)'
    plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=10)

    plt.tight_layout()
    plt.savefig('sample_cer_distribution_histogram.png', dpi=300, bbox_inches='tight')
    plt.show()

    return sample_cer

def analyze_threshold_impact(cer_values):
    """Analyze how different thresholds would affect class distribution"""
    print(f"\n{'='*60}")
    print("THRESHOLD IMPACT ANALYSIS")
    print(f"{'='*60}")

    thresholds = np.arange(0.0, 0.22, 0.02)

    print(f"{'Threshold':<10} {'Good Samples':<15} {'Bad Samples':<15} {'Good %':<10} {'Bad %':<10}")
    print("-" * 65)

    for threshold in thresholds:
        good_count = np.sum(cer_values <= threshold)
        bad_count = len(cer_values) - good_count
        good_pct = good_count / len(cer_values) * 100
        bad_pct = bad_count / len(cer_values) * 100

        print(f"{threshold:<10.2f} {good_count:<15d} {bad_count:<15d} {good_pct:<10.1f} {bad_pct:<10.1f}")

def main():
    csv_path = 'combined_token_results.csv'

    # Load token-level data
    df = load_token_data(csv_path)

    # Create token-level histogram
    print(f"\n{'='*60}")
    print("TOKEN-LEVEL CER DISTRIBUTION")
    print(f"{'='*60}")
    token_cer = create_cer_histogram(df)

    # Create sample-level histogram
    print(f"\n{'='*60}")
    print("SAMPLE-LEVEL CER DISTRIBUTION")
    print(f"{'='*60}")
    sample_cer = create_sample_level_cer_histogram(df)

    # Analyze threshold impact (using sample-level data)
    analyze_threshold_impact(sample_cer)

if __name__ == "__main__":
    main()
