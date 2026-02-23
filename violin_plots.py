import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate violin plots for entropy, gini, and losses."
    )
    parser.add_argument(
        "--input",
        default="./results/combined_token_results.csv",
        help="Path to combined token results CSV.",
    )
    parser.add_argument(
        "--output",
        default="./results/plots/violin_metrics_losses.png",
        help="Output path for the generated figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    df = pd.read_csv(input_path)

    required_columns = [
        "gradcam_entropy",
        "gradcam_gini",
        "attention_entropy",
        "attention_gini",
        "token_loss",
        "sample_id",
        "image_cer",
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Missing required columns: {missing_str}")

    line_cer = (
        df[["sample_id", "image_cer"]].dropna().drop_duplicates(subset=["sample_id"])
    )

    plot_specs = [
        ("gradcam_entropy", "GradCAM Entropy", df["gradcam_entropy"]),
        ("gradcam_gini", "GradCAM Gini", df["gradcam_gini"]),
        ("attention_entropy", "Attention Entropy", df["attention_entropy"]),
        ("attention_gini", "Attention Gini", df["attention_gini"]),
        ("token_loss", "Token Loss", df["token_loss"]),
        ("line_wise_cer", "Line-wise CER", line_cer["image_cer"]),
    ]

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for axis, (_, title, values) in zip(axes, plot_specs):
        clean_values = values.dropna()
        sns.violinplot(y=clean_values, ax=axis, inner="quartile", color="#4C72B0")
        axis.set_title(title)
        axis.set_xlabel("")
        axis.set_ylabel("Value")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()
