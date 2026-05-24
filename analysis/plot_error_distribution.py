import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path
import argparse


def parse_tags(cell):
    if pd.isna(cell):
        return []
    text = str(cell).strip()
    if not text:
        return []
    text = text.replace("，", ",").replace("、", ",").replace("；", ",").replace(";", ",")
    tags = [x.strip() for x in text.split(",")]
    return [x for x in tags if x]


def main():
    parser = argparse.ArgumentParser(description="Plot error tag distribution from xlsx")
    parser.add_argument("-i", "--input", default=None)
    parser.add_argument("-o", "--output", default="error_distribution.png")
    parser.add_argument("--col", default="error", help="Error label column name")
    args = parser.parse_args()

    df = pd.read_excel(args.input)

    error_col = args.col
    if error_col not in df.columns:
        raise ValueError(f"Column '{error_col}' not found. Available: {list(df.columns)}")

    counter = Counter()
    for cell in df[error_col]:
        counter.update(parse_tags(cell))

    if not counter:
        raise ValueError("No tags found in error column")

    stat_df = pd.DataFrame(counter.items(), columns=["tag", "count"])
    stat_df = stat_df.sort_values("count", ascending=False).reset_index(drop=True)

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False

    n = len(stat_df)
    fig_width = max(8, min(0.6 * n + 4, 20))
    fig, ax = plt.subplots(figsize=(fig_width, 6), dpi=300)

    bars = ax.bar(stat_df["tag"], stat_df["count"], edgecolor="black", linewidth=0.8)

    ax.set_xlabel("Error Tag", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distribution of Error Tags", fontsize=13, pad=12)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)

    plt.xticks(rotation=35, ha="right", fontsize=10)

    for bar, value in zip(bars, stat_df["count"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                str(value), ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(args.output, bbox_inches="tight", dpi=300)
    plt.close()

    output_excel = Path(args.output).with_suffix(".xlsx")
    stat_df.to_excel(output_excel, index=False)

    print(f"Figure saved to: {args.output}")
    print(f"Stats saved to: {output_excel}")
    print(stat_df.to_string(index=False))


if __name__ == "__main__":
    main()
