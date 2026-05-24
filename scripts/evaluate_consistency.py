"""
Compute human-model consistency metrics:
- Exact Match Accuracy
- Weighted Kappa
- MAE
- Confusion Matrix
- Error Distribution
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr, pearsonr

DIMENSION_GROUPS = {
    "diag": ["D1", "D2", "D3", "D4", "D5"],
    "evo": ["E1", "E2", "E3", "E4"],
    "struct": ["S1", "S2", "S3", "S4"],
}
LABEL_ORDER = [0.0, 0.5, 1.0]


def safe_weighted_kappa(y_true, y_pred, weights="linear"):
    if len(y_true) == 0:
        return np.nan
    y_true = [int(x) for x in y_true]
    y_pred = [int(x) for x in y_pred]
    unique_values = sorted(set(y_true) | set(y_pred))
    kappa = cohen_kappa_score(y_true, y_pred, labels=unique_values, weights=weights)
    if np.isnan(kappa):
        if np.array_equal(np.array(y_true), np.array(y_pred)):
            return 1.0
    return kappa


def safe_corr(x, y, method="spearman"):
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    if method == "spearman":
        return spearmanr(x, y).statistic
    elif method == "pearson":
        return pearsonr(x, y).statistic
    return np.nan


def main():
    parser = argparse.ArgumentParser(description="Compute human-model consistency metrics")
    parser.add_argument("--input_file", default=None)
    parser.add_argument("--output_file", default="output/consistency_results.xlsx")
    parser.add_argument("--human_suffix", default="_human", help="Human score column suffix")
    parser.add_argument("--model_suffix", default="_model", help="Model score column suffix")
    args = parser.parse_args()

    input_path = Path(args.input_file or "output/evaluation.xlsx")
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_excel(input_path)

    rows = []
    for group, dims in DIMENSION_GROUPS.items():
        for dim in dims:
            human_col = f"{dim}{args.human_suffix}" if args.human_suffix else dim
            model_col = f"{dim}{args.model_suffix}" if args.model_suffix else dim

            if human_col in df.columns and model_col in df.columns:
                sub = df[[human_col, model_col]].dropna()
                y_true = sub[human_col].astype(float)
                y_pred = sub[model_col].astype(float)

                rows.append({
                    "Group": group,
                    "Dimension": dim,
                    "N": len(sub),
                    "Exact Match Acc": np.mean(y_true == y_pred),
                    "Weighted Kappa": safe_weighted_kappa(y_true, y_pred),
                    "MAE": mean_absolute_error(y_true, y_pred),
                    "Human Mean": np.mean(y_true),
                    "Model Mean": np.mean(y_pred),
                })

    if not rows:
        print("No matching columns found. Please check column name suffixes.")
        return

    result_df = pd.DataFrame(rows)
    result_df.to_excel(args.output_file, index=False)
    print(f"Results saved to: {args.output_file}")
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
