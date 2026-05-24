"""
Run DAGExecutor row by row with checkpoint.
Resumes from where it left off if interrupted.
"""
import pandas as pd
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.dag_executor_concurrent import process_single_row
import argparse


def main():
    parser = argparse.ArgumentParser(description="Run DAG executor row by row with checkpoint resume")
    parser.add_argument("-i", "--input", default="output/preprocessed_input.xlsx")
    parser.add_argument("-o", "--output", default="output/executor_output.xlsx")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    df = pd.read_excel(args.input)
    total = len(df)

    checkpoint_file = args.output.replace(".xlsx", "_state.json")
    os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)

    results = {}
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            results = json.load(f)
        print(f"Found checkpoint, {len(results)}/{total} rows already processed")

    for idx in range(total):
        sidx = str(idx)
        if sidx in results:
            continue

        try:
            result = process_single_row(idx, df.iloc[idx])
            results[sidx] = {
                "execute_round": result.get("execute_round", -1),
                "status": result.get("status", "error"),
            }
        except Exception as e:
            results[sidx] = {"execute_round": -1, "status": "error", "error": str(e)}

        with open(checkpoint_file, "w") as f:
            json.dump(results, f)
        print(f"  Row {idx}/{total}: execute_round={results[sidx]['execute_round']}")

    output_df = pd.DataFrame({
        "idx": range(total),
        "execute_round": [results.get(str(i), {}).get("execute_round", -1) for i in range(total)],
    })
    output_df.to_excel(args.output, index=False)
    print(f"\nAll done! Results saved to {args.output}")

    rounds = output_df["execute_round"]
    print(f"  Distribution: error={(rounds==-1).sum()}, 0={(rounds==0).sum()}, "
          f"1~19={((rounds>=1)&(rounds<20)).sum()}, 20={(rounds==20).sum()}")


if __name__ == "__main__":
    main()
