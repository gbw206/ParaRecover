"""
Batch concurrent DAG executor with checkpoint resume support.
Reads preprocessed input, runs DAGExecutor for each row,
and saves checkpoint after each row.
"""
import pandas as pd
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.dag_executor_concurrent import process_single_row
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse


def main():
    parser = argparse.ArgumentParser(description="Execute DAG concurrently with checkpoint resume")
    parser.add_argument("-i", "--input", default="output/preprocessed_input.xlsx")
    parser.add_argument("-o", "--output", default="output/executor_output.xlsx")
    parser.add_argument("-c", "--concurrency", type=int, default=15)
    parser.add_argument("--batch", type=int, default=30, help="Rows per batch")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    df = pd.read_excel(args.input)
    total = len(df)

    checkpoint_file = args.output.replace(".xlsx", "_state.json")
    results = {}
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            results = json.load(f)
        print(f"Found checkpoint: {len(results)}/{total} rows already completed")

    remaining = [i for i in range(total) if str(i) not in results]
    print(f"Total rows: {total}, remaining: {len(remaining)}")

    for batch_start in range(0, len(remaining), args.batch):
        batch_ids = remaining[batch_start:batch_start + args.batch]
        print(f"\n--- Batch: rows {batch_ids[0]}-{batch_ids[-1]} ({len(batch_ids)} rows) ---")

        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(process_single_row, idx, df.iloc[idx]): idx for idx in batch_ids}
            with tqdm(total=len(batch_ids)) as pbar:
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        result = future.result()
                        results[str(idx)] = {
                            "execute_round": result.get("execute_round", -1),
                            "status": result.get("status", "error"),
                        }
                    except Exception as e:
                        results[str(idx)] = {"execute_round": -1, "status": "error", "error": str(e)}
                    with open(checkpoint_file, "w") as f:
                        json.dump(results, f)
                    pbar.update(1)

    output_df = pd.DataFrame({
        "idx": range(total),
        "execute_round": [results.get(str(i), {}).get("execute_round", -1) for i in range(total)],
    })
    output_df.to_excel(args.output, index=False)
    print(f"\nAll done! Results saved to {args.output}")

    rounds = output_df["execute_round"]
    print(f"  Distribution: error={(rounds==-1).sum()}, 1~19={((rounds>=1)&(rounds<20)).sum()}, 20={(rounds==20).sum()}")


if __name__ == "__main__":
    main()
