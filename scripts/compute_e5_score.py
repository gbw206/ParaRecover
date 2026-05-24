"""
Compute E5 (Evolutionary Efficiency) scores.
Pipeline: preprocess input -> run DAG executor -> compute E5 scores.
"""
import pandas as pd
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.dag_executor_concurrent import process_single_row
import argparse

MAX_ROUNDS = 40


def count_pre_rounds(plan_dag):
    tasks = plan_dag["tasks"]
    task_map = {t["id"]: t for t in tasks}
    done = {"root"}
    rounds = 0
    true_ids = {t["id"] for t in tasks if t.get("status") is True and t["id"] != "root"}

    while true_ids:
        executable = set()
        for tid in true_ids:
            deps = task_map[tid].get("dep", [])
            if not deps or deps == ["root"]:
                executable.add(tid)
            elif all(d in done for d in deps):
                executable.add(tid)
        if not executable:
            break
        done.update(executable)
        true_ids -= executable
        rounds += 1
    return rounds


def main():
    parser = argparse.ArgumentParser(description="Compute E5 efficiency score")
    parser.add_argument("--input", default=None, help="Input xlsx file path (with replan column)")
    parser.add_argument("--output", default="output/e5_scores.xlsx", help="Output xlsx file path")
    parser.add_argument("--test_n", type=int, default=9999, help="Number of test rows")
    parser.add_argument("--skip_executor", action="store_true", help="Skip executor step, compute directly")
    parser.add_argument("--executor_output", default="output/executor_output.xlsx", help="Executor output file path")
    args = parser.parse_args()

    if not args.input and not args.skip_executor:
        print("Please specify --input or use --skip_executor")
        return

    os.makedirs("output", exist_ok=True)

    preprocessed_file = "output/preprocessed_input.xlsx"
    executor_output_file = args.executor_output

    if not args.skip_executor:
        df = pd.read_excel(args.input)
        df = df.head(args.test_n).copy()

        nan_mask = df["replan"].isna()
        total = len(df)
        nan_count = nan_mask.sum()
        print(f"Total rows: {total}, replan is NaN: {nan_count}")

        df_pre = pd.DataFrame()
        df_pre["id"] = df["id"]
        df_pre["tool_list"] = df["tool_list"]
        df_pre["query"] = df["query"]
        df_pre["plan_thought"] = ""
        df_pre["plan_dag"] = df["replan"]
        df_pre["round_gt"] = df.get("round_gt", 1)
        df_pre.to_excel(preprocessed_file, index=False)
        print(f"Preprocessing done, saved to {preprocessed_file}")

        print("\nPlease run run_dag_executor.py to generate executor results, or use --skip_executor to use existing results")
        return

    pre_df = pd.read_excel(preprocessed_file)
    exec_df = pd.read_excel(executor_output_file)

    n = min(len(pre_df), len(exec_df))
    scores = []

    for i in range(n):
        round_gt = int(pre_df["round_gt"].iloc[i])
        plan_dag_str = pre_df["plan_dag"].iloc[i]
        execute_round = exec_df["execute_round"].iloc[i]

        if pd.isna(plan_dag_str) or not isinstance(plan_dag_str, str):
            score = 0.0
        elif pd.isna(execute_round) or execute_round == -1 or execute_round >= MAX_ROUNDS:
            score = 0.0
        else:
            plan_dag = json.loads(plan_dag_str)
            pre_rounds = count_pre_rounds(plan_dag)
            total_rounds = pre_rounds + int(execute_round)
            score = min(1.0, round_gt / total_rounds) if total_rounds > 0 else 1.0

        scores.append(score)

    final = pd.read_excel(args.input).head(args.test_n).copy()
    final["evo_E5_score_sim"] = scores

    final.to_excel(args.output, index=False)

    valid_scores = [s for s in scores if s > 0.0]
    print(f"\nResults saved -> {args.output}")
    print(f"Total rows: {len(scores)}, Valid scores (>0): {len(valid_scores)}")
    if valid_scores:
        print(f"Average: {sum(scores)/len(scores):.4f}  (valid only: {sum(valid_scores)/len(valid_scores):.4f})")


if __name__ == "__main__":
    main()
