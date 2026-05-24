"""
Multi-dimension LLM-as-a-Judge evaluation for replanner outputs.
Evaluates across 3 dimensions: Diagnostic (D1-D5), Evolutionary (E1-E5), Structural (S1-S5).
"""
import argparse
import concurrent.futures
import json
import logging
import re
import textwrap
import time
from dataclasses import dataclass
from functools import wraps
from threading import Lock
from typing import Callable, Optional

import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
import os

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SCORE_DIMS = ["S1", "S2", "S3", "S4", "S5", "D1", "D2", "D3", "D4", "D5", "E1", "E2", "E3", "E4", "E5"]

STRICT_COMMON_HEADER = """
## Role
You are a very strict, conservative, and fine-grained replanner decision evaluation expert.

## General Requirements
You need to evaluate whether the given replanner's thought and replan DAG are truly high quality.
You must default to strict scoring, not default to medium or high scores.

## Input
[Current User Original Task]
{query}

[Available Tool List]
{tool_list}

[Previous Plan]
{prev_plan}

[Previous Plan Execution Results / Tool Returns]
{tool_result}

[Replanner Output]
[Thought]
{thought}

[Updated Plan]
{new_plan}
"""


def build_diagnostic_prompt(item: dict) -> str:
    return textwrap.dedent(
        STRICT_COMMON_HEADER
        + """
## Evaluation Dimension: diagnostic
Please strictly evaluate the following 5 sub-items from the "Diagnostic Quality" perspective:

- D1 Plan Evidence Anchoring: Does the thought clearly reference and correctly understand specific nodes, dependencies, statuses, or steps from the previous plan?
- D2 Tool Evidence Anchoring: Does the thought clearly reference and correctly understand specific errors, return values, parameter names, or tool behaviors from the tool_result?
- D3 Root Cause Localization Precision: Does it locate the true root cause rather than just describing surface symptoms?
- D4 Impact Scope Identification: Does it identify downstream nodes, dependency chains, status progressions, or response generation affected by the error?
- D5 Evidence Sufficiency and Restraint: Does it provide sufficient evidence without fabricating information not present in the input?

## Output Format
Output only a JSON object:
```json
{{"D1_score": "1/0.5/0", "D1_reason": "...", "D2_score": "1/0.5/0", "D2_reason": "...", "D3_score": "1/0.5/0", "D3_reason": "...", "D4_score": "1/0.5/0", "D4_reason": "...", "D5_score": "1/0.5/0", "D5_reason": "...", "dimension_score": "1/0.5/0", "dimension_summary": "..."}}
```
"""
    ).strip().format(
        query=item["query"], tool_list=item["tool_list"],
        prev_plan=item["prev_plan"], tool_result=item["tool_result"],
        thought=item["thought"], new_plan=item["new_plan"],
    )


def build_evolutionary_prompt(item: dict) -> str:
    return textwrap.dedent(
        STRICT_COMMON_HEADER
        + """
## Evaluation Dimension: evolutionary
Please strictly evaluate the following 5 sub-items from the "Fix and Improvement Strategy Quality" perspective:

- E1 Change Minimality: Does the replan only modify necessary parts, without irrelevant rewrites?
- E2 Fix Closure: Does it not only fix the current error point but also address related downstream nodes, parameters, statuses, or response chains?
- E3 Goal Preservation: Does the new plan remain faithfully aligned with the original user task, without deviation, reduction, or missing key objectives?
- E4 Strategy Necessity: Do added steps, deleted steps, or reordered dependencies all have clear justification?
- E5 Problem-Solving Effectiveness: If executed, is the replan likely to actually resolve the core issues identified?

## Output Format
Output only a JSON object:
```json
{{"E1_score": "1/0.5/0", "E1_reason": "...", "E2_score": "1/0.5/0", "E2_reason": "...", "E3_score": "1/0.5/0", "E3_reason": "...", "E4_score": "1/0.5/0", "E4_reason": "...", "E5_score": "1/0.5/0", "E5_reason": "...", "dimension_score": "1/0.5/0", "dimension_summary": "..."}}
```
"""
    ).strip().format(
        query=item["query"], tool_list=item["tool_list"],
        prev_plan=item["prev_plan"], tool_result=item["tool_result"],
        thought=item["thought"], new_plan=item["new_plan"],
    )


def build_structural_prompt(item: dict) -> str:
    return textwrap.dedent(
        STRICT_COMMON_HEADER
        + """
## Evaluation Dimension: structural
Please strictly evaluate the following 4 sub-items judged by LLM (S5 is automatically filled by the program):

- S1 thought-DAG Consistency: Are the actions, fixes, and progressions claimed in the thought actually reflected in the replan DAG?
- S2 Topological Dependency and Overall Executability: Are node dependencies, ordering, and upstream/downstream relationships correct?
- S3 Tool Call Legality: Do tool names, parameter names, and parameter value formats strictly conform to the tool list constraints?
- S4 Status Progression Correctness: Are the statuses of executed nodes, unexecuted nodes, failure-retained nodes, and response node reasonable?

## Output Format
Output only a JSON object:
```json
{{"S1_score": "1/0.5/0", "S1_reason": "...", "S2_score": "1/0.5/0", "S2_reason": "...", "S3_score": "1/0.5/0", "S3_reason": "...", "S4_score": "1/0.5/0", "S4_reason": "...", "S5_score": "Auto-filled by program", "S5_reason": "Auto-filled by program", "dimension_score": "1/0.5/0", "dimension_summary": "..."}}
```
"""
    ).strip().format(
        query=item["query"], tool_list=item["tool_list"],
        prev_plan=item["prev_plan"], tool_result=item["tool_result"],
        thought=item["thought"], new_plan=item["new_plan"],
    )


@dataclass(frozen=True)
class DimensionSpec:
    name: str
    prefix: str
    prompt_builder: Callable


DIMENSIONS = {
    "diagnostic": DimensionSpec(name="diagnostic", prefix="diag", prompt_builder=build_diagnostic_prompt),
    "evolutionary": DimensionSpec(name="evolutionary", prefix="evo", prompt_builder=build_evolutionary_prompt),
    "structural": DimensionSpec(name="structural", prefix="struct", prompt_builder=build_structural_prompt),
}


def score_structural_script(replan) -> tuple:
    if isinstance(replan, dict):
        dag = replan
    elif isinstance(replan, str):
        try:
            dag = json.loads(replan)
        except Exception:
            return "0", "Replan is not parseable into a DAG."
    else:
        return "0", "Replan is not parseable."

    tasks = dag.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        return "0", "Empty or invalid tasks list."

    id_set = set()
    adjacency = {}
    indegree = {}

    for task in tasks:
        if not isinstance(task, dict):
            return "0", "Task node is not a JSON object."
        node_id = task.get("id")
        if node_id is None or str(node_id).strip() == "":
            return "0", "Missing valid id."
        node_id = str(node_id)
        if node_id in id_set:
            return "0", f"Duplicate node id: {node_id}."
        id_set.add(node_id)
        adjacency[node_id] = []
        indegree[node_id] = 0

    for task in tasks:
        node_id = str(task["id"])
        deps = task.get("dep", []) or []
        if not isinstance(deps, list):
            return "0", f"Node {node_id} has non-list dep."
        for dep in deps:
            dep_id = str(dep)
            if dep_id not in id_set:
                return "0", f"Node {node_id} depends on missing node {dep_id}."
            adjacency[dep_id].append(node_id)
            indegree[node_id] += 1

    queue = [nid for nid in id_set if indegree[nid] == 0]
    visited = 0
    cursor = 0
    while cursor < len(queue):
        current = queue[cursor]
        cursor += 1
        visited += 1
        for nxt in adjacency[current]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)

    if visited != len(id_set):
        return "0", "DAG contains a cycle."
    return "1", "Valid acyclic DAG."


def extract_json_object(text: str) -> dict:
    if not isinstance(text, str):
        return {}
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return {}
    try:
        return json.loads(match.group())
    except Exception:
        return {}


def call_model(client, model, messages):
    response = client.chat.completions.create(model=model, messages=messages, stream=False)
    return response.choices[0].message.content or ""


def retry(max_retries=10, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            last_exception = None
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_exception = exc
                    retries += 1
                    logger.warning("retrying (%s/%s): %s", retries, max_retries, str(exc))
                    time.sleep(delay * retries)
            raise Exception(f"exceeded max retries {max_retries}") from last_exception
        return wrapper
    return decorator


class QPSLimitedExecutor:
    def __init__(self, api_key, base_url, model, qps=20, max_workers=100):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.qps = qps
        self.max_workers = max_workers
        self.last_request_time = 0.0
        self.lock = Lock()

    @retry(max_retries=10, delay=1)
    def run(self, params):
        messages = params["request_body"].get("messages")
        if not messages:
            return ""
        logger.info("id %s start", params["id"])
        result = call_model(self.client, self.model, messages)
        logger.info("id %s done", params["id"])
        return result

    def qps_limited_executor(self, params):
        with self.lock:
            elapsed = time.time() - self.last_request_time
            wait_time = max(0, 1.0 / self.qps - elapsed)
            if wait_time > 0:
                time.sleep(wait_time)
            self.last_request_time = time.time()
        return self.run(params)

    def start_concurrent_executing(self, params_list=None):
        params_list = params_list or []
        results = []
        with tqdm(total=len(params_list), desc="processing") as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(self.qps_limited_executor, p): p for p in params_list}
                for future in concurrent.futures.as_completed(futures):
                    params = futures[future]
                    try:
                        params["output"] = future.result()
                    except Exception as exc:
                        params["output"] = ""
                        params["error"] = str(exc)
                    results.append(params)
                    pbar.update(1)
        return results


def build_requests(rows, dimensions, model_prefix=""):
    requests = []
    for row in rows:
        item = {
            "query": row.get("query", ""),
            "tool_list": row.get("tool_list", ""),
            "prev_plan": row.get("plan", ""),
            "tool_result": row.get("fun_call", ""),
            "thought": row.get("thought", ""),
            "new_plan": row.get("replan", ""),
        }
        sample_id = str(row.get("id", ""))
        for dim_name, spec in dimensions.items():
            prompt = spec.prompt_builder(item)
            requests.append({
                "id": f"{sample_id}__{dim_name}",
                "sample_id": sample_id,
                "dimension": dim_name,
                "origin": row,
                "request_body": {
                    "messages": [{"role": "system", "content": ""}, {"role": "user", "content": prompt}]
                },
            })
    return requests


def parse_dimension_output(text, prefix):
    parsed = {}
    data = extract_json_object(text)
    for dim in ["D1", "D2", "D3", "D4", "D5", "E1", "E2", "E3", "E4", "E5", "S1", "S2", "S3", "S4", "S5"]:
        parsed[f"{prefix}_{dim}_score"] = data.get(f"{dim}_score", "")
        parsed[f"{prefix}_{dim}_reason"] = data.get(f"{dim}_reason", "")
    parsed[f"{prefix}_dimension_score"] = data.get("dimension_score", "")
    parsed[f"{prefix}_dimension_summary"] = data.get("dimension_summary", "")
    return parsed


def main():
    parser = argparse.ArgumentParser(description="Multi-dimension LLM-as-a-Judge evaluation")
    parser.add_argument("-f", "--input_file", default=None)
    parser.add_argument("-s", "--save_file", default="output/evaluation.xlsx")
    parser.add_argument("--limit", default=None, type=int)
    parser.add_argument("--model", default=None)
    parser.add_argument("--qps", default=10, type=int)
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
    model = args.model or os.getenv("OPENAI_MODEL", "deepseek-chat")

    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")

    os.makedirs("output", exist_ok=True)
    input_file = args.input_file or os.path.join("data", "LEVEL-1", "LEVEL-1_augmented.jsonl")

    logger.info(f"loading: {input_file}")

    if input_file.endswith(".jsonl"):
        rows = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    else:
        df = pd.read_excel(input_file)
        rows = df.to_dict('records')

    if args.limit:
        rows = rows[:args.limit]

    logger.info(f"loaded {len(rows)} rows")
    requests = build_requests(rows, DIMENSIONS)

    executor = QPSLimitedExecutor(api_key=api_key, base_url=base_url, model=model, qps=args.qps)
    results = executor.start_concurrent_executing(params_list=requests)

    grouped = {}
    for result in results:
        sid = result["sample_id"]
        if sid not in grouped:
            grouped[sid] = {"id": sid}
        spec = DIMENSIONS[result["dimension"]]
        output = result.get("output", "")
        grouped[sid][f"{spec.prefix}_raw_output"] = output
        grouped[sid][f"{spec.prefix}_error"] = result.get("error", "")
        grouped[sid].update(parse_dimension_output(output, spec.prefix))

    output_rows = []
    for row in rows:
        sid = str(row.get("id", ""))
        merged = {"id": sid}
        merged.update(row)
        if sid in grouped:
            merged.update(grouped[sid])
        output_rows.append(merged)

    output_df = pd.DataFrame(output_rows)
    output_df.to_excel(args.save_file, index=False)
    logger.info(f"saved to {args.save_file}")


if __name__ == "__main__":
    main()
