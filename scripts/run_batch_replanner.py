"""
Batch call LLM to perform replanning on benchmark data.
Reads JSONL input, sends prompts to LLM, saves results to XLSX.
"""
import json
import re
import os
import time
import logging
import argparse
import textwrap
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
from functools import wraps
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def build_prompt(query: str, tool_list: str, plan_dag: str, tool_call_dag: str) -> str:
    PROMPT_TEMPLATE = textwrap.dedent("""
## Your task：
1. Analyze the user's original query, combine it with the DAG list of the previous plan and the results of the previous tool calls, and consider what needs to be done to complete the task.
2. Based on the tools that still need to be called, update the current complete task list to guide the entire subsequent process.

## Output requirements
- First, output the <thought> section. Include reflections on the original user task and perform a requirement analysis: first, identify which subtasks have already been completed toward answering the user's question and which subtasks still need to be completed. Then, provide a detailed sequence of steps required to accomplish the original task, along with a sound justification for the rationale behind each step. In your reasoning, include a detailed analysis of the reasonableness of the previous round's planning, as well as a detailed analysis of the reasonableness of the previous round's tool invocation arrangements (including the tools used and their parameters).
- Second, output a parseable JSON representing the task chain list of tool calls.
    1. Clearly specify the dependency relationships for each subtask. A dependency exists only if the return result of a preceding task is required to supplement or complete the parameters of a subsequent tool call. For any subtask that does not require preceding information, its dependency should be set to "root".
    2. For subtask nodes that have not yet been executed, you are free to modify, add, or delete them as needed to better invoke tools and complete the task. Note that subtask nodes which have not yet been executed must not be altered.
    3. The first subtask node has been determined as the "root" node, with the subtask set to:
{{"id": "root",  # Root node, representing the starting point of the task.
"desc": {query}, # Original query content provided by the user.
"dep": [],  # The root node does not need to depend on other nodes.
"status": True   # The root node has already been executed by default.}}
    The last subtask node has been determined as the "response" node, with the subtask set to:
{{"id": "response",  # The final node indicates the end point of the task.
"desc": "Analyze and summarize, and then respond.", # Summarize and analyze the results of all subtasks, and then respond.
"dep": [...],  # All nodes that are not dependent on by other subtasks.
"status": False    # The last response node has not been executed by default.}}
    4. In the task node you receive, "status" represents the task's execution status: True means it has been executed, and False means it has not been executed.
    5. If there are subtasks that need to be executed, output all subtask nodes, including the root and response.
    6. You need to analyze the [Previous Round Execution Results] to determine which subtasks have been completed and mark the status of completed subtasks as True.
    7. Based on the "Results of Previous Execution", you need to add the description of the tool's execution results (including failure results) into the corresponding node's desc. Do not change the original description; instead, add the content after it.
    8. For subtasks that return an Error or other failure signal from the tool, you need to mark the status of the subtask node as True and mark the relevant failure signal in the descriptive section. Then, design new subtask nodes to solve the problem, because the output plan needs to be referential and should include all success and failure signals.
    9. Because you cannot modify a tool call task node that has already been executed (regardless of the result), if you want to execute a subtask similar to some already executed node, you can only create a new task node.
    10. If all tasks except for the response node have been completed at this point, you need to output the complete task list, set the status of the response node to True, and provide the final task completion status in the desc field of the response node.
    11. You can only use the tools in the [Available Tools List]. You cannot create or fabricate tools on your own. When the tool returns an error, you need to reflect on whether there are any errors in your calling method, tool name, parameters, etc. If there are no errors, you can create a new node and try to retries up to three times.

##The output format is as follows (strictly output the reasoning section and the task list JSON. Do not include comments. Please output a compact JSON format, removing all unnecessary line breaks and indentation spaces, while preserving necessary spaces within string content):
<thought>Output your thoughts on the user's current query; you need to re-analyze and rethink it. Conduct a requirements analysis, first analyzing what you have already accomplished to complete the user's original query, and what tools are still needed. Then, detail the steps required to answer the user's query, and reasonably explain the rationale for each step. Your thinking should include a detailed analysis of the rationality of the previous planning, as well as the rationality of the tool usage arrangement (including tools used and parameters). Note that your thinking should not exceed 500 words.</thought>

{{"tasks": [  # A list of subtasks, where each element is a tool call subtask node, containing the task ID, the name of the called tool, the task description, and dependencies. All subtask nodes must be provided (regardless of whether they have been executed), including the root node and the response subtask node.
{{"id": string,  # A unique identifier for a subtask, which is identified by a number "n", starting from "1".
"name": string,  # The name of the tool called in the subtask.
"para": [string],  # The parameters used in the tool call are in the format {{"parameter1 name":"parameter1 value","parameter2 name":"parameter2 value", ......}}.
"desc": string, # Description of task nodes
"dep": [string],  # The subtask dependencies, where each element in the list is the id of the dependent subtask.
"status": bool  # This indicates whether the subtask has been executed. True means it has been executed, and False means it has not been executed.}}, ......]}}

##Original User Query
{query}

##Available Tools List
{tool_list}

##Front-Wheel Plan
{plan_dag}

##Front-Wheel Tool Call Results
{tool_call_dag}
    """).strip()
    return PROMPT_TEMPLATE.format(query=query, tool_list=tool_list, plan_dag=plan_dag, tool_call_dag=tool_call_dag)


THOUGHT_PATTERN = re.compile(r"<thought>(.*?)</thought>", re.S)


def extract_thought_and_plan(text: str):
    if not isinstance(text, str):
        return "", ""
    thought = ""
    plan_dag = ""
    m = THOUGHT_PATTERN.search(text)
    if m:
        thought = m.group(1).strip()
        rest = text[m.end():].strip()
    else:
        rest = text.strip()
    try:
        first_brace = rest.find("{")
        if first_brace != -1:
            plan_dag = json.loads(rest[first_brace:])
    except Exception:
        plan_dag = ""
    return thought, plan_dag


def create_client():
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable or create a .env file")
    return OpenAI(api_key=api_key, base_url=base_url)


def retry(max_retries=10, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            last_exception = None
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    retries += 1
                    logger.warning(f"Request failed, retrying ({retries}/{max_retries})... error: {str(e)}")
                    time.sleep(delay * retries)
            raise Exception(f"Exceeded max retries {max_retries}") from last_exception
        return wrapper
    return decorator


class QPSLimitedExecutor:
    def __init__(self, client, model, qps=20, max_workers=100):
        self.client = client
        self.model = model
        self.qps = qps
        self.max_workers = max_workers
        self.last_request_time = 0
        self.lock = Lock()

    @retry(max_retries=10, delay=1)
    def run(self, params):
        if 'id' not in params or 'request_body' not in params:
            return ''
        messages = params['request_body'].get('messages', [])
        if not messages:
            return ''
        logger.info(f"id {params['id']} start")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=False
        )
        result = response.choices[0].message.content
        logger.info(f"id {params['id']} done")
        return result

    def qps_limited_executor(self, params):
        with self.lock:
            elapsed = time.time() - self.last_request_time
            wait_time = max(0, 1.0 / self.qps - elapsed)
            if wait_time > 0:
                time.sleep(wait_time)
            self.last_request_time = time.time()
        return self.run(params)

    def start_concurrent_executing(self, params_list=None, total_jobs=None, save_interval=100, save_func=None):
        params_list = params_list or []
        max_jobs = len(params_list)
        if total_jobs is not None:
            max_jobs = min(total_jobs, max_jobs)
        params_list = params_list[:max_jobs]
        results = []
        completed_count = 0

        with tqdm(total=max_jobs, desc="Processing progress") as pbar:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {}
                for ind in range(max_jobs):
                    futures[executor.submit(self.qps_limited_executor, params_list[ind])] = params_list[ind]

                for future in as_completed(futures):
                    params = futures[future]
                    try:
                        result = future.result()
                        params['output'] = result
                        results.append(params)
                        completed_count += 1
                        pbar.update(1)
                        pbar.set_postfix_str(f"Latest: {params['id']}...")

                        if save_func and completed_count % save_interval == 0:
                            rows = []
                            for r in results:
                                origin = r.get("origin", {})
                                output = r.get("output", "")
                                row = {}
                                for k, v in origin.items():
                                    if isinstance(v, (dict, list)):
                                        row[k] = json.dumps(v, ensure_ascii=False)
                                    else:
                                        row[k] = v
                                content = output if isinstance(output, str) else ""
                                thought, plan_dag = extract_thought_and_plan(content)
                                row["thought"] = thought
                                row["replan_dag"] = json.dumps(plan_dag, ensure_ascii=False) if plan_dag else ""
                                rows.append(row)
                            df = pd.DataFrame(rows)
                            df.to_excel(save_func, index=False)
                            logger.info(f"Incremental save: {completed_count} rows")
                    except Exception as exc:
                        pbar.write(f"Error: {params['id']} failed: {exc}")
                        pbar.update(1)

            return results


def main():
    parser = argparse.ArgumentParser(description="Batch call LLM to perform replanning")
    parser.add_argument("-q", "--qps", default=10, type=int)
    parser.add_argument("-f", "--input_file", default=None)
    parser.add_argument("-s", "--save_file", default="output/replanner_results.xlsx")
    parser.add_argument("--model", default=None)
    parser.add_argument("--limit", default=None, type=int)
    args = parser.parse_args()

    input_file = args.input_file or os.path.join("data", "LEVEL-1", "LEVEL-1_augmented.jsonl")
    os.makedirs(os.path.dirname(args.save_file) or ".", exist_ok=True)

    model = args.model or os.getenv("OPENAI_MODEL", "deepseek-chat")

    logger.info(f'job begin, qps: {args.qps}, input: {input_file}, save: {args.save_file}, model: {model}')

    records = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if args.limit:
        records = records[:args.limit]
    logger.info(f"Loaded {len(records)} records")

    test_params = []
    for rec in records:
        query = rec.get("query", "")
        tool_list = json.dumps(rec.get("tool_list", []), ensure_ascii=False)
        plan_dag = json.dumps(rec.get("plan_dag", {}), ensure_ascii=False)
        fun_call = json.dumps(rec.get("fun_call", []), ensure_ascii=False)
        prompt = build_prompt(query=query, tool_list=tool_list, plan_dag=plan_dag, tool_call_dag=fun_call)
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt}
        ]
        test_params.append({
            "id": str(rec.get("id", "")),
            "origin": rec,
            "request_body": {"messages": messages}
        })

    client = create_client()
    poster = QPSLimitedExecutor(client=client, model=model, qps=args.qps, max_workers=50)
    results = poster.start_concurrent_executing(
        params_list=test_params,
        save_interval=100,
        save_func=args.save_file.replace(".xlsx", "_checkpoint.xlsx")
    )

    rows = []
    for result in tqdm(results, desc="building xlsx rows"):
        origin = result.get("origin", {})
        output = result.get("output", "")
        row = {}
        for k, v in origin.items():
            if isinstance(v, (dict, list)):
                row[k] = json.dumps(v, ensure_ascii=False)
            else:
                row[k] = v
        content = output if isinstance(output, str) else ""
        thought, plan_dag = extract_thought_and_plan(content)
        row["thought"] = thought
        row["replan_dag"] = json.dumps(plan_dag, ensure_ascii=False) if plan_dag else ""
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_excel(args.save_file, index=False)
    logger.info(f"xlsx saved to {args.save_file}")
    logger.info(f'job end')


if __name__ == "__main__":
    main()
