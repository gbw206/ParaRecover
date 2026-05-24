import json
import re
import time
import os
import argparse
import ast
import pandas as pd
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import Dict, Any, List, Optional, Tuple

MAX_RETRIES = 3
RETRY_DELAY = 2
THOUGHT_PATTERN = re.compile(r"<thought>(.*?)</thought>", re.S)
PLAN_PATTERN = re.compile(r"<plan>(.*?)</plan>", re.S)


class ReplannerAgent:
    def __init__(self, model="deepseek-chat", api_key=None, base_url=None):
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
        )
        self.model = model

    def build_prompt(self, query, tool_list, previous_dag, tool_results):
        prompt = f"""
You are an expert at solving problems by calling various tools. Focus on observing the [Previous Plan] and [Previous Tool Call Results], and combine them with the [Original User Task] and [Available Tool List] to make appropriate reflections and plans.

## Task:
1. Analyze the user's original task, combined with the previous DAG plan and previous tool call results, to determine what still needs to be done to complete the task.
2. Based on the tools that still need to be called, update the complete task list to guide the subsequent workflow.

## Output Requirements
- First, output the <thought> section. Do not repeat previous thought content; re-analyze from scratch.
- Second, output a parseable JSON within <plan> tags, representing the task dependency graph for tool calls.
    1. Clearly define each sub-task's dependencies.
    2. For sub-task nodes that have not yet been executed, you may modify, add, or delete them freely.
    3. The first sub-task node is designated as the root node, and the last as the response node.
    4. "status" represents the execution state: True means executed, False means not executed.
    5. Based on [Previous Execution Results], analyze which sub-tasks have been completed and mark their status as True.
    6. For sub-tasks where the tool returned an Error or other failure signal, set the node's status to True, mark the failure signal in the desc field, then design and create new sub-task nodes to resolve the issue.
    7. For execution failures due to objective reasons such as network timeout (not tool call format errors), you may retry up to three times.
    8. Already-executed tool call nodes must not be modified.
    9. If all nodes except the response node have been completed, set the response node's status to True and provide the final task completion summary in its desc field.
    10. You may only use tools from the [Tool List]; you must not fabricate tools.

## Output Format
<thought>...</thought>
<plan>
{{"tasks": [...]}}
</plan>

## Original Task
{query}

## Available Tool List
{tool_list}

## Previous DAG Plan
{previous_dag}

## Tool Return Results
{json.dumps(tool_results, ensure_ascii=False, indent=2)}
"""
        return prompt

    def call(self, messages):
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=False,
                    timeout=60
                )
                return response.choices[0].message.content
            except Exception:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    raise

    def parse_agent_output(self, text: str) -> Tuple[str, Optional[dict]]:
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")

        text = text.strip()

        thought_match = re.search(r"<thought>\s*(.*?)\s*</thought>", text, flags=re.S | re.I)
        thought = thought_match.group(1).strip() if thought_match else ""

        plan_match = re.search(r"<plan>\s*(.*?)\s*</plan>", text, flags=re.S | re.I)

        if not plan_match:
            return thought, None

        plan_raw = plan_match.group(1).strip()
        if not plan_raw:
            return thought, None

        code_block_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", plan_raw, flags=re.S | re.I)
        if code_block_match:
            plan_raw = code_block_match.group(1).strip()

        plan_dag = self._parse_plan_dag_robust(plan_raw)
        if plan_dag is not None:
            return thought, plan_dag

        json_str = self.extract_outermost_json_object(plan_raw)
        if json_str is None:
            return thought, None

        return thought, None

    def _parse_plan_dag_robust(self, raw: str) -> Optional[dict]:
        if not isinstance(raw, str) or not raw.strip():
            return None

        candidates = [raw.strip()]
        outer = self.extract_outermost_json_object(raw)
        if outer:
            candidates.append(outer.strip())

        tried = set()
        for candidate in candidates:
            if not candidate or candidate in tried:
                continue
            tried.add(candidate)

            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

            normalized = self._normalize_plan_json_like(candidate)
            try:
                parsed = json.loads(normalized)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

            py_literal = self._replace_tokens_outside_strings(
                normalized,
                {"true": "True", "false": "False", "null": "None"}
            )
            try:
                parsed = ast.literal_eval(py_literal)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

        return None

    def _normalize_plan_json_like(self, text: str) -> str:
        s = text.strip()
        s = self._remove_json_like_comments(s)
        s = self._remove_trailing_commas(s)
        s = self._replace_tokens_outside_strings(
            s,
            {"True": "true", "False": "false", "None": "null"}
        )
        return s

    def _remove_json_like_comments(self, text: str) -> str:
        out = []
        in_string = False
        escape = False
        quote_char = '"'
        i = 0
        n = len(text)

        while i < n:
            ch = text[i]
            if in_string:
                out.append(ch)
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == quote_char:
                    in_string = False
                i += 1
                continue
            if ch in ('"', "'"):
                in_string = True
                quote_char = ch
                out.append(ch)
                i += 1
                continue
            if ch == "#":
                while i < n and text[i] not in ("\n", "\r"):
                    i += 1
                continue
            if ch == "/" and i + 1 < n and text[i + 1] == "/":
                i += 2
                while i < n and text[i] not in ("\n", "\r"):
                    i += 1
                continue
            out.append(ch)
            i += 1
        return "".join(out)

    def _remove_trailing_commas(self, text: str) -> str:
        out = []
        in_string = False
        escape = False
        quote_char = '"'
        i = 0
        n = len(text)
        while i < n:
            ch = text[i]
            if in_string:
                out.append(ch)
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == quote_char:
                    in_string = False
                i += 1
                continue
            if ch in ('"', "'"):
                in_string = True
                quote_char = ch
                out.append(ch)
                i += 1
                continue
            if ch == ",":
                j = i + 1
                while j < n and text[j] in (" ", "\t", "\r", "\n"):
                    j += 1
                if j < n and text[j] in ("]", "}"):
                    i += 1
                    continue
            out.append(ch)
            i += 1
        return "".join(out)

    def _replace_tokens_outside_strings(self, text: str, mapping: Dict[str, str]) -> str:
        out = []
        in_string = False
        escape = False
        quote_char = '"'
        i = 0
        n = len(text)
        while i < n:
            ch = text[i]
            if in_string:
                out.append(ch)
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == quote_char:
                    in_string = False
                i += 1
                continue
            if ch in ('"', "'"):
                in_string = True
                quote_char = ch
                out.append(ch)
                i += 1
                continue
            replaced = False
            for src, dst in mapping.items():
                if text.startswith(src, i):
                    prev_char = text[i - 1] if i > 0 else ""
                    next_index = i + len(src)
                    next_char = text[next_index] if next_index < n else ""
                    prev_word = (isinstance(prev_char, str) and len(prev_char) == 1 and (prev_char.isalnum() or prev_char == "_"))
                    next_word = (isinstance(next_char, str) and len(next_char) == 1 and (next_char.isalnum() or next_char == "_"))
                    if not prev_word and not next_word:
                        out.append(dst)
                        i += len(src)
                        replaced = True
                        break
            if replaced:
                continue
            out.append(ch)
            i += 1
        return "".join(out)

    def extract_outermost_json_object(self, text: str) -> Optional[str]:
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        return None

    def replan(self, query, tool_list, previous_dag, tool_results):
        prompt = self.build_prompt(
            query=query,
            tool_list=tool_list,
            previous_dag=json.dumps(previous_dag, ensure_ascii=False),
            tool_results=tool_results
        )
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt}
        ]
        output = self.call(messages)
        thought, plan_raw = self.extract_thought_and_plan_raw(output)
        try:
            parsed_thought, parsed_plan = self.parse_agent_output(output)
            if parsed_thought:
                thought = parsed_thought
            return thought, parsed_plan, plan_raw, "ok", ""
        except Exception as e:
            return thought, None, plan_raw, "parse_failed", str(e)

    def extract_thought_and_plan_raw(self, text: str) -> Tuple[str, str]:
        if not isinstance(text, str):
            return "", ""
        text = text.strip()
        thought = ""
        plan_raw = ""
        thought_match = re.search(r"<thought>\s*(.*?)\s*</thought>", text, flags=re.S | re.I)
        if thought_match:
            thought = thought_match.group(1).strip()
        plan_match = re.search(r"<plan>\s*(.*?)\s*</plan>", text, flags=re.S | re.I)
        if plan_match:
            plan_raw = plan_match.group(1).strip()
            code_block_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", plan_raw, flags=re.S | re.I)
            if code_block_match:
                plan_raw = code_block_match.group(1).strip()
            return thought, plan_raw
        extracted = self.extract_outermost_json_object(text)
        if extracted:
            plan_raw = extracted
        return thought, plan_raw


class ToolAgent:
    def __init__(self, model="deepseek-chat", api_key=None, base_url=None):
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
        )
        self.model = model

    def build_prompt(self, query, tool_list, plan_dag, tool_results_history):
        PROMPT_TEMPLATE = """
You are an expert at simulating agent tool calls. Based on the [User Task] and [Available Tool List], simulate the return results of tool execution.

## Your Task
- Carefully analyze the next wave of task nodes to be executed from the [Previous DAG Plan] (nodes with status False whose dependencies are satisfied)
- Analyze whether the tool invocations in the [Previous DAG Plan] are correct; check if the tool name, parameter names, and parameter types conform to the standard calling conventions in the [Available Tool List]. If there are any discrepancies, you should directly return a reasonable error message.
- If the tool call is correct, simulate the reasonable results that the tool should return based on each task node's description and tool name
- For calculation tools, provide correct computational results
- For search tools, return a reasonable search result summary
- When the tool call is correct, introduce a 2% probability of returning erroneous results (e.g., timeout errors, empty results)

## Output Format
<thought>...</thought>
[
    {{
        "name": string,
        "arguments": {{}},
        "results": string
    }}
]

## User Task
{query}

## Available Tool List
{tool_list}

## Previous DAG Plan
{replan_dag}
"""
        return PROMPT_TEMPLATE.format(
            query=query,
            tool_list=tool_list,
            replan_dag=plan_dag,
            tool_results_history=tool_results_history
        )

    def call(self, messages):
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=False,
                    timeout=60
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    wait_time = RETRY_DELAY * (attempt + 1)
                    time.sleep(wait_time)
                else:
                    raise e

    def execute_next_tasks(self, query, tool_list, plan_dag):
        prompt = self.build_prompt(query, tool_list, plan_dag, "")
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt}
        ]
        output = self.call(messages)
        return self.parse_tool_output(output)

    def _extract_thought(self, text: str):
        if not text:
            return "", ""
        m = THOUGHT_PATTERN.search(text)
        if m:
            thought = m.group(1).strip()
            rest = text[m.end():].strip()
            return thought, rest
        start_match = re.search(r"<thought\s*>", text, re.I)
        if start_match:
            thought = text[start_match.end():].strip()
            return thought, ""
        return "", text.strip()

    def _extract_first_json_array(self, text: str):
        if not text:
            return None
        start = text.find("[")
        while start != -1:
            in_string = False
            escape = False
            depth = 0
            for i in range(start, len(text)):
                ch = text[i]
                if in_string:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_string = False
                else:
                    if ch == '"':
                        in_string = True
                    elif ch == "[":
                        depth += 1
                    elif ch == "]":
                        depth -= 1
                        if depth == 0:
                            candidate = text[start:i + 1]
                            try:
                                json.loads(candidate)
                                return candidate
                            except Exception:
                                break
            start = text.find("[", start + 1)
        return None

    def parse_tool_output(self, text):
        thought = ""
        tool_results = []
        if not isinstance(text, str) or not text.strip():
            return thought, tool_results
        thought, rest = self._extract_thought(text)
        json_str = self._extract_first_json_array(rest)
        if json_str is None:
            json_str = self._extract_first_json_array(text)
        if json_str:
            try:
                parsed = json.loads(json_str)
                if isinstance(parsed, list):
                    tool_results = [item for item in parsed if isinstance(item, dict)]
            except Exception:
                tool_results = []
        return thought, tool_results


def get_next_executable_tasks(plan_dag):
    if not plan_dag or "tasks" not in plan_dag:
        return []
    tasks = plan_dag["tasks"]
    task_map = {t["id"]: t for t in tasks}
    executable = []
    for task in tasks:
        if task.get("status", False):
            continue
        deps = task.get("dep", [])
        if not deps or deps == ["root"]:
            executable.append(task)
            continue
        all_deps_completed = True
        for dep in deps:
            if dep not in task_map:
                all_deps_completed = False
                break
            if not task_map[dep].get("status", False):
                all_deps_completed = False
                break
        if all_deps_completed:
            executable.append(task)
    return executable


def is_all_completed(plan_dag):
    if not isinstance(plan_dag, dict):
        return False
    tasks = plan_dag.get("tasks")
    if not isinstance(tasks, list):
        return False
    for task in tasks:
        if not isinstance(task, dict):
            continue
        if task.get("id") == "response":
            status = task.get("status", False)
            if status is True:
                return True
            if isinstance(status, str) and status.strip().lower() == "true":
                return True
            return False
    return True


def get_root_desc(plan_dag):
    if not plan_dag or "tasks" not in plan_dag:
        return ""
    for task in plan_dag["tasks"]:
        if task.get("id") == "root":
            return task.get("desc", "")
    return ""


class DAGExecutor:
    def __init__(self, api_key=None, base_url=None, max_rounds=20):
        self.tool_agent = ToolAgent(api_key=api_key, base_url=base_url)
        self.replanner_agent = ReplannerAgent(api_key=api_key, base_url=base_url)
        self.max_rounds = max_rounds
        self.execution_log = []

    def run(self, plan_dag, tool_list=None):
        if tool_list is None:
            tool_list = json.dumps([
                {"name": "set_light_schedule", "description": "Set lighting system schedule"},
                {"name": "set_irrigation_schedule", "description": "Set irrigation system schedule"},
                {"name": "retrieve_light_schedule", "description": "Retrieve current lighting system schedule"},
                {"name": "retrieve_irrigation_schedule", "description": "Retrieve current irrigation system schedule"}
            ], ensure_ascii=False)

        current_dag = plan_dag
        user_query = get_root_desc(plan_dag)
        round_num = 0

        while round_num < self.max_rounds:
            round_num += 1
            next_tasks = get_next_executable_tasks(current_dag)
            if not next_tasks:
                break

            tool_thought, tool_results = self.tool_agent.execute_next_tasks(
                query=user_query,
                tool_list=tool_list,
                plan_dag=json.dumps(current_dag, ensure_ascii=False)
            )

            replan_thought, new_dag, replan_raw, replan_parse_status, replan_parse_error = self.replanner_agent.replan(
                query=user_query,
                tool_list=tool_list,
                previous_dag=current_dag,
                tool_results=tool_results
            )

            if new_dag is None:
                self.execution_log.append({
                    "round": round_num,
                    "input_dag": current_dag,
                    "tool_thought": tool_thought,
                    "tool_results": tool_results,
                    "replan_thought": replan_thought,
                    "replan_raw": replan_raw,
                    "replan_parse_status": replan_parse_status,
                    "replan_parse_error": replan_parse_error,
                    "output_dag": current_dag,
                    "error": "Replanner output DAG parse failed"
                })
                return {
                    "status": "error",
                    "rounds": round_num,
                    "final_dag": current_dag,
                    "execution_log": self.execution_log,
                    "error": "Replanner output DAG parse failed"
                }

            self.execution_log.append({
                "round": round_num,
                "input_dag": current_dag,
                "tool_thought": tool_thought,
                "tool_results": tool_results,
                "replan_thought": replan_thought,
                "replan_raw": replan_raw,
                "replan_parse_status": replan_parse_status,
                "replan_parse_error": replan_parse_error,
                "output_dag": new_dag
            })

            current_dag = new_dag

            if is_all_completed(current_dag):
                return {
                    "status": "success",
                    "rounds": round_num,
                    "final_dag": current_dag,
                    "execution_log": self.execution_log
                }

        return {
            "status": "incomplete",
            "rounds": round_num,
            "final_dag": current_dag,
            "execution_log": self.execution_log
        }


def process_single_row(idx, row):
    original_cols = {
        "id": row.get("id", ""),
        "tool_list": row.get("tool_list", "[]"),
        "query": row.get("query", ""),
        "plan_thought": row.get("plan_thought", ""),
        "plan_dag": row.get("plan_dag", "{}")
    }

    query = row.get("query", "")
    tool_list = row.get("tool_list", "[]")
    replan_dag_str = row.get("plan_dag", "{}")

    plan_dag = None
    try:
        plan_dag = json.loads(replan_dag_str) if replan_dag_str else {}
    except json.JSONDecodeError:
        return {
            "idx": idx,
            "status": "error",
            "error": "replan_dag parse failed",
            "execute_round": -1,
            "final_dag": replan_dag_str,
            "columns": {},
            "original": original_cols
        }

    if not plan_dag or "tasks" not in plan_dag:
        return {
            "idx": idx,
            "status": "error",
            "error": "replan_dag is empty",
            "execute_round": -1,
            "final_dag": replan_dag_str,
            "columns": {},
            "original": original_cols
        }

    executor = DAGExecutor()
    result = executor.run(plan_dag, tool_list)

    execute_round = result.get("rounds", -1)
    execution_log = result.get("execution_log", [])
    final_dag = result.get("final_dag", {})

    columns = {}
    for log in execution_log:
        r = log.get("round")
        columns[f"thought_fun_call_{r}"] = log.get("tool_thought", "")
        columns[f"fun_call_{r}"] = json.dumps(log.get("tool_results", []), ensure_ascii=False)
        columns[f"thought_replan_{r}"] = log.get("replan_thought", "")
        columns[f"replan_raw_{r}"] = log.get("replan_raw", "")
        columns[f"replan_parse_status_{r}"] = log.get("replan_parse_status", "")
        columns[f"replan_parse_error_{r}"] = log.get("replan_parse_error", "")
        columns[f"replan_{r}"] = json.dumps(log.get("output_dag", {}), ensure_ascii=False)

    return {
        "idx": idx,
        "status": result.get("status", "error"),
        "execute_round": execute_round,
        "final_dag": json.dumps(final_dag, ensure_ascii=False) if final_dag else replan_dag_str,
        "columns": columns,
        "original": original_cols
    }


def execute_from_xlsx_concurrent(input_file, output_file, concurrency=15):
    df = pd.read_excel(input_file)
    results = [None] * len(df)
    total = len(df)

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(process_single_row, idx, row): idx
            for idx, row in df.iterrows()
        }
        with tqdm(total=total, desc="Processing") as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    results[idx] = result
                except Exception as e:
                    failed_row = df.iloc[idx]
                    results[idx] = {
                        "idx": idx, "status": "error", "error": str(e),
                        "execute_round": -1, "final_dag": "", "columns": {},
                        "original": {
                            "id": failed_row.get("id", ""),
                            "tool_list": failed_row.get("tool_list", "[]"),
                            "query": failed_row.get("query", ""),
                            "plan_thought": failed_row.get("plan_thought", ""),
                            "plan_dag": failed_row.get("plan_dag", "{}")
                        }
                    }
                pbar.update(1)

    max_cols = 0
    for r in results:
        if r and r.get("columns"):
            max_cols = max(max_cols, len(r["columns"]))

    output_df = pd.DataFrame()
    output_df["id"] = [r.get("original", {}).get("id", "") if r else "" for r in results]
    output_df["tool_list"] = [r.get("original", {}).get("tool_list", "") if r else "" for r in results]
    output_df["query"] = [r.get("original", {}).get("query", "") if r else "" for r in results]
    output_df["plan_thought"] = [r.get("original", {}).get("plan_thought", "") if r else "" for r in results]
    output_df["plan_dag"] = [r.get("original", {}).get("plan_dag", "") if r else "" for r in results]
    output_df["execute_round"] = [r.get("execute_round", -1) if r else -1 for r in results]
    output_df["final_dag"] = [r.get("final_dag", "") if r else "" for r in results]

    def sort_columns(key):
        parts = key.split("_")
        if len(parts) >= 2:
            suffix = parts[-1]
            if suffix.isdigit():
                round_num = int(suffix)
                prefix = "_".join(parts[:-1])
                order = {
                    "thought_fun_call": 1, "fun_call": 2, "thought_replan": 3,
                    "replan_raw": 4, "replan_parse_status": 5, "replan_parse_error": 6, "replan": 7
                }
                return (round_num, order.get(prefix, 0))
        return (0, 0)

    all_keys = set()
    for r in results:
        if r and r.get("columns"):
            all_keys.update(r["columns"].keys())
    sorted_all_keys = sorted(all_keys, key=sort_columns)

    for key in sorted_all_keys:
        col_data = []
        for r in results:
            if r and r.get("columns") and key in r["columns"]:
                col_data.append(r["columns"][key])
            else:
                col_data.append("")
        if len(col_data) == len(results):
            output_df[key] = col_data

    output_df.to_excel(output_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute DAG from xlsx with high concurrency")
    parser.add_argument("-i", "--input", required=False, help="Input xlsx file path")
    parser.add_argument("-o", "--output", required=False, help="Output xlsx file path")
    parser.add_argument("-c", "--concurrency", type=int, default=15, help="Concurrency level")
    args = parser.parse_args()
    execute_from_xlsx_concurrent(args.input or "input.xlsx", args.output or "output.xlsx", args.concurrency)
