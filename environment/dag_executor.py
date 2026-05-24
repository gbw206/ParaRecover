import json
import re
import time
import os
import argparse
import pandas as pd
from openai import OpenAI

MAX_RETRIES = 3
RETRY_DELAY = 2
THOUGHT_PATTERN = re.compile(r"<thought>(.*?)</thought>", re.S)


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
- First, output the <thought> section. Do not repeat previous thought content; re-analyze from scratch. Include analysis of the original user task, conduct a requirements analysis, identify which sub-tasks have been completed and which still need to be done, then detail the steps required to complete the original task and explain the reasoning for each step. The thought must include a detailed analysis of the previous plan's rationality as well as a detailed analysis of the previous tool call arrangement (including tool selection and parameters).
- Second, output a parseable JSON representing the task dependency graph for tool calls.
    1. Clearly define each sub-task's dependencies. A dependency exists only when a sub-task needs the return value from a predecessor task to fill or complete its own tool call parameters. All sub-tasks that do not require predecessor information should list "root" as their dependency.
    2. For sub-task nodes that have not yet been executed, you may modify, add, or delete them freely to better call tools and complete the task. Note that already-executed sub-task nodes must not be modified.
    3. The first sub-task node is designated as the root node:
{{"id": "root",
"desc": query,
"dep": [],
"status": True}}
    The last sub-task node is designated as the response node:
{{"id": "response",
"desc": "Analyze and summarize, generate response",
"dep": [...],
"status": False}}
    4. "status" represents the execution state: True means executed, False means not executed.
    5. If there are sub-tasks that need to be executed, output ALL sub-task nodes, including root and response.
    6. Based on [Previous Execution Results], analyze which sub-tasks have been completed and mark their status as True.
    7. For sub-tasks where the tool returned an Error or other failure signal, set the node's status to True, mark the failure signal in the desc field, then design and create new sub-task nodes to resolve the issue.
    8. For execution failures due to objective reasons such as network timeout (not tool call format errors), you may retry up to three times.
    9. Already-executed tool call nodes must not be modified. If you need to execute sub-tasks similar to already-executed nodes, you must create new task nodes.
    10. If all nodes except the response node have been completed, set the response node's status to True and provide the final task completion summary in its desc field.

## Output Format
<thought>...</thought>
{{"tasks": [...]}}

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

    def parse_output(self, text: str):
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
        return self.parse_output(output)


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
- Based on each task node's description and tool name, simulate the reasonable results that tool should return
- For calculation tools, provide correct computational results
- For search tools, return a reasonable search result summary

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
            {"role": "system", "content": "You are a professional tool call simulator."},
            {"role": "user", "content": prompt}
        ]

        output = self.call(messages)
        return self.parse_output(output)

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

    def parse_output(self, text):
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

    def run_from_file(self, dag_file_path, tool_list=None):
        if not os.path.exists(dag_file_path):
            raise FileNotFoundError(f"DAG file not found: {dag_file_path}")

        with open(dag_file_path, 'r', encoding='utf-8') as f:
            dag_json = f.read()

        try:
            plan_dag = json.loads(dag_json)
        except json.JSONDecodeError:
            raise ValueError("DAG file format error, unable to parse as JSON")

        return self.run(plan_dag, tool_list)

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

            replan_thought, new_dag = self.replanner_agent.replan(
                query=user_query,
                tool_list=tool_list,
                previous_dag=current_dag,
                tool_results=tool_results
            )

            if new_dag is None:
                return {
                    "status": "error",
                    "rounds": round_num,
                    "final_dag": current_dag,
                    "execution_log": self.execution_log,
                    "error": "Replanner output DAG parse failed"
                }

        return {
            "status": "incomplete",
            "rounds": round_num,
            "final_dag": current_dag,
            "execution_log": self.execution_log
        }


def execute_from_xlsx(input_file, output_file, api_key=None, base_url=None):
    df = pd.read_excel(input_file)

    df["execute_round"] = -1
    df["execute_process"] = ""
    df["final_dag"] = ""

    for idx, row in df.iterrows():
        query = row.get("query", "")
        tool_list = row.get("tool_list", "[]")
        replan_dag_str = row.get("replan_dag", "{}")

        plan_dag = None
        try:
            plan_dag = json.loads(replan_dag_str) if replan_dag_str else {}
        except json.JSONDecodeError:
            df.at[idx, "execute_round"] = -1
            df.at[idx, "execute_process"] = "replan_dag parse failed"
            df.at[idx, "final_dag"] = replan_dag_str
            continue

        if not plan_dag or "tasks" not in plan_dag:
            df.at[idx, "execute_round"] = -1
            df.at[idx, "execute_process"] = "replan_dag is empty"
            df.at[idx, "final_dag"] = replan_dag_str
            continue

        executor = DAGExecutor(api_key=api_key, base_url=base_url)
        result = executor.run(plan_dag, tool_list)

        df.at[idx, "execute_round"] = result.get("rounds", -1)
        df.at[idx, "execute_process"] = json.dumps(result.get("execution_log", []), ensure_ascii=False)
        df.at[idx, "final_dag"] = json.dumps(result.get("final_dag", {}), ensure_ascii=False)

    df.to_excel(output_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute DAG from xlsx")
    parser.add_argument("-i", "--input", default="input.xlsx", help="Input xlsx file path")
    parser.add_argument("-o", "--output", default="output.xlsx", help="Output xlsx file path")
    args = parser.parse_args()
    execute_from_xlsx(args.input, args.output)
