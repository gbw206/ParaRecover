import json
import re
import time
import os
from openai import OpenAI

MAX_RETRIES = 3
RETRY_DELAY = 2

class ToolAgent:
    def __init__(self, model=None, api_key=None, base_url=None):
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
        )
        self.model = model or os.getenv("OPENAI_MODEL", "deepseek-chat")

    def build_prompt(self, query, tool_list, replan_dag, tool_results_history):
        PROMPT_TEMPLATE = """
You are an expert at simulating agent tool calls. Based on the [User Task] and [Available Tool List], simulate the return results of tool execution.

## Your Task
- Carefully analyze the next wave of task nodes to be executed from the [Previous DAG Plan] (nodes with status False whose dependencies are satisfied)
- Based on each task node's description and tool name, simulate the reasonable results that tool should return
- Simulation results must be plausible and match real-world scenarios
- For calculation tools, provide correct computational results
- For search tools, return a reasonable search result summary (you may fabricate reasonable values, but they must be relevant to the task)

## DAG Description
- The previous plan is presented as a DAG, starting from the root node and ending at the response node
- DAG structure: {{"tasks": [{{"id": string, "name": string, "para": [string], "desc": string, "dep": [string], "status": bool}}, ...]}}
- The next wave to execute consists of nodes with status False whose predecessor nodes all have status True

## Tool Return Result Format (JSON Array):
[
    {{
        "name": string,
        "arguments": {{"param_name": "param_value"}},
        "results": string
    }},
    ...
]

## Output Requirements
- First, output your analysis in <thought></thought> tags
- Then, output the tool return results as a JSON array

## User Task
{query}

## Available Tool List
{tool_list}

## Previous DAG Plan
{replan_dag}

## Notes
- You only need to simulate tool execution results; do not filter them
- Simulation results must be reasonable and task-relevant
- If the tool is a calculator, the calculation result must be correct
- If the tool is a search or weather tool, return reasonable simulated values
"""
        return PROMPT_TEMPLATE.format(
            query=query,
            tool_list=tool_list,
            replan_dag=replan_dag,
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
        return self.parse_output(output)

    def parse_output(self, text):
        thought = ""
        tool_results = []

        thought_match = re.search(r"<thought>(.*?)</thought>", text, re.S)
        if thought_match:
            thought = thought_match.group(1).strip()

        json_match = re.search(r'\[.*\]', text, re.S)
        if json_match:
            try:
                tool_results = json.loads(json_match.group())
                if not isinstance(tool_results, list):
                    tool_results = []
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
            if dep not in task_map or not task_map[dep].get("status", False):
                all_deps_completed = False
                break

        if all_deps_completed:
            executable.append(task)

    return executable


def simulate_tool_call(tool_name, arguments, tool_descriptions):
    for tool_desc in tool_descriptions:
        if tool_desc.get("name") == tool_name:
            return f"[Simulated] Called tool {tool_name}, arguments: {arguments}, result: Simulated tool execution result"

    return f"[Simulated] Tool {tool_name} not found"
