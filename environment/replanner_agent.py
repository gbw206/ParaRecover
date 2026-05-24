import json
import re
import time
import os
from openai import OpenAI

MAX_RETRIES = 3
RETRY_DELAY = 2

class RePlannerAgent:
    def __init__(self, model=None, api_key=None, base_url=None):
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
        )
        self.model = model or os.getenv("OPENAI_MODEL", "deepseek-reasoner")

    def build_prompt(self, query, tool_list, plan_dag, tool_call_dag):
        PROMPT_TEMPLATE = """
You are an expert at solving problems by calling various tools. Focus on observing the [Previous Plan] and [Previous Tool Call Results], and combine them with the [Original User Task] and [Available Tool List] to make appropriate reflections and plans.

## Task:
1. Analyze the user's original task, combined with the previous DAG plan and previous tool call results, to determine what still needs to be done to complete the task.
2. Based on the tools that still need to be called, update the complete task list to guide the subsequent workflow.

## Output Requirements
- First, output the <thought> section. Do not repeat previous thought content; re-analyze from scratch. Include analysis of the original user task, conduct a requirements analysis, identify which sub-tasks have been completed and which still need to be done, then detail the steps required to complete the original task and explain the reasoning for each step. The thought must include a detailed analysis of the previous plan's rationality as well as a detailed analysis of the previous tool call arrangement (including tool selection and parameters).
- Second, output a parseable JSON representing the task dependency graph for tool calls.
    1. Clearly define each sub-task's dependencies. A dependency exists only when a sub-task needs the return value from a predecessor task to fill or complete its own tool call parameters. All sub-tasks that do not require predecessor information should list "root" as their dependency.
    2. For sub-task nodes that have not yet been executed, you may modify, add, or delete them freely to better call tools and complete the task. Note that already-executed sub-task nodes must not be modified and should NOT be output.
    3. The first sub-task node is designated as the root node:
{{"id": "root",  # Root node: represents the starting point of the task
"desc": query, # The original user task content
"dep": [],  # Root node has no dependencies
"status": True   # Root node defaults to executed}}
    The last sub-task node is designated as the response node:
{{"id": "response",  # Final node: represents the endpoint of the task
"desc": "Analyze and summarize, generate response", # Aggregate all sub-task results and generate a response
"dep": [...],  # All nodes not depended on by any other sub-task
"status": False    # Response node defaults to not yet executed}}
    4. In the task nodes, "status" represents the execution state: True means executed, False means not executed.
    5. If there are sub-tasks that need to be executed, output ALL sub-task nodes, including root and response.
    6. Based on [Previous Execution Results], analyze which sub-tasks have been completed and mark their status as True.
    7. Based on [Previous Execution Results], append the tool execution results (including failed results) to the corresponding node's desc field. Do not change the original description; append content to it.
    8. For sub-tasks where the tool returned an Error or other failure signal, set the node's status to True, mark the failure signal in the desc field, then design and create new sub-task nodes to resolve the issue. The output plan must be informative and include both success and failure signals.
    9. If no sub-tasks remain to be executed, output an empty task list and provide a final answer based on the original user task. The final answer must be wrapped in <Answer></Answer> tags.
    10. Since you cannot modify already-executed tool call task nodes (regardless of their results), if you need to execute sub-tasks similar to already-executed nodes, you must create new task nodes.

## Output Format (strict: output thought section and JSON task list):
<thought>Output your analysis of the current user query...</thought>

{{"tasks": [...]}}

## Original Task
{query}

## Available Tool List
{tool_list}

## Previous Plan
{plan_dag}

## Previous Execution Results
{tool_call_dag}
"""
        return PROMPT_TEMPLATE.format(query=query, tool_list=tool_list, plan_dag=plan_dag, tool_call_dag=tool_call_dag)

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

    def plan(self, query, tool_list, plan_dag=None, tool_call_dag=None):
        if plan_dag is None:
            plan_dag = json.dumps({"tasks": [{"id": "root", "desc": query, "dep": [], "status": True}]})
        if tool_call_dag is None:
            tool_call_dag = "[]"

        prompt = self.build_prompt(query, tool_list, plan_dag, tool_call_dag)
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt}
        ]

        output = self.call(messages)
        return self.parse_output(output)

    def parse_output(self, text):
        thought = ""
        plan_dag = {}

        thought_match = re.search(r"<thought>(.*?)</thought>", text, re.S)
        if thought_match:
            thought = thought_match.group(1).strip()

        json_match = re.search(r'\{.*\}', text, re.S)
        if json_match:
            try:
                plan_dag = json.loads(json_match.group())
            except Exception:
                plan_dag = {}

        answer_match = re.search(r"<Answer>(.*?)</Answer>", text, re.S)
        final_answer = answer_match.group(1).strip() if answer_match else None

        return thought, plan_dag, final_answer


def build_initial_dag(query):
    return {
        "tasks": [
            {"id": "root", "desc": query, "dep": [], "status": True}
        ]
    }


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

        all_deps_completed = all(
            task_map.get(dep, {}).get("status", False)
            for dep in deps
            if dep in task_map
        )

        if all_deps_completed:
            executable.append(task)

    return executable


def update_dag_with_results(plan_dag, tool_results):
    if not plan_dag or "tasks" not in plan_dag:
        return plan_dag

    tasks = plan_dag["tasks"]
    for result in tool_results:
        tool_name = result.get("name", "")
        result_text = result.get("results", "")

        for task in tasks:
            if task.get("name") == tool_name and not task.get("status", False):
                task["status"] = True
                task["desc"] = task.get("desc", "") + f"\n[Result]: {result_text}"
                break

    return plan_dag


def is_task_completed(plan_dag):
    if not plan_dag or "tasks" not in plan_dag:
        return False, None

    for task in plan_dag["tasks"]:
        if task.get("id") == "response" and task.get("status", False):
            return True, task.get("desc", "")

    return False, None
