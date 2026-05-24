import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from environment.replanner_agent import (
    build_initial_dag,
    get_next_executable_tasks,
    is_task_completed,
    update_dag_with_results,
    RePlannerAgent,
)


def test_build_initial_dag():
    dag = build_initial_dag("test query")
    assert "tasks" in dag
    assert len(dag["tasks"]) == 1
    assert dag["tasks"][0]["id"] == "root"
    assert dag["tasks"][0]["desc"] == "test query"


def test_parse_output_with_thought_and_json():
    agent = RePlannerAgent(api_key="sk-test", base_url="https://api.deepseek.com")
    text = "<thought>analysis: need to search weather</thought>\n{\"tasks\": [{\"id\": \"root\", \"desc\": \"test\", \"dep\": [], \"status\": true}]}"
    thought, plan_dag, answer = agent.parse_output(text)
    assert thought == "analysis: need to search weather"
    assert isinstance(plan_dag, dict)
    assert "tasks" in plan_dag


def test_parse_output_with_answer():
    agent = RePlannerAgent(api_key="sk-test", base_url="https://api.deepseek.com")
    text = """<thought>All tasks completed</thought>
{"tasks": []}
<Answer>The final answer is 42</Answer>"""
    thought, plan_dag, answer = agent.parse_output(text)
    assert answer == "The final answer is 42"
    assert plan_dag == {"tasks": []}


def test_parse_output_with_malformed_json():
    agent = RePlannerAgent(api_key="sk-test", base_url="https://api.deepseek.com")
    text = "<thought>test</thought> this is not json"
    thought, plan_dag, answer = agent.parse_output(text)
    assert thought == "test"
    assert plan_dag == {}


def test_get_next_executable_tasks():
    dag = {
        "tasks": [
            {"id": "root", "desc": "test", "dep": [], "status": True},
            {"id": "1", "name": "search", "desc": "search", "dep": ["root"], "status": False},
            {"id": "2", "name": "calc", "desc": "calc", "dep": ["1"], "status": False},
        ]
    }
    tasks = get_next_executable_tasks(dag)
    assert len(tasks) == 1
    assert tasks[0]["id"] == "1"


def test_is_task_completed():
    dag = {
        "tasks": [
            {"id": "root", "desc": "test", "dep": [], "status": True},
            {"id": "response", "desc": "done", "dep": ["root"], "status": True},
        ]
    }
    completed, desc = is_task_completed(dag)
    assert completed
    assert desc == "done"


def test_update_dag_with_results():
    dag = {
        "tasks": [
            {"id": "root", "desc": "test", "dep": [], "status": True},
            {"id": "1", "name": "search", "desc": "search", "dep": ["root"], "status": False},
        ]
    }
    tool_results = [{"name": "search", "results": "found results"}]
    dag = update_dag_with_results(dag, tool_results)
    assert dag["tasks"][1]["status"] is True
    assert "[Result]: found results" in dag["tasks"][1]["desc"]


if __name__ == "__main__":
    test_build_initial_dag()
    test_parse_output_with_thought_and_json()
    test_parse_output_with_answer()
    test_parse_output_with_malformed_json()
    test_get_next_executable_tasks()
    test_is_task_completed()
    test_update_dag_with_results()
    print("All replanner_parser tests passed!")
