import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from environment.dag_executor import (
    get_next_executable_tasks,
    is_all_completed,
    get_root_desc,
    DAGExecutor,
)


def sample_dag():
    return {
        "tasks": [
            {"id": "root", "name": "", "para": {}, "desc": "Test query",
             "dep": [], "status": True},
            {"id": "1", "name": "search", "para": {"q": "test"},
             "desc": "Search task", "dep": ["root"], "status": False},
            {"id": "2", "name": "calculator", "para": {"expr": "1+1"},
             "desc": "Calculate", "dep": ["root"], "status": False},
            {"id": "3", "name": "format", "para": {},
             "desc": "Format results", "dep": ["1", "2"], "status": False},
            {"id": "response", "name": "", "para": {},
             "desc": "Respond", "dep": ["3"], "status": False},
        ]
    }


def test_get_next_executable_tasks():
    dag = sample_dag()
    tasks = get_next_executable_tasks(dag)
    task_ids = [t["id"] for t in tasks]
    assert "1" in task_ids, "Task 1 should be executable"
    assert "2" in task_ids, "Task 2 should be executable"
    assert "3" not in task_ids, "Task 3 should not be executable (depends on 1,2)"


def test_get_next_executable_after_completion():
    dag = sample_dag()
    dag["tasks"][1]["status"] = True  # task 1 completed
    dag["tasks"][2]["status"] = True  # task 2 completed
    tasks = get_next_executable_tasks(dag)
    task_ids = [t["id"] for t in tasks]
    assert "3" in task_ids, "Task 3 should now be executable"


def test_is_all_completed_not_done():
    dag = sample_dag()
    assert not is_all_completed(dag), "Should not be completed yet"


def test_is_all_completed_done():
    dag = sample_dag()
    for task in dag["tasks"]:
        task["status"] = True
    assert is_all_completed(dag), "Should be completed"


def test_get_root_desc():
    dag = sample_dag()
    desc = get_root_desc(dag)
    assert desc == "Test query", f"Expected 'Test query', got '{desc}'"


def test_get_root_desc_empty():
    assert get_root_desc({}) == ""
    assert get_root_desc({"tasks": []}) == ""


if __name__ == "__main__":
    test_get_next_executable_tasks()
    test_get_next_executable_after_completion()
    test_is_all_completed_not_done()
    test_is_all_completed_done()
    test_get_root_desc()
    test_get_root_desc_empty()
    print("All dag_executor tests passed!")
