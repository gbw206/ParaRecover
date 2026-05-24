import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_json_roundtrip():
    sample = {
        "tasks": [
            {"id": "root", "desc": "test", "dep": [], "status": True},
            {"id": "1", "name": "search", "para": {"q": "test"},
             "desc": "search", "dep": ["root"], "status": False},
        ]
    }
    serialized = json.dumps(sample, ensure_ascii=False)
    deserialized = json.loads(serialized)
    assert deserialized == sample
    assert len(deserialized["tasks"]) == 2


def test_jsonl_line_format():
    records = [
        {"id": 1, "query": "test1"},
        {"id": 2, "query": "test2"},
    ]
    lines = [json.dumps(r, ensure_ascii=False) for r in records]
    serialized = "\n".join(lines)
    deserialized = [json.loads(line) for line in serialized.split("\n")]
    assert len(deserialized) == 2
    assert deserialized[0]["id"] == 1


def test_dag_validator():
    from scripts.evaluate_multi_dimension import score_structural_script

    valid_dag = {
        "tasks": [
            {"id": "root", "dep": [], "status": True},
            {"id": "1", "name": "search", "dep": ["root"], "status": False},
            {"id": "response", "dep": ["1"], "status": False},
        ]
    }
    score, _ = score_structural_script(valid_dag)
    assert score == "1", f"Expected 1, got {score}"

    cyclic_dag = {
        "tasks": [
            {"id": "root", "dep": ["1"], "status": True},
            {"id": "1", "dep": ["root"], "status": False},
        ]
    }
    score, _ = score_structural_script(cyclic_dag)
    assert score == "0", f"Expected 0 for cyclic, got {score}"

    missing_dep_dag = {
        "tasks": [
            {"id": "root", "dep": [], "status": True},
            {"id": "1", "dep": ["nonexistent"], "status": False},
        ]
    }
    score, _ = score_structural_script(missing_dep_dag)
    assert score == "0", f"Expected 0 for missing dep, got {score}"


if __name__ == "__main__":
    test_json_roundtrip()
    test_jsonl_line_format()
    test_dag_validator()
    print("All xlsx/jsonl tests passed!")
