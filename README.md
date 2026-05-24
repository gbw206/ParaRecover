# ParaRecover: Benchmarking RePlanner Agents on Error Recovery

**ParaRecover** is a benchmark for evaluating the re-planning (recovery) ability of LLM-based agents. It tests whether an agent can detect execution errors, diagnose root causes, and repair its plan accordingly when tool calls fail.

## Dataset Overview

| Split | Samples | Description |
|---|---|---|
| LEVEL-1 Augmented | ~12K | Single-round errors (clear, isolated tool call failures) |
| LEVEL-1 Real Rollout | ~3.5K | Single-round errors from real agent rollouts |
| LEVEL-2 Augmented | ~2.2K | Multi-round cascading errors |
| LEVEL-2 Real Rollout | ~400 | Multi-round errors from real agent rollouts |

Each sample contains:
- `query` — The original user task
- `tool_list` — Available tools with specifications
- `plan_dag` — The plan DAG before replanning
- `fun_call` — Tool call results (including errors)
- `thought` + `replan` — The ground-truth replanning decision

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up API credentials
cp .env.example .env
# Edit .env with your API key

# 3. Run the batch replanner on the benchmark
python scripts/run_batch_replanner.py \
    --input data/LEVEL-1/LEVEL-1_augmented.jsonl \
    --output output/results.xlsx

# 4. Evaluate replanning quality
python scripts/evaluate_multi_dimension.py \
    --input output/results.xlsx \
    --output output/evaluation.xlsx
```

## Project Structure

```
ParaRecover/
├── data/               # Benchmark datasets (JSONL)
├── environment/        # Core agent environment modules
├── scripts/            # Entry-point scripts for execution & evaluation
├── analysis/           # Statistics, visualization, data cleaning
└── tests/              # Unit tests
```

## Evaluation Dimensions

The benchmark evaluates replanner output across three dimensions (14 sub-items + 1 script check):

- **Structural (S1–S5)**: Plan structure, dependency correctness, tool call legality, DAG validity
- **Diagnostic (D1–D5)**: Evidence anchoring, root cause localization, impact analysis
- **Evolutionary (E1–E5)**: Minimality of changes, fix closure, goal preservation

## Citation

```bibtex
@article{pararecover2026,
  title={ParaRecover: Benchmarking RePlanner Agents on Error Recovery},
  author={...},
  journal={...},
  year={2026}
}
```

## License

MIT
