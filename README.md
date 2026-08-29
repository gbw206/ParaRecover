# ParaRecover: A Process-Level Benchmark for Error Localization and Recovery in Parallel Tool-Use Agents

**ParaRecover** is a benchmark for evaluating the re-planning (recovery) ability of LLM-based agents. It tests whether an agent can detect execution errors, diagnose root causes, and repair its plan accordingly when tool calls fail.

## Quick Start

```bash
# 1. Create and activate conda environment
conda create -n pararecover python=3.9 -y
conda activate pararecover

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up API credentials
cp .env.example .env
# Edit .env with your API key

# 4. Run the batch replanner on the benchmark
python scripts/run_batch_replanner.py \
    --input data/LEVEL-1/LEVEL-1_augmented.jsonl \
    --output output/results.xlsx

# 5. Evaluate replanning quality
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
```

## Evaluation Dimensions

The benchmark evaluates replanner output across three dimensions (14 sub-items + 1 script check):

- **Structural (S1–S5)**: Plan structure, dependency correctness, tool call legality, DAG validity
- **Diagnostic (D1–D5)**: Evidence anchoring, root cause localization, impact analysis
- **Evolutionary (E1–E5)**: Minimality of changes, fix closure, goal preservation

