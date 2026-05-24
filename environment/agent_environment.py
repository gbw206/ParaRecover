import json
import os
from replanner_agent import RePlannerAgent, build_initial_dag, get_next_executable_tasks, is_task_completed
from tool_agent import ToolAgent
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


class AgentEnvironment:
    def __init__(self, api_key=None, base_url=None, max_rounds=10):
        self.replanner = RePlannerAgent(api_key=api_key, base_url=base_url)
        self.tool_agent = ToolAgent(api_key=api_key, base_url=base_url)
        self.max_rounds = max_rounds
        self.tool_results_history = []

    def run(self, query, tool_list):
        logger.info(f"Processing query: {query[:50]}...")

        current_dag = build_initial_dag(query)
        round_num = 0

        while round_num < self.max_rounds:
            round_num += 1
            logger.info(f"=== Round {round_num} ===")

            thought, plan_dag, final_answer = self.replanner.plan(
                query=query,
                tool_list=tool_list,
                plan_dag=json.dumps(current_dag, ensure_ascii=False),
                tool_call_dag=json.dumps(self.tool_results_history, ensure_ascii=False)
            )

            if final_answer:
                logger.info(f"Got final answer")
                return {
                    "status": "success",
                    "answer": final_answer,
                    "rounds": round_num,
                    "final_dag": plan_dag
                }

            if not plan_dag or "tasks" not in plan_dag:
                logger.warning("Plan DAG is empty, trying to continue...")
                continue

            current_dag = plan_dag

            completed, response_desc = is_task_completed(plan_dag)
            if completed:
                logger.info("Task completed")
                return {
                    "status": "success",
                    "answer": response_desc,
                    "rounds": round_num,
                    "final_dag": plan_dag
                }

            next_tasks = get_next_executable_tasks(plan_dag)
            logger.info(f"Executable tasks in next wave: {len(next_tasks)}")

            if not next_tasks:
                logger.warning("No executable tasks")
                continue

            tool_results = self.tool_agent.execute_next_tasks(
                query=query,
                tool_list=tool_list,
                plan_dag=json.dumps(plan_dag, ensure_ascii=False)
            )

            if tool_results:
                logger.info(f"Tool call results count: {len(tool_results)}")
                self.tool_results_history.extend(tool_results)
            else:
                logger.warning("No tool call results obtained")

        logger.warning(f"Reached max rounds {self.max_rounds}, task not completed")
        return {
            "status": "timeout",
            "rounds": round_num,
            "final_dag": current_dag,
            "tool_results": self.tool_results_history
        }
