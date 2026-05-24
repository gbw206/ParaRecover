import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def convert_status(dag):
    if isinstance(dag, dict):
        for key, value in dag.items():
            if key == "status":
                if isinstance(value, str):
                    dag[key] = value.lower() == "true"
            elif isinstance(value, (dict, list)):
                convert_status(value)
    elif isinstance(dag, list):
        for item in dag:
            convert_status(item)
    return dag


def draw_dag(dag_json, output_path=None):
    if isinstance(dag_json, str):
        dag = json.loads(dag_json)
    else:
        dag = dag_json

    dag = convert_status(dag)
    tasks = dag.get("tasks", [])
    if not tasks:
        print("No tasks found in DAG")
        return

    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    node_info = {}
    for task in tasks:
        node_info[task.get("id", "")] = task

    def get_level(node_id, visited=None):
        if visited is None:
            visited = set()
        if node_id in visited:
            return 0
        visited.add(node_id)
        node = node_info.get(node_id)
        if not node:
            return 0
        deps = node.get("dep", [])
        if not deps or node_id == "root":
            return 0
        max_dep_level = 0
        for dep in deps:
            dep_level = get_level(dep, visited.copy())
            max_dep_level = max(max_dep_level, dep_level)
        return max_dep_level + 1

    levels = {}
    for task in tasks:
        task_id = task.get("id", "")
        level = get_level(task_id)
        if level not in levels:
            levels[level] = []
        levels[level].append(task_id)

    positions = {}
    node_radius = 0.4

    for level, node_ids in levels.items():
        n = len(node_ids)
        y = 10 - 1.5 - level * 1.5
        for i, node_id in enumerate(node_ids):
            x = 1 + i * (9 / max(n, 1))
            positions[node_id] = (x, y)

    for task in tasks:
        task_id = task.get("id", "")
        name = task.get("name", "")
        status = task.get("status", False)

        if task_id not in positions:
            continue
        x, y = positions[task_id]

        if task_id == "root":
            color = '#3498db'
        elif task_id == "response":
            color = '#e74c3c'
        elif status:
            color = '#2ecc71'
        else:
            color = '#95a5a6'

        circle = plt.Circle((x, y), node_radius, color=color, ec='black', linewidth=2, alpha=0.8)
        ax.add_patch(circle)
        ax.text(x, y, task_id, ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        if name:
            ax.text(x, y - node_radius - 0.2, name, ha='center', va='top', fontsize=8, color='black')

    for task in tasks:
        task_id = task.get("id", "")
        deps = task.get("dep", [])
        if task_id not in positions:
            continue
        x1, y1 = positions[task_id]
        for dep in deps:
            if dep not in positions:
                continue
            x2, y2 = positions[dep]
            dx = x2 - x1
            dy = y2 - y1
            length = np.sqrt(dx ** 2 + dy ** 2)
            if length > 0:
                start_x = x1 + (dx / length) * node_radius
                start_y = y1 + (dy / length) * node_radius
                end_x = x2 - (dx / length) * (node_radius + 0.1)
                end_y = y2 - (dy / length) * (node_radius + 0.1)
                ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                           arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    legend_elements = [
        mpatches.Patch(color='#3498db', label='Root'),
        mpatches.Patch(color='#2ecc71', label='Completed'),
        mpatches.Patch(color='#95a5a6', label='Pending'),
        mpatches.Patch(color='#e74c3c', label='Response'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    ax.set_title('DAG Task Flow', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved to: {output_path}")
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Draw DAG visualization")
    parser.add_argument("-i", "--input", help="JSON file with DAG data")
    parser.add_argument("-o", "--output", default="dag_output.png", help="Output image path")
    args = parser.parse_args()

    if args.input:
        with open(args.input, 'r', encoding='utf-8') as f:
            dag_data = json.load(f)
    else:
        dag_data = {"tasks": [
            {"id": "root", "desc": "Sample query", "dep": [], "status": True},
            {"id": "1", "name": "search", "para": {"q": "test"}, "desc": "Search", "dep": ["root"], "status": False},
            {"id": "response", "desc": "Respond", "dep": ["1"], "status": False}
        ]}

    draw_dag(dag_data, args.output)
