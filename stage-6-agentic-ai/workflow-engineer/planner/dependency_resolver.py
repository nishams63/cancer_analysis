"""DAG Dependency Resolver, Cycle Detector, and Topological Sorter."""
from __future__ import annotations
from typing import List, Dict, Set, Tuple, Optional
from schemas.task import Task


class DependencyResolver:
    """Validates and orders task graphs using Directed Acyclic Graph (DAG) algorithms."""

    @staticmethod
    def build_adjacency_list(tasks: List[Task]) -> Dict[str, List[str]]:
        """Map task_id -> list of successor task_ids."""
        adj: Dict[str, List[str]] = {t.task_id: [] for t in tasks}
        for task in tasks:
            for dep in task.dependencies:
                if dep in adj:
                    adj[dep].append(task.task_id)
        return adj

    @classmethod
    def detect_cycles(cls, tasks: List[Task]) -> Tuple[bool, Optional[str]]:
        """Detect circular dependencies using DFS graph coloring.
        
        Colors: 0 = unvisited (white), 1 = visiting (gray), 2 = visited (black).
        Returns (has_cycle, cycle_description).
        """
        task_ids = {t.task_id for t in tasks}
        # Verify all dependencies reference valid tasks
        for t in tasks:
            for dep in t.dependencies:
                if dep not in task_ids:
                    return True, f"Task '{t.task_id}' references non-existent dependency '{dep}'"

        # Build parent -> child adjacency
        adj = cls.build_adjacency_list(tasks)
        state: Dict[str, int] = {t_id: 0 for t_id in task_ids}
        cycle_path: List[str] = []

        def dfs(node: str, path: List[str]) -> bool:
            state[node] = 1  # gray (in current call stack)
            path.append(node)
            for neighbor in adj.get(node, []):
                if state[neighbor] == 1:
                    cycle_path.extend(path[path.index(neighbor):] + [neighbor])
                    return True
                elif state[neighbor] == 0:
                    if dfs(neighbor, path):
                        return True
            path.pop()
            state[node] = 2  # black (done)
            return False

        for t_id in task_ids:
            if state[t_id] == 0:
                if dfs(t_id, []):
                    return True, f"Circular dependency detected: {' -> '.join(cycle_path)}"

        return False, None

    @classmethod
    def topological_sort(cls, tasks: List[Task]) -> List[str]:
        """Compute valid deterministic linear execution order using Kahn's algorithm."""
        has_cycle, err = cls.detect_cycles(tasks)
        if has_cycle:
            raise ValueError(f"Cannot topologically sort graph with cycles: {err}")

        task_map = {t.task_id: t for t in tasks}
        # In-degree = count of dependencies
        in_degree: Dict[str, int] = {t.task_id: len(t.dependencies) for t in tasks}
        adj = cls.build_adjacency_list(tasks)

        # Start with all nodes with in-degree 0 (roots), sorted deterministically
        queue = sorted([t_id for t_id, deg in in_degree.items() if deg == 0])
        ordered: List[str] = []

        while queue:
            curr = queue.pop(0)
            ordered.append(curr)

            for child in sorted(adj.get(curr, [])):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
            queue.sort()

        if len(ordered) != len(tasks):
            raise ValueError("Topological sort failed: graph contains disconnected or cyclical components")

        return ordered

    @classmethod
    def find_unreachable_tasks(cls, tasks: List[Task], entry_task_id: str) -> List[str]:
        """Identify tasks that cannot be reached starting from the entry task."""
        adj = cls.build_adjacency_list(tasks)
        visited: Set[str] = set()

        def dfs(node: str):
            visited.add(node)
            for child in adj.get(node, []):
                if child not in visited:
                    dfs(child)

        if any(t.task_id == entry_task_id for t in tasks):
            dfs(entry_task_id)

        all_ids = {t.task_id for t in tasks}
        return sorted(list(all_ids - visited))
