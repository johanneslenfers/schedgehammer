from __future__ import annotations

from dataclasses import dataclass, field
from statistics import fmean, stdev
from typing import Any


@dataclass
class MetaTreeNode:
    children: dict[str, MetaTreeNode]
    operation: Any | None
    costs: list[float]
    parent: MetaTreeNode | None

    def get_avg_cost(self):
        # If several schedules have a good cost it counts twice as much
        return fmean(self.costs) / 2 if len(self.costs) > 1 else fmean(self.costs)

    def __str__(self):
        stats = f"({self.operation.name}|Len:{len(self.costs)}|Min:\033[92m{min(self.costs):.4f}\033[0m"
        if len(self.costs) >= 2:
            stats += f"|StdDev:\033[91m{stdev(self.costs):.4f}\033[0m"
        stats += f"|Avg:\033[93m{fmean(self.costs):.4f}\033[0m)"
        return stats

    def get_geneaology(self) -> list[Any]:
        tree_node = self
        geneaology = []
        while tree_node.operation:
            geneaology.insert(0, tree_node.operation)
            tree_node = tree_node.parent
        return geneaology


@dataclass
class MetaTree:
    param: Any
    root: MetaTreeNode = field(default_factory=lambda: MetaTreeNode({}, None, [], None))

    def __str__(self):
        def print_node(node, prefix="", is_last=True):
            result = []
            if node.operation:
                result.append(f"{prefix}{'└── ' if is_last else '├── '}{str(node)}")
            else:
                result.append(f"{prefix}root")

            children = list(node.children.items())
            for i, (child_name, child) in enumerate(children):
                is_last_child = i == len(children) - 1
                child_prefix = prefix + ("    " if is_last else "│   ")
                result.extend(print_node(child, child_prefix, is_last_child))
            return result

        return "\n".join(print_node(self.root))

    def get_n_best_children(self, n: int):
        # Use BFS to find n best nodes in the tree
        best_nodes = []
        queue = [(self.root, 0)]  # (node, depth) pairs

        while queue:
            node, depth = queue.pop(0)

            # Add all children to queue
            for child_name, child_node in node.children.items():
                queue.append((child_node, depth + 1))

            # Skip root node since it's not an operation
            if node.operation:
                best_nodes.append((node, node.get_avg_cost()))

        # Sort by average cost and return top n nodes
        best_nodes.sort(key=lambda x: x[1])
        return [node for node, _ in best_nodes[:n]]
