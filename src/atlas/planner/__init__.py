"""DAG planner interfaces for workflow planning.

The planner creates manifests that define the complete workflow:
- Chunk specifications from the splitter
- Merge DAG with bounded fan-in
- Review policies for the challenger

Design Principles:
- Merge nodes have bounded fan-in (8-20 inputs)
- Merge DAG is acyclic (no cycles)
- Root merge produces final documentation
"""

from atlas.planner.base import Planner, PlanResult
from atlas.planner.dag_planner import DAGPlanner

__all__ = [
    "Planner",
    "PlanResult",
    "DAGPlanner",
]
