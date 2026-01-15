"""Abstract base class for workflow planners.

Planners create manifests that define the complete analysis workflow,
including chunk specifications and the merge DAG.

Key Requirements:
- Merge DAG must be acyclic
- Merge nodes have bounded fan-in (8-20 inputs recommended)
- Root merge node produces final documentation
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from atlas.models.artifact import Artifact
from atlas.models.manifest import (
    Manifest,
    ChunkSpec,
    MergeNode,
    AnalysisProfile,
    SplitterProfile,
    ReviewPolicy,
    ArtifactOutputConfig,
)
from atlas.splitter.base import Splitter


@dataclass
class PlanResult:
    """Result of workflow planning.

    Attributes:
        manifest: The generated manifest.
        chunk_count: Number of chunks created.
        merge_levels: Number of merge levels in the DAG.
        estimated_total_tokens: Total estimated tokens across chunks.
        warnings: Any warnings during planning.
    """

    manifest: Manifest
    chunk_count: int = 0
    merge_levels: int = 0
    estimated_total_tokens: int = 0
    warnings: list[str] = field(default_factory=list)


class Planner(ABC):
    """Abstract interface for workflow planning.

    Planners coordinate between the splitter (which creates chunks)
    and the controller (which executes the plan). They produce a
    manifest that describes the complete workflow.

    Design Principles:
        - Merge nodes MUST have bounded fan-in (suggested: 8-20 inputs)
        - Merge DAG MUST be a DAG (no cycles)
        - Root merge node produces the final doc + doc model
        - Include merge nodes for large subdivisions so challenger
          can target those units later

    Example Implementation:
        >>> class HierarchicalPlanner(Planner):
        ...     def plan(self, artifact, source, profile) -> PlanResult:
        ...         # Split source into chunks
        ...         split_result = self.splitter.split(source, profile.splitter)
        ...         # Build merge DAG
        ...         merge_dag = self._build_merge_dag(split_result.chunks)
        ...         # Create manifest
        ...         manifest = Manifest(...)
        ...         return PlanResult(manifest=manifest)

    TODO: Implement concrete planners for:
        - Hierarchical merge DAG (balanced tree)
        - Division-based merge (COBOL-specific)
        - Flat merge (small files)
    """

    def __init__(
        self,
        splitter: Splitter,
        max_merge_fan_in: int = 15,
    ):
        """Initialize the planner.

        Args:
            splitter: Splitter instance for creating chunks.
            max_merge_fan_in: Maximum inputs per merge node.
        """
        self.splitter = splitter
        self.max_merge_fan_in = max_merge_fan_in

    @abstractmethod
    def plan(
        self,
        artifact: Artifact,
        source: str,
        analysis_profile: AnalysisProfile,
        splitter_profile: SplitterProfile,
        context_budget: int,
        output_config: ArtifactOutputConfig,
        review_policy: ReviewPolicy | None = None,
    ) -> PlanResult:
        """Create a workflow plan for analyzing an artifact.

        Args:
            artifact: The source artifact to analyze.
            source: Source code content.
            analysis_profile: Documentation extraction profile.
            splitter_profile: Chunking configuration.
            context_budget: Maximum tokens per task.
            output_config: Where to write outputs.
            review_policy: Optional challenger configuration.

        Returns:
            PlanResult with the complete manifest.

        TODO: Implement workflow planning logic.
        """
        pass

    @abstractmethod
    def build_merge_dag(
        self,
        chunks: list[ChunkSpec],
    ) -> list[MergeNode]:
        """Build a merge DAG from chunks.

        Creates a hierarchical merge structure with bounded fan-in
        at each level.

        Args:
            chunks: List of chunk specifications.

        Returns:
            List of merge nodes forming a DAG.

        Design Requirements:
            - Fan-in at each node <= max_merge_fan_in
            - Exactly one root node (is_root=True)
            - No cycles in the DAG

        TODO: Implement DAG construction.
        """
        pass

    def validate_dag(self, merge_nodes: list[MergeNode]) -> list[str]:
        """Validate the merge DAG.

        Checks for:
        - Exactly one root node
        - No cycles
        - All references are valid
        - Fan-in constraints

        Args:
            merge_nodes: The merge DAG to validate.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors: list[str] = []

        # Check for root node
        root_nodes = [n for n in merge_nodes if n.is_root]
        if len(root_nodes) == 0:
            errors.append("No root merge node found")
        elif len(root_nodes) > 1:
            ids = [n.merge_node_id for n in root_nodes]
            errors.append(f"Multiple root merge nodes: {ids}")

        # Check fan-in constraints
        for node in merge_nodes:
            if len(node.input_ids) > self.max_merge_fan_in:
                errors.append(
                    f"Merge node {node.merge_node_id} exceeds fan-in limit: "
                    f"{len(node.input_ids)} > {self.max_merge_fan_in}"
                )

        # TODO: Implement cycle detection
        # TODO: Validate all input_ids reference valid chunks or merge nodes

        return errors

    def compute_merge_levels(self, merge_nodes: list[MergeNode]) -> int:
        """Compute the number of levels in the merge DAG.

        Args:
            merge_nodes: The merge DAG.

        Returns:
            Number of merge levels (depth of DAG).
        """
        if not merge_nodes:
            return 0

        return max(node.level for node in merge_nodes) + 1

    def assign_output_uris(
        self,
        chunks: list[ChunkSpec],
        merge_nodes: list[MergeNode],
        output_config: ArtifactOutputConfig,
    ) -> None:
        """Assign output URIs to chunks and merge nodes.

        Mutates the chunks and merge_nodes in place.

        Args:
            chunks: Chunk specifications.
            merge_nodes: Merge DAG nodes.
            output_config: Output configuration with URI templates.
        """
        for chunk in chunks:
            chunk.result_uri = (
                f"{output_config.base_uri}/"
                f"{output_config.chunk_results_path.format(chunk_id=chunk.chunk_id)}"
            )

        for node in merge_nodes:
            node.output_uri = (
                f"{output_config.base_uri}/"
                f"{output_config.merge_results_path.format(merge_node_id=node.merge_node_id)}"
            )
