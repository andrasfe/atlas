"""DAG Planner for hierarchical merge tree construction.

This planner creates merge DAGs from chunks that respect context limits
and provide bounded fan-in at each merge level. The merge tree enables
efficient hierarchical aggregation of chunk analysis results.

Key Features:
- Configurable fan-in (default 4-8 chunks per merge)
- Support for semantic grouping (e.g., by COBOL division)
- Balanced tree construction for optimal merge depth
- Validation of DAG properties (acyclic, single root, bounded fan-in)
"""

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

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
from atlas.models.enums import ChunkKind
from atlas.planner.base import Planner, PlanResult
from atlas.splitter.base import Splitter


@dataclass
class MergeGroup:
    """A group of items to merge at a particular level.

    Attributes:
        group_id: Identifier for this group.
        input_ids: IDs of chunks or merge nodes to combine.
        group_kind: The kind of content in this group.
        division: COBOL division if semantically grouped.
    """

    group_id: str
    input_ids: list[str]
    group_kind: ChunkKind | None = None
    division: str | None = None


class DAGPlanner(Planner):
    """Hierarchical DAG planner for merge tree construction.

    This planner creates a merge DAG that:
    - Has bounded fan-in at each level (configurable)
    - Groups chunks semantically when possible (by division)
    - Produces a balanced tree for optimal depth
    - Includes a single root merge node

    Design Requirements from Spec:
        - Merge nodes MUST have bounded fan-in (suggested: 8-20 inputs)
        - Merge DAG MUST be a DAG (no cycles)
        - Root merge node produces the final doc + doc model
        - Include merge nodes for large subdivisions so challenger
          can target those units later

    Example:
        >>> splitter = COBOLSplitter()
        >>> planner = DAGPlanner(splitter, min_fan_in=4, max_fan_in=8)
        >>> result = planner.plan(artifact, source, profile, ...)
        >>> print(f"Created {result.chunk_count} chunks, {result.merge_levels} merge levels")

    Attributes:
        splitter: The splitter instance for creating chunks.
        min_fan_in: Minimum inputs per merge node (default 2).
        max_fan_in: Maximum inputs per merge node (default 8).
        prefer_semantic_grouping: Group by semantic boundaries first.
    """

    def __init__(
        self,
        splitter: Splitter,
        min_fan_in: int = 2,
        max_fan_in: int = 8,
        prefer_semantic_grouping: bool = True,
    ) -> None:
        """Initialize the DAG planner.

        Args:
            splitter: Splitter instance for creating chunks.
            min_fan_in: Minimum inputs per merge node.
            max_fan_in: Maximum inputs per merge node.
            prefer_semantic_grouping: Group by semantic boundaries first.
        """
        super().__init__(splitter, max_fan_in)
        self.min_fan_in = min_fan_in
        self.prefer_semantic_grouping = prefer_semantic_grouping

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

        This method:
        1. Splits the source into chunks using the splitter
        2. Builds a merge DAG from the chunks
        3. Assigns output URIs to all nodes
        4. Creates and validates the manifest

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
        """
        warnings: list[str] = []

        # Split source into chunks
        split_result = self.splitter.split(
            source,
            splitter_profile,
            artifact.artifact_id,
        )

        chunks = split_result.chunks
        warnings.extend(split_result.warnings)

        # Handle edge cases
        if not chunks:
            warnings.append("No chunks created from source")

        # Build merge DAG
        merge_dag = self.build_merge_dag(chunks)

        # Validate DAG
        dag_errors = self.validate_dag(merge_dag)
        if dag_errors:
            warnings.extend(dag_errors)

        # Assign output URIs
        self.assign_output_uris(chunks, merge_dag, output_config)

        # Compute merge levels
        merge_levels = self.compute_merge_levels(merge_dag)

        # Create manifest
        manifest = Manifest(
            job_id=f"job-{artifact.artifact_id}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
            artifact_ref=artifact.to_ref(),
            analysis_profile=analysis_profile,
            splitter_profile=splitter_profile,
            context_budget=context_budget,
            chunks=chunks,
            merge_dag=merge_dag,
            review_policy=review_policy or ReviewPolicy(),
            artifacts=output_config,
            cycle_number=1,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        return PlanResult(
            manifest=manifest,
            chunk_count=len(chunks),
            merge_levels=merge_levels,
            estimated_total_tokens=split_result.total_estimated_tokens,
            warnings=warnings,
        )

    def build_merge_dag(
        self,
        chunks: list[ChunkSpec],
    ) -> list[MergeNode]:
        """Build a merge DAG from chunks.

        Creates a hierarchical merge structure with bounded fan-in
        at each level. If semantic grouping is enabled, groups chunks
        by division/section first before building the tree.

        Args:
            chunks: List of chunk specifications.

        Returns:
            List of merge nodes forming a DAG.
        """
        if not chunks:
            return []

        # Single chunk - create a trivial root merge
        if len(chunks) == 1:
            return [
                MergeNode(
                    merge_node_id="merge_root",
                    input_ids=[chunks[0].chunk_id],
                    is_root=True,
                    level=0,
                )
            ]

        # Group chunks semantically if enabled
        if self.prefer_semantic_grouping:
            groups = self._group_chunks_semantically(chunks)
        else:
            groups = [MergeGroup(
                group_id="all",
                input_ids=[c.chunk_id for c in chunks],
            )]

        merge_nodes: list[MergeNode] = []
        current_level = 0

        # Process each semantic group
        level_inputs: list[str] = []

        for group in groups:
            if len(group.input_ids) == 1:
                # Single chunk in group, pass through to next level
                level_inputs.extend(group.input_ids)
            else:
                # Create merge nodes for this group
                group_merges = self._create_balanced_merges(
                    group.input_ids,
                    current_level,
                    prefix=group.group_id,
                    division=group.division,
                )
                merge_nodes.extend(group_merges)

                # Find the output of this group (the highest level merge)
                if group_merges:
                    # Get all output IDs from this group for the next level
                    group_outputs = self._get_level_outputs(group_merges)
                    level_inputs.extend(group_outputs)

        # If we have no merge nodes yet, create them from raw chunks
        if not merge_nodes and level_inputs:
            # level_inputs contains chunk IDs
            pass
        elif not merge_nodes:
            # Create merges directly from chunks
            level_inputs = [c.chunk_id for c in chunks]

        # Now build upper levels until we have a single root
        if not merge_nodes:
            # Start fresh with chunk IDs
            current_ids = [c.chunk_id for c in chunks]
            current_level = 0
        else:
            # Continue from where semantic grouping left off
            current_ids = level_inputs
            current_level = max(n.level for n in merge_nodes) + 1 if merge_nodes else 0

        # Build tree until single root
        while len(current_ids) > 1:
            next_level_nodes = self._create_level_merges(
                current_ids,
                current_level,
            )
            merge_nodes.extend(next_level_nodes)
            current_ids = [n.merge_node_id for n in next_level_nodes]
            current_level += 1

        # Mark the root node
        if merge_nodes:
            # Find the node(s) at the highest level
            max_level = max(n.level for n in merge_nodes)
            root_candidates = [n for n in merge_nodes if n.level == max_level]

            if len(root_candidates) == 1:
                root_candidates[0].is_root = True
            else:
                # Multiple nodes at top level, create a final root merge
                root_node = MergeNode(
                    merge_node_id="merge_root",
                    input_ids=[n.merge_node_id for n in root_candidates],
                    is_root=True,
                    level=max_level + 1,
                )
                merge_nodes.append(root_node)

        return merge_nodes

    def _group_chunks_semantically(
        self,
        chunks: list[ChunkSpec],
    ) -> list[MergeGroup]:
        """Group chunks by semantic boundaries (division/section).

        Args:
            chunks: List of chunk specifications.

        Returns:
            List of merge groups, one per semantic unit.
        """
        # Group by division
        division_groups: dict[str, list[str]] = {}

        for chunk in chunks:
            division = chunk.division or "other"
            if division not in division_groups:
                division_groups[division] = []
            division_groups[division].append(chunk.chunk_id)

        # Create MergeGroup for each division
        groups: list[MergeGroup] = []

        # Define a stable ordering for divisions
        division_order = [
            "IDENTIFICATION",
            "ENVIRONMENT",
            "DATA",
            "PROCEDURE",
            "other",
        ]

        for division in division_order:
            if division in division_groups:
                group = MergeGroup(
                    group_id=f"merge_{division.lower()}",
                    input_ids=division_groups[division],
                    division=division,
                )
                groups.append(group)

        # Add any remaining divisions not in the standard order
        for division, chunk_ids in division_groups.items():
            if division not in division_order:
                group = MergeGroup(
                    group_id=f"merge_{division.lower()}",
                    input_ids=chunk_ids,
                    division=division,
                )
                groups.append(group)

        return groups

    def _create_balanced_merges(
        self,
        input_ids: list[str],
        start_level: int,
        prefix: str = "merge",
        division: str | None = None,
    ) -> list[MergeNode]:
        """Create a balanced tree of merge nodes for inputs.

        Args:
            input_ids: IDs to merge (chunks or child merges).
            start_level: Starting level for merge nodes.
            prefix: Prefix for merge node IDs.
            division: Optional division name for metadata.

        Returns:
            List of merge nodes forming a balanced subtree.
        """
        if len(input_ids) <= 1:
            return []

        merge_nodes: list[MergeNode] = []
        current_ids = input_ids
        current_level = start_level
        merge_counter = 0

        while len(current_ids) > 1:
            next_ids: list[str] = []

            # Split current IDs into groups of max_fan_in
            for i in range(0, len(current_ids), self.max_merge_fan_in):
                group = current_ids[i:i + self.max_merge_fan_in]

                # If we have a small leftover group, try to balance
                if len(group) < self.min_fan_in and next_ids:
                    # Merge with previous group if possible
                    prev_node = merge_nodes[-1]
                    if len(prev_node.input_ids) + len(group) <= self.max_merge_fan_in:
                        prev_node.input_ids.extend(group)
                        continue

                merge_counter += 1
                node_id = f"{prefix}_level{current_level}_{merge_counter:03d}"

                node = MergeNode(
                    merge_node_id=node_id,
                    input_ids=group,
                    is_root=False,
                    level=current_level,
                    metadata={"division": division} if division else {},
                )
                merge_nodes.append(node)
                next_ids.append(node_id)

            current_ids = next_ids
            current_level += 1

        return merge_nodes

    def _create_level_merges(
        self,
        input_ids: list[str],
        level: int,
    ) -> list[MergeNode]:
        """Create merge nodes for a single level.

        Args:
            input_ids: IDs to merge at this level.
            level: The merge level.

        Returns:
            List of merge nodes for this level.
        """
        if len(input_ids) <= 1:
            # Single input, no merge needed
            return []

        merge_nodes: list[MergeNode] = []
        merge_counter = 0

        # Calculate optimal group size for balanced tree
        num_groups = math.ceil(len(input_ids) / self.max_merge_fan_in)
        base_size = len(input_ids) // num_groups
        remainder = len(input_ids) % num_groups

        idx = 0
        for group_idx in range(num_groups):
            # Distribute remainder across first groups
            group_size = base_size + (1 if group_idx < remainder else 0)
            group = input_ids[idx:idx + group_size]
            idx += group_size

            if len(group) > 0:
                merge_counter += 1
                node_id = f"merge_level{level}_{merge_counter:03d}"

                node = MergeNode(
                    merge_node_id=node_id,
                    input_ids=group,
                    is_root=False,
                    level=level,
                )
                merge_nodes.append(node)

        return merge_nodes

    def _get_level_outputs(
        self,
        merge_nodes: list[MergeNode],
    ) -> list[str]:
        """Get the output IDs from the highest level in a node list.

        Args:
            merge_nodes: List of merge nodes.

        Returns:
            IDs of nodes at the highest level.
        """
        if not merge_nodes:
            return []

        max_level = max(n.level for n in merge_nodes)
        return [n.merge_node_id for n in merge_nodes if n.level == max_level]

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
        errors = super().validate_dag(merge_nodes)

        if not merge_nodes:
            return errors

        # Build node lookup
        node_ids = {n.merge_node_id for n in merge_nodes}

        # Check for cycles using DFS
        visited: set[str] = set()
        rec_stack: set[str] = set()

        # Build reverse adjacency (parent -> children)
        children: dict[str, list[str]] = {n.merge_node_id: [] for n in merge_nodes}
        for node in merge_nodes:
            for input_id in node.input_ids:
                if input_id in children:
                    children[input_id].append(node.merge_node_id)

        def has_cycle(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)

            for child in children.get(node_id, []):
                if child not in visited:
                    if has_cycle(child):
                        return True
                elif child in rec_stack:
                    return True

            rec_stack.remove(node_id)
            return False

        for node in merge_nodes:
            if node.merge_node_id not in visited:
                if has_cycle(node.merge_node_id):
                    errors.append("Cycle detected in merge DAG")
                    break

        # Validate input references exist (either as chunks or merge nodes)
        # Note: We can't validate chunk references here without chunk list
        for node in merge_nodes:
            for input_id in node.input_ids:
                # Input should be either a chunk or another merge node
                # Merge node references are validated here
                if input_id.startswith("merge_") and input_id not in node_ids:
                    errors.append(
                        f"Merge node {node.merge_node_id} references "
                        f"non-existent merge node: {input_id}"
                    )

        return errors

    def estimate_merge_tokens(
        self,
        input_count: int,
        avg_input_tokens: int = 500,
    ) -> int:
        """Estimate tokens needed for a merge operation.

        Args:
            input_count: Number of inputs to merge.
            avg_input_tokens: Average tokens per input.

        Returns:
            Estimated token count for the merge.
        """
        # Base overhead for merge prompt
        overhead = 200

        # Each input contributes its summary/facts
        input_tokens = input_count * avg_input_tokens

        return overhead + input_tokens
