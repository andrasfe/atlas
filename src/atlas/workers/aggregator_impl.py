"""Aggregator worker implementation for merging chunk results.

The Aggregator merges chunk results or child merges into higher-level
summaries, building up the documentation hierarchically.

This implementation:
- Merges facts from multiple inputs (symbols, calls, I/O, error handling)
- Identifies conflicts and inconsistencies
- Produces consolidated documentation sections
- Tracks provenance (which inputs contributed to each fact)
"""

import logging
from typing import Any

from atlas.adapters.artifact_store import ArtifactStoreAdapter
from atlas.adapters.llm import LLMAdapter, LLMMessage, LLMRole
from atlas.adapters.ticket_system import TicketSystemAdapter
from atlas.models.results import (
    CallTarget,
    ConsolidatedFacts,
    DocumentationModel,
    DocIndex,
    ErrorHandlingPattern,
    IOOperation,
    MergeConflict,
    MergeCoverage,
    MergeResult,
    Section,
    SymbolDef,
)
from atlas.models.work_item import DocMergePayload
from atlas.workers.aggregator import Aggregator

logger = logging.getLogger(__name__)


MERGE_SYSTEM_PROMPT = """You are an expert documentation writer merging analysis results from multiple code chunks.

You will be given structured analysis results from several chunks and must:
1. Consolidate the facts into a unified view
2. Identify any conflicts or inconsistencies between chunks
3. Generate narrative documentation sections

Produce valid JSON with this structure:
{
  "consolidated_facts": {
    "call_graph_edges": [{"from": "...", "to": "...", "type": "...", "source_chunk": "..."}],
    "io_map": [{"operation": "...", "file_name": "...", "record_name": "...", "source_chunk": "..."}],
    "error_handling_behaviors": [{"pattern_type": "...", "description": "...", "source_chunks": [...]}],
    "symbols": [{"name": "...", "kind": "...", "attributes": {...}, "source_chunk": "..."}]
  },
  "conflicts": [
    {"description": "...", "input_ids": ["chunk1", "chunk2"], "suggested_followup_scope": "..."}
  ],
  "narrative_sections": [
    {"section_id": "...", "title": "...", "content": "...", "source_refs": ["chunk1", "chunk2"]}
  ]
}

Guidelines:
- Merge ALL symbols, calls, I/O operations, and error handling from inputs
- Track which chunk each fact came from (provenance)
- Identify disagreements between chunks (e.g., different descriptions of same symbol)
- Generate clear narrative sections with proper source references
- Consolidate call graph showing how paragraphs relate to each other
"""

MERGE_USER_PROMPT = """Merge the following {input_count} analysis results for {artifact_id}.

Merge Node: {merge_node_id}
Is Root: {is_root}

---INPUTS START---
{inputs_json}
---INPUTS END---

Consolidate all facts, identify conflicts, and generate documentation sections.
"""

PATCH_MERGE_SYSTEM_PROMPT = """You are an expert documentation editor applying updates from follow-up analysis.

You will be given:
1. Base documentation
2. Follow-up answers addressing specific issues

Update the documentation to incorporate the new information while preserving the overall structure.

Produce valid JSON with:
{
  "updated_content": "Full updated documentation text",
  "sections_modified": ["section_id_1", "section_id_2"],
  "changes_summary": "Brief description of what was updated"
}
"""

PATCH_MERGE_USER_PROMPT = """Apply the following follow-up answers to update the documentation.

---BASE DOCUMENTATION---
{base_doc}
---END BASE DOCUMENTATION---

---FOLLOW-UP ANSWERS---
{answers_json}
---END FOLLOW-UP ANSWERS---

Update the documentation to address each follow-up answer.
"""


class AggregatorWorker(Aggregator):
    """Concrete implementation of the Aggregator worker.

    Merges chunk results or child merges into consolidated summaries,
    tracking provenance and identifying conflicts.

    Design Principles:
        - Merge structured facts from inputs
        - Identify and record conflicts between inputs
        - Produce consolidated documentation sections
        - Track coverage (what was included vs. missing)
    """

    def __init__(
        self,
        worker_id: str,
        ticket_system: TicketSystemAdapter,
        artifact_store: ArtifactStoreAdapter,
        llm: LLMAdapter,
    ):
        """Initialize the aggregator worker.

        Args:
            worker_id: Unique identifier for this worker instance.
            ticket_system: Ticket system adapter.
            artifact_store: Artifact store adapter.
            llm: LLM adapter for generating narrative.
        """
        super().__init__(worker_id, ticket_system, artifact_store, llm)

    async def merge_results(
        self,
        inputs: list[dict[str, Any]],
        payload: DocMergePayload,
        manifest: dict[str, Any],
    ) -> MergeResult:
        """Merge multiple input results.

        Consolidates facts from chunk results or child merges into
        a unified view with conflict detection.

        Args:
            inputs: List of input result artifacts.
            payload: Merge work item payload.
            manifest: The workflow manifest.

        Returns:
            MergeResult with consolidated facts.
        """
        logger.debug(f"Merging {len(inputs)} inputs for node {payload.merge_node_id}")

        # Determine which inputs are present vs missing
        included_ids: list[str] = []
        missing_ids: list[str] = []

        for input_data in inputs:
            if input_data:
                chunk_id = input_data.get("chunk_id") or input_data.get("merge_node_id", "unknown")
                included_ids.append(chunk_id)

        # Check expected inputs from manifest
        merge_node = None
        if manifest and "merge_dag" in manifest:
            for node in manifest["merge_dag"]:
                if node.get("merge_node_id") == payload.merge_node_id:
                    merge_node = node
                    break

        if merge_node:
            expected_ids = set(merge_node.get("input_ids", []))
            missing_ids = list(expected_ids - set(included_ids))

        # Determine if this is the root merge
        is_root = merge_node.get("is_root", False) if merge_node else False

        # Merge the facts programmatically first
        consolidated = self._merge_facts(inputs)

        # Use LLM to generate narrative and detect conflicts
        artifact_id = manifest.get("artifact_ref", {}).get("artifact_id", "unknown") if manifest else "unknown"

        import json
        inputs_json = json.dumps(inputs, indent=2, default=str)

        user_prompt = MERGE_USER_PROMPT.format(
            input_count=len(inputs),
            artifact_id=artifact_id,
            merge_node_id=payload.merge_node_id,
            is_root=is_root,
            inputs_json=inputs_json[:10000],  # Truncate if too large
        )

        messages = [
            LLMMessage(role=LLMRole.SYSTEM, content=MERGE_SYSTEM_PROMPT),
            LLMMessage(role=LLMRole.USER, content=user_prompt),
        ]

        try:
            response = await self.llm.complete_json(messages)
            conflicts = self._parse_conflicts(response.get("conflicts", []))
            narrative_sections = response.get("narrative_sections", [])
        except Exception as e:
            logger.error(f"LLM merge narrative failed: {e}")
            conflicts = []
            narrative_sections = []

        # Get artifact version from manifest
        artifact_version = ""
        if manifest:
            artifact_ref = manifest.get("artifact_ref", {})
            artifact_version = artifact_ref.get("artifact_version", "")

        return MergeResult(
            job_id=payload.job_id,
            artifact_id=artifact_id,
            artifact_version=artifact_version,
            merge_node_id=payload.merge_node_id,
            coverage=MergeCoverage(
                included_input_ids=included_ids,
                missing_input_ids=missing_ids,
            ),
            consolidated_facts=consolidated,
            conflicts=conflicts,
            narrative_sections=narrative_sections,
        )

    async def apply_patches(
        self,
        base_doc: str,
        base_model: DocumentationModel,
        answers: list[dict[str, Any]],
    ) -> tuple[str, DocumentationModel]:
        """Apply follow-up answers to documentation.

        Updates the documentation to incorporate new information from
        follow-up investigations while preserving structure.

        Args:
            base_doc: The base documentation text.
            base_model: The base documentation model.
            answers: List of follow-up answer artifacts.

        Returns:
            Tuple of (updated doc, updated model).
        """
        logger.debug(f"Applying {len(answers)} patches to documentation")

        import json
        answers_json = json.dumps(answers, indent=2, default=str)

        user_prompt = PATCH_MERGE_USER_PROMPT.format(
            base_doc=base_doc[:8000],  # Truncate if needed
            answers_json=answers_json[:4000],
        )

        messages = [
            LLMMessage(role=LLMRole.SYSTEM, content=PATCH_MERGE_SYSTEM_PROMPT),
            LLMMessage(role=LLMRole.USER, content=user_prompt),
        ]

        try:
            response = await self.llm.complete_json(messages)
            updated_content = response.get("updated_content", base_doc)
            sections_modified = response.get("sections_modified", [])
            changes_summary = response.get("changes_summary", "")
        except Exception as e:
            logger.error(f"LLM patch merge failed: {e}")
            # Fall back to simple concatenation
            updated_content = base_doc
            for answer in answers:
                updated_content += f"\n\n## Follow-up: {answer.get('issue_id', 'unknown')}\n"
                updated_content += answer.get("answer", "")
            sections_modified = []
            changes_summary = f"Error in patch merge: {e}"

        # Update the documentation model
        updated_model = self._update_doc_model(
            base_model, answers, sections_modified, changes_summary
        )

        return updated_content, updated_model

    def _merge_facts(self, inputs: list[dict[str, Any]]) -> ConsolidatedFacts:
        """Programmatically merge facts from inputs.

        Args:
            inputs: List of input result artifacts.

        Returns:
            ConsolidatedFacts with all merged data.
        """
        call_graph_edges: list[dict[str, Any]] = []
        io_map: list[IOOperation] = []
        error_handling_behaviors: list[ErrorHandlingPattern] = []
        symbols: list[SymbolDef] = []

        # Track seen items to avoid duplicates
        seen_calls: set[tuple[str, str]] = set()
        seen_io: set[tuple[str, str, str]] = set()
        seen_symbols: set[str] = set()

        for input_data in inputs:
            if not input_data:
                continue

            source_chunk = input_data.get("chunk_id") or input_data.get("merge_node_id", "")

            # Handle facts from chunk results
            facts = input_data.get("facts", {})
            if facts:
                # Merge calls
                for call in facts.get("calls", []):
                    target = call.get("target", "")
                    call_type = call.get("call_type", "perform")
                    key = (target, call_type)
                    if key not in seen_calls:
                        seen_calls.add(key)
                        call_graph_edges.append({
                            "from": source_chunk,
                            "to": target,
                            "type": call_type,
                            "source_chunk": source_chunk,
                            "is_external": call.get("is_external", False),
                        })

                # Merge I/O operations
                for io in facts.get("io_operations", []):
                    op = io.get("operation", "")
                    file_name = io.get("file_name", "")
                    record = io.get("record_name", "")
                    key = (op, file_name, record)
                    if key not in seen_io:
                        seen_io.add(key)
                        io_map.append(IOOperation(
                            operation=op,
                            file_name=file_name or None,
                            record_name=record or None,
                            line_number=io.get("line_number"),
                            status_check=io.get("status_check", False),
                        ))

                # Merge error handling
                for eh in facts.get("error_handling", []):
                    error_handling_behaviors.append(ErrorHandlingPattern(
                        pattern_type=eh.get("pattern_type", ""),
                        description=eh.get("description", ""),
                        line_numbers=eh.get("line_numbers", []),
                        related_symbols=eh.get("related_symbols", []),
                    ))

                # Merge symbols
                for sym in facts.get("symbols_defined", []):
                    name = sym.get("name", "")
                    if name and name not in seen_symbols:
                        seen_symbols.add(name)
                        symbols.append(SymbolDef(
                            name=name,
                            kind=sym.get("kind", "unknown"),
                            attributes=sym.get("attributes", {}),
                            line_number=sym.get("line_number"),
                        ))

            # Handle consolidated_facts from merge results
            consolidated = input_data.get("consolidated_facts", {})
            if consolidated:
                # Merge call graph edges
                for edge in consolidated.get("call_graph_edges", []):
                    target = edge.get("to", "")
                    call_type = edge.get("type", "perform")
                    key = (target, call_type)
                    if key not in seen_calls:
                        seen_calls.add(key)
                        edge["source_chunk"] = source_chunk
                        call_graph_edges.append(edge)

                # Merge I/O map
                for io in consolidated.get("io_map", []):
                    op = io.get("operation", "")
                    file_name = io.get("file_name", "")
                    record = io.get("record_name", "")
                    key = (op, file_name, record)
                    if key not in seen_io:
                        seen_io.add(key)
                        io_map.append(IOOperation(
                            operation=op,
                            file_name=file_name or None,
                            record_name=record or None,
                            line_number=io.get("line_number"),
                            status_check=io.get("status_check", False),
                        ))

                # Merge error handling
                for eh in consolidated.get("error_handling_behaviors", []):
                    error_handling_behaviors.append(ErrorHandlingPattern(
                        pattern_type=eh.get("pattern_type", ""),
                        description=eh.get("description", ""),
                        line_numbers=eh.get("line_numbers", []),
                        related_symbols=eh.get("related_symbols", []),
                    ))

                # Merge symbols
                for sym in consolidated.get("symbols", []):
                    name = sym.get("name", "")
                    if name and name not in seen_symbols:
                        seen_symbols.add(name)
                        symbols.append(SymbolDef(
                            name=name,
                            kind=sym.get("kind", "unknown"),
                            attributes=sym.get("attributes", {}),
                            line_number=sym.get("line_number"),
                        ))

        return ConsolidatedFacts(
            call_graph_edges=call_graph_edges,
            io_map=io_map,
            error_handling_behaviors=error_handling_behaviors,
            symbols=symbols,
        )

    def _parse_conflicts(
        self, conflicts_data: list[dict[str, Any]]
    ) -> list[MergeConflict]:
        """Parse conflict data from LLM response.

        Args:
            conflicts_data: Raw conflict data.

        Returns:
            List of MergeConflict objects.
        """
        conflicts: list[MergeConflict] = []
        for c in conflicts_data:
            conflicts.append(MergeConflict(
                description=c.get("description", "Unknown conflict"),
                input_ids=c.get("input_ids", []),
                suggested_followup_scope=c.get("suggested_followup_scope"),
            ))
        return conflicts

    def _update_doc_model(
        self,
        base_model: DocumentationModel,
        answers: list[dict[str, Any]],
        sections_modified: list[str],
        changes_summary: str,
    ) -> DocumentationModel:
        """Update documentation model after patch merge.

        Args:
            base_model: Original documentation model.
            answers: Follow-up answers applied.
            sections_modified: IDs of modified sections.
            changes_summary: Description of changes.

        Returns:
            Updated DocumentationModel.
        """
        # Copy existing sections
        sections = list(base_model.sections)

        # Add new sections for follow-up answers if they don't update existing
        for answer in answers:
            issue_id = answer.get("issue_id", "unknown")
            section_id = f"followup_{issue_id}"

            # Check if this updates an existing section
            existing_idx = None
            for i, s in enumerate(sections):
                if s.section_id in sections_modified or s.section_id == section_id:
                    existing_idx = i
                    break

            if existing_idx is not None:
                # Update existing section
                sections[existing_idx] = Section(
                    section_id=sections[existing_idx].section_id,
                    title=sections[existing_idx].title,
                    content=answer.get("answer", sections[existing_idx].content),
                    source_refs=sections[existing_idx].source_refs
                    + answer.get("scope", {}).get("chunk_ids", []),
                )
            else:
                # Add new section
                sections.append(Section(
                    section_id=section_id,
                    title=f"Follow-up: {issue_id}",
                    content=answer.get("answer", ""),
                    source_refs=answer.get("scope", {}).get("chunk_ids", []),
                ))

        # Update index with new information
        updated_index = self._rebuild_index(sections, answers)

        return DocumentationModel(
            doc_uri=base_model.doc_uri,
            sections=sections,
            index=updated_index,
            metadata={
                **base_model.metadata,
                "last_patch": changes_summary,
                "patches_applied": len(answers),
            },
        )

    def _rebuild_index(
        self,
        sections: list[Section],
        answers: list[dict[str, Any]],
    ) -> DocIndex:
        """Rebuild documentation index.

        Args:
            sections: All documentation sections.
            answers: Follow-up answers for additional indexing.

        Returns:
            Updated DocIndex.
        """
        symbol_to_chunks: dict[str, list[str]] = {}
        paragraph_to_chunk: dict[str, str] = {}
        file_to_chunks: dict[str, list[str]] = {}

        # Index from sections
        for section in sections:
            for chunk_id in section.source_refs:
                # This is a simplified index - in production would parse content
                pass

        # Index from answers
        for answer in answers:
            facts = answer.get("facts", {})
            chunk_ids = answer.get("scope", {}).get("chunk_ids", [])

            for sym in facts.get("symbols_defined", []):
                name = sym.get("name", "")
                if name:
                    if name not in symbol_to_chunks:
                        symbol_to_chunks[name] = []
                    symbol_to_chunks[name].extend(chunk_ids)

            for para in facts.get("paragraphs_defined", []):
                if para and chunk_ids:
                    paragraph_to_chunk[para] = chunk_ids[0]

        return DocIndex(
            symbol_to_chunks=symbol_to_chunks,
            paragraph_to_chunk=paragraph_to_chunk,
            file_to_chunks=file_to_chunks,
        )
