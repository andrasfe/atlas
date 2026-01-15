"""Follow-up worker implementation for targeted issue investigation.

The Follow-up worker handles targeted investigation of specific issues
raised by the Challenger. It re-analyzes code sections based on
challenger questions and produces detailed answers.

This implementation:
- Handles bounded follow-up investigations
- Re-analyzes code chunks with specific questions in mind
- Produces structured answers with evidence
- Supports cross-cutting follow-up plans
"""

import logging
from typing import Any

from atlas.adapters.artifact_store import ArtifactStoreAdapter
from atlas.adapters.llm import LLMAdapter, LLMMessage, LLMRole
from atlas.adapters.ticket_system import TicketSystemAdapter
from atlas.models.enums import WorkItemType
from atlas.models.results import (
    CallTarget,
    ChunkFacts,
    ErrorHandlingPattern,
    Evidence,
    FollowupAnswer,
    IOOperation,
    SymbolDef,
)
from atlas.models.work_item import DocFollowupPayload, WorkItem
from atlas.workers.base import Worker

logger = logging.getLogger(__name__)


FOLLOWUP_SYSTEM_PROMPT = """You are an expert code analyst answering a specific question about legacy code.

You will be given:
1. A specific question or issue to investigate
2. Relevant code chunks and/or prior analysis results
3. The scope of your investigation

Your task is to provide a thorough, evidence-based answer.

Produce valid JSON:
{
  "answer": "Detailed answer to the question with specific findings",
  "facts": {
    "symbols_defined": [{"name": "...", "kind": "...", "attributes": {...}, "line_number": ...}],
    "symbols_used": ["symbol1", "symbol2"],
    "entrypoints": [],
    "paragraphs_defined": [],
    "calls": [{"target": "...", "call_type": "...", "is_external": false, "line_number": ...}],
    "io_operations": [{"operation": "...", "file_name": "...", "record_name": "...", "status_check": false}],
    "error_handling": [{"pattern_type": "...", "description": "...", "line_numbers": [], "related_symbols": []}]
  },
  "evidence": [{"evidence_type": "line_range", "start_line": ..., "end_line": ..., "note": "..."}],
  "confidence": 0.85,
  "additional_context_needed": "If you cannot fully answer, explain what additional context would help"
}

Guidelines:
- Be specific and cite evidence from the code
- Include line number references where possible
- Extract any relevant facts that help answer the question
- If the question cannot be fully answered with available context, explain what's missing
- Confidence should reflect how completely you can answer (0.0-1.0)
"""

FOLLOWUP_USER_PROMPT = """Answer the following question:

Question: {question}
Issue ID: {issue_id}

Scope of investigation:
- Chunk IDs: {chunk_ids}
- Type: {scope_type}
- Division: {division}

---PRIOR ANALYSIS RESULTS---
{prior_analysis}
---END PRIOR ANALYSIS---

---SOURCE CODE (if available)---
{source_code}
---END SOURCE CODE---

Provide a comprehensive answer based on the evidence available.
"""

CROSS_CUTTING_PROMPT = """This is a cross-cutting investigation spanning multiple code areas.

Focus area: {focus_area}

Answer the question by synthesizing information from all available chunks.
Pay particular attention to:
- Patterns that span multiple paragraphs/sections
- Interactions between different code areas
- Overall behavior that emerges from combined analysis
"""


class FollowupWorker(Worker):
    """Concrete implementation of the Follow-up worker.

    Handles targeted investigation of specific issues raised by the
    Challenger, producing detailed answers with evidence.

    Design Principles:
        - Bounded scope: Each follow-up targets a limited set of chunks
        - Evidence-based: Answers cite specific code locations
        - Explicit unknowns: Clearly states what additional context is needed
        - Structured output: Facts and evidence are machine-readable
    """

    def __init__(
        self,
        worker_id: str,
        ticket_system: TicketSystemAdapter,
        artifact_store: ArtifactStoreAdapter,
        llm: LLMAdapter,
    ):
        """Initialize the follow-up worker.

        Args:
            worker_id: Unique identifier for this worker instance.
            ticket_system: Ticket system adapter.
            artifact_store: Artifact store adapter.
            llm: LLM adapter for analysis.
        """
        super().__init__(worker_id, ticket_system, artifact_store, llm)

    @property
    def supported_work_types(self) -> list[WorkItemType]:
        """Follow-up worker handles DOC_FOLLOWUP items."""
        return [WorkItemType.DOC_FOLLOWUP]

    async def process(self, work_item: WorkItem) -> FollowupAnswer:
        """Process a follow-up work item.

        Investigates a specific issue by analyzing the relevant code
        chunks and producing a detailed answer.

        Args:
            work_item: The follow-up work item.

        Returns:
            FollowupAnswer with the investigation results.
        """
        payload = work_item.payload
        if not isinstance(payload, DocFollowupPayload):
            raise ValueError("Expected DocFollowupPayload")

        logger.debug(f"Processing follow-up for issue {payload.issue_id}")

        # Load prior analysis results
        prior_analysis: list[dict[str, Any]] = []
        for input_uri in payload.inputs:
            try:
                data = await self.artifact_store.read_json(input_uri)
                prior_analysis.append(data)
            except Exception as e:
                logger.warning(f"Could not load input {input_uri}: {e}")

        # Try to load source code for the relevant chunks
        source_code = await self._load_source_for_scope(payload)

        # Determine scope type
        scope_type = payload.scope.get("type", "targeted")
        is_cross_cutting = scope_type == "cross_cutting"

        # Build and execute the follow-up analysis
        answer = await self._investigate(
            issue_id=payload.issue_id,
            scope=payload.scope,
            prior_analysis=prior_analysis,
            source_code=source_code,
            is_cross_cutting=is_cross_cutting,
        )

        # Write result artifact
        await self.artifact_store.write_json(
            payload.output_uri,
            answer.model_dump(),
        )

        return answer

    async def _investigate(
        self,
        issue_id: str,
        scope: dict[str, Any],
        prior_analysis: list[dict[str, Any]],
        source_code: str,
        is_cross_cutting: bool,
    ) -> FollowupAnswer:
        """Perform the follow-up investigation.

        Args:
            issue_id: The issue being investigated.
            scope: The investigation scope.
            prior_analysis: Prior analysis results for context.
            source_code: Source code for the scope (if available).
            is_cross_cutting: Whether this is a cross-cutting investigation.

        Returns:
            FollowupAnswer with findings.
        """
        # Extract question from scope
        question = scope.get("question", f"Investigate issue {issue_id}")
        chunk_ids = scope.get("chunk_ids", [])
        division = scope.get("division", "unknown")

        # Format prior analysis
        import json
        prior_json = json.dumps(prior_analysis, indent=2, default=str)[:8000]

        # Build prompt
        user_prompt = FOLLOWUP_USER_PROMPT.format(
            question=question,
            issue_id=issue_id,
            chunk_ids=", ".join(chunk_ids) if chunk_ids else "None specified",
            scope_type="cross-cutting" if is_cross_cutting else "targeted",
            division=division,
            prior_analysis=prior_json,
            source_code=source_code[:6000] if source_code else "Not available",
        )

        # Add cross-cutting context if needed
        system_prompt = FOLLOWUP_SYSTEM_PROMPT
        if is_cross_cutting:
            system_prompt += "\n\n" + CROSS_CUTTING_PROMPT.format(
                focus_area=scope.get("focus_area", "general analysis")
            )

        messages = [
            LLMMessage(role=LLMRole.SYSTEM, content=system_prompt),
            LLMMessage(role=LLMRole.USER, content=user_prompt),
        ]

        try:
            response = await self.llm.complete_json(messages)
            answer = self._parse_response(response, issue_id, scope)
        except Exception as e:
            logger.error(f"LLM follow-up investigation failed: {e}")
            answer = FollowupAnswer(
                issue_id=issue_id,
                scope=scope,
                answer=f"Investigation failed: {e}",
                confidence=0.0,
                metadata={"error": str(e)},
            )

        return answer

    def _parse_response(
        self,
        response: dict[str, Any],
        issue_id: str,
        scope: dict[str, Any],
    ) -> FollowupAnswer:
        """Parse LLM response into FollowupAnswer.

        Args:
            response: Raw LLM JSON response.
            issue_id: The issue ID.
            scope: The investigation scope.

        Returns:
            Structured FollowupAnswer.
        """
        # Extract facts
        facts_data = response.get("facts", {})
        facts = ChunkFacts(
            symbols_defined=[
                SymbolDef(
                    name=s.get("name", ""),
                    kind=s.get("kind", "unknown"),
                    attributes=s.get("attributes", {}),
                    line_number=s.get("line_number"),
                )
                for s in facts_data.get("symbols_defined", [])
            ],
            symbols_used=facts_data.get("symbols_used", []),
            entrypoints=facts_data.get("entrypoints", []),
            paragraphs_defined=facts_data.get("paragraphs_defined", []),
            calls=[
                CallTarget(
                    target=c.get("target", ""),
                    call_type=c.get("call_type", "perform"),
                    is_external=c.get("is_external", False),
                    line_number=c.get("line_number"),
                )
                for c in facts_data.get("calls", [])
            ],
            io_operations=[
                IOOperation(
                    operation=io.get("operation", ""),
                    file_name=io.get("file_name"),
                    record_name=io.get("record_name"),
                    line_number=io.get("line_number"),
                    status_check=io.get("status_check", False),
                )
                for io in facts_data.get("io_operations", [])
            ],
            error_handling=[
                ErrorHandlingPattern(
                    pattern_type=eh.get("pattern_type", ""),
                    description=eh.get("description", ""),
                    line_numbers=eh.get("line_numbers", []),
                    related_symbols=eh.get("related_symbols", []),
                )
                for eh in facts_data.get("error_handling", [])
            ],
        )

        # Extract evidence
        evidence = [
            Evidence(
                evidence_type=e.get("evidence_type", "line_range"),
                start_line=e.get("start_line"),
                end_line=e.get("end_line"),
                note=e.get("note"),
            )
            for e in response.get("evidence", [])
        ]

        # Build metadata
        metadata: dict[str, Any] = {}
        if response.get("additional_context_needed"):
            metadata["additional_context_needed"] = response["additional_context_needed"]

        return FollowupAnswer(
            issue_id=issue_id,
            scope=scope,
            answer=response.get("answer", ""),
            facts=facts,
            evidence=evidence,
            confidence=response.get("confidence", 0.5),
            metadata=metadata,
        )

    async def _load_source_for_scope(
        self,
        payload: DocFollowupPayload,
    ) -> str:
        """Load source code for the investigation scope.

        Args:
            payload: The follow-up payload with scope info.

        Returns:
            Source code text, or empty string if not available.
        """
        # Try to load source from artifact ref
        if payload.artifact_ref and payload.artifact_ref.artifact_uri:
            try:
                source = await self.artifact_store.read_text(
                    payload.artifact_ref.artifact_uri
                )

                # If we have specific line ranges, extract them
                chunk_ids = payload.scope.get("chunk_ids", [])
                if chunk_ids and payload.manifest_uri:
                    # Try to load manifest to get chunk boundaries
                    try:
                        manifest = await self.artifact_store.read_json(payload.manifest_uri)
                        source = self._extract_scope_lines(source, manifest, chunk_ids)
                    except Exception:
                        pass

                return source
            except Exception as e:
                logger.debug(f"Could not load source: {e}")

        return ""

    def _extract_scope_lines(
        self,
        source: str,
        manifest: dict[str, Any],
        chunk_ids: list[str],
    ) -> str:
        """Extract source lines for specific chunks.

        Args:
            source: Full source code.
            manifest: Workflow manifest with chunk specs.
            chunk_ids: IDs of chunks to extract.

        Returns:
            Extracted source code for the scope.
        """
        lines = source.splitlines()
        extracted_parts: list[str] = []

        chunks = manifest.get("chunks", [])
        for chunk in chunks:
            if chunk.get("chunk_id") in chunk_ids:
                start = chunk.get("start_line", 1) - 1
                end = chunk.get("end_line", len(lines))
                chunk_lines = lines[start:end]
                extracted_parts.append(
                    f"--- Chunk {chunk.get('chunk_id')} (lines {start+1}-{end}) ---\n"
                    + "\n".join(chunk_lines)
                )

        return "\n\n".join(extracted_parts)

    async def _output_exists(self, work_item: WorkItem) -> bool:
        """Check if output already exists."""
        output_uri = self._get_output_uri(work_item)
        if output_uri:
            return await self.artifact_store.exists(output_uri)
        return False
