"""Scribe worker implementation for chunk analysis.

The Scribe analyzes individual chunks of source code and produces
structured ChunkResult artifacts with facts, evidence, and open questions.

This implementation uses LLM-based analysis to extract:
- Symbol definitions (variables, paragraphs, sections)
- Symbol usage (referenced names)
- Call targets (PERFORM, CALL statements)
- I/O operations (READ, WRITE, etc.)
- Error handling patterns (FILE STATUS, ON ERROR, etc.)
"""

import logging
from typing import Any

from atlas.adapters.artifact_store import ArtifactStoreAdapter
from atlas.adapters.llm import LLMAdapter, LLMMessage, LLMRole
from atlas.adapters.ticket_system import TicketSystemAdapter
from atlas.models.enums import ChunkKind
from atlas.models.results import (
    CallTarget,
    ChunkFacts,
    ChunkResult,
    ErrorHandlingPattern,
    Evidence,
    FollowupAnswer,
    IOOperation,
    OpenQuestion,
    SymbolDef,
)
from atlas.models.work_item import DocChunkPayload
from atlas.workers.scribe import Scribe

logger = logging.getLogger(__name__)


# Prompt templates for chunk analysis
CHUNK_ANALYSIS_SYSTEM_PROMPT = """You are an expert code analyst specializing in legacy systems analysis.
Your task is to analyze a chunk of source code and extract structured information.

You MUST produce valid JSON output with the following structure:
{
  "summary": "Brief narrative summary of what this code does",
  "facts": {
    "symbols_defined": [{"name": "...", "kind": "...", "attributes": {...}, "line_number": ...}],
    "symbols_used": ["symbol1", "symbol2"],
    "entrypoints": ["entry1"],
    "paragraphs_defined": ["PARA1", "PARA2"],
    "calls": [{"target": "...", "call_type": "perform|call", "is_external": false, "line_number": ...}],
    "io_operations": [{"operation": "READ|WRITE|...", "file_name": "...", "record_name": "...", "line_number": ..., "status_check": false}],
    "error_handling": [{"pattern_type": "...", "description": "...", "line_numbers": [...], "related_symbols": [...]}]
  },
  "evidence": [{"evidence_type": "line_range", "start_line": ..., "end_line": ..., "note": "..."}],
  "open_questions": [{"question": "...", "context_needed": "...", "suspected_location": "..."}],
  "confidence": 0.85
}

Guidelines:
- Extract ALL symbol definitions (variables, paragraphs, sections, etc.)
- Track ALL symbol usages
- Identify ALL call targets (PERFORM statements, CALL statements)
- Document ALL I/O operations with file status checking info
- Note error handling patterns (FILE STATUS checks, ON ERROR, EVALUATE)
- Record evidence with line number ranges
- If context is insufficient to determine something, add it to open_questions
- Confidence should reflect how complete your analysis is (0.0-1.0)
"""

CHUNK_ANALYSIS_USER_PROMPT = """Analyze the following {chunk_kind} code chunk from {artifact_id}.

Chunk ID: {chunk_id}
Lines: {start_line} to {end_line}
Division: {division}
Section: {section}
Paragraphs: {paragraphs}

---CODE START---
{content}
---CODE END---

Extract all structured information and produce the JSON output.
Remember to record open questions for anything you cannot determine from this chunk alone.
"""

FOLLOWUP_SYSTEM_PROMPT = """You are an expert code analyst answering a specific question about code.

You MUST produce valid JSON output with the following structure:
{
  "answer": "Clear, detailed answer to the question",
  "facts": {
    "symbols_defined": [],
    "symbols_used": [],
    "entrypoints": [],
    "paragraphs_defined": [],
    "calls": [],
    "io_operations": [],
    "error_handling": []
  },
  "evidence": [{"evidence_type": "line_range", "start_line": ..., "end_line": ..., "note": "..."}],
  "confidence": 0.85
}

Be thorough and specific. If you cannot fully answer the question with the provided context,
explain what additional information would be needed.
"""

FOLLOWUP_USER_PROMPT = """Question to answer:
{question}

Issue ID: {issue_id}
Scope: {scope}

Available context from prior analysis:
{context}

Provide a comprehensive answer based on the available information.
"""


class ScribeWorker(Scribe):
    """Concrete implementation of the Scribe worker.

    Analyzes source code chunks using LLM and produces structured
    ChunkResult artifacts with extracted facts, evidence, and
    explicit open questions.

    Design Principles:
        - Extract structured facts (symbols, calls, I/O, error handling)
        - Record evidence with line number references
        - Document open questions when context is insufficient
        - Never guess - record unknowns explicitly
    """

    def __init__(
        self,
        worker_id: str,
        ticket_system: TicketSystemAdapter,
        artifact_store: ArtifactStoreAdapter,
        llm: LLMAdapter,
    ):
        """Initialize the scribe worker.

        Args:
            worker_id: Unique identifier for this worker instance.
            ticket_system: Ticket system adapter.
            artifact_store: Artifact store adapter.
            llm: LLM adapter for analysis.
        """
        super().__init__(worker_id, ticket_system, artifact_store, llm)

    async def analyze_chunk(
        self,
        content: str,
        payload: DocChunkPayload,
        manifest: dict[str, Any],
    ) -> ChunkResult:
        """Analyze a chunk of source code.

        Uses LLM to extract structured information from the code chunk,
        including symbols, calls, I/O operations, and error handling patterns.

        Args:
            content: The chunk source code.
            payload: Chunk work item payload.
            manifest: The workflow manifest.

        Returns:
            ChunkResult with structured analysis.
        """
        logger.debug(f"Analyzing chunk {payload.chunk_id}")

        # Determine chunk metadata
        chunk_spec = None
        if manifest and "chunks" in manifest:
            for chunk in manifest["chunks"]:
                if chunk.get("chunk_id") == payload.chunk_id:
                    chunk_spec = chunk
                    break

        chunk_kind = ChunkKind.GENERIC
        if chunk_spec:
            try:
                chunk_kind = ChunkKind(chunk_spec.get("chunk_kind", "generic"))
            except ValueError:
                pass

        # Build analysis prompt
        user_prompt = CHUNK_ANALYSIS_USER_PROMPT.format(
            chunk_kind=chunk_kind.value,
            artifact_id=payload.artifact_ref.artifact_id if payload.artifact_ref else "unknown",
            chunk_id=payload.chunk_id,
            start_line=payload.chunk_locator.start_line or 1,
            end_line=payload.chunk_locator.end_line or "end",
            division=payload.chunk_locator.division or "N/A",
            section=payload.chunk_locator.section or "N/A",
            paragraphs=", ".join(payload.chunk_locator.paragraphs) or "N/A",
            content=content,
        )

        messages = [
            LLMMessage(role=LLMRole.SYSTEM, content=CHUNK_ANALYSIS_SYSTEM_PROMPT),
            LLMMessage(role=LLMRole.USER, content=user_prompt),
        ]

        # Get LLM analysis
        try:
            response = await self.llm.complete_json(messages)
            result = self._parse_chunk_response(response, payload, chunk_kind)
        except Exception as e:
            logger.error(f"LLM analysis failed for chunk {payload.chunk_id}: {e}")
            # Return minimal result with error noted
            result = self._create_error_result(payload, chunk_kind, str(e))

        return result

    async def answer_followup(
        self,
        issue_id: str,
        scope: dict[str, Any],
        inputs: list[dict[str, Any]],
    ) -> FollowupAnswer:
        """Answer a follow-up question.

        Analyzes the provided context to answer a specific challenger
        question about the code.

        Args:
            issue_id: The issue being addressed.
            scope: The analysis scope.
            inputs: Relevant input artifacts (chunk results, etc.).

        Returns:
            FollowupAnswer with the response.
        """
        logger.debug(f"Answering follow-up for issue {issue_id}")

        # Extract question from scope
        question = scope.get("question", f"Investigate issue {issue_id}")

        # Build context from inputs
        context_parts: list[str] = []
        for i, input_data in enumerate(inputs):
            if "chunk_id" in input_data:
                context_parts.append(
                    f"Chunk {input_data.get('chunk_id', i)}:\n"
                    f"  Summary: {input_data.get('summary', 'N/A')}\n"
                    f"  Facts: {input_data.get('facts', {})}\n"
                )
            else:
                context_parts.append(f"Input {i}: {input_data}")

        context = "\n".join(context_parts) if context_parts else "No prior analysis available."

        # Build prompt
        user_prompt = FOLLOWUP_USER_PROMPT.format(
            question=question,
            issue_id=issue_id,
            scope=scope,
            context=context,
        )

        messages = [
            LLMMessage(role=LLMRole.SYSTEM, content=FOLLOWUP_SYSTEM_PROMPT),
            LLMMessage(role=LLMRole.USER, content=user_prompt),
        ]

        # Get LLM response
        try:
            response = await self.llm.complete_json(messages)
            answer = self._parse_followup_response(response, issue_id, scope)
        except Exception as e:
            logger.error(f"LLM follow-up failed for issue {issue_id}: {e}")
            answer = FollowupAnswer(
                issue_id=issue_id,
                scope=scope,
                answer=f"Analysis failed: {e}",
                confidence=0.0,
            )

        return answer

    def _parse_chunk_response(
        self,
        response: dict[str, Any],
        payload: DocChunkPayload,
        chunk_kind: ChunkKind,
    ) -> ChunkResult:
        """Parse LLM response into ChunkResult.

        Args:
            response: Raw LLM JSON response.
            payload: The chunk payload.
            chunk_kind: The chunk kind.

        Returns:
            Structured ChunkResult.
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

        # Extract open questions
        open_questions = [
            OpenQuestion(
                question=q.get("question", ""),
                context_needed=q.get("context_needed"),
                suspected_location=q.get("suspected_location"),
            )
            for q in response.get("open_questions", [])
        ]

        return ChunkResult(
            job_id=payload.job_id,
            artifact_id=payload.artifact_ref.artifact_id if payload.artifact_ref else "",
            artifact_version=payload.artifact_ref.artifact_version if payload.artifact_ref else "",
            chunk_id=payload.chunk_id,
            chunk_locator={
                "start_line": payload.chunk_locator.start_line,
                "end_line": payload.chunk_locator.end_line,
                "division": payload.chunk_locator.division,
                "section": payload.chunk_locator.section,
                "paragraphs": payload.chunk_locator.paragraphs,
            },
            chunk_kind=chunk_kind.value,
            summary=response.get("summary", ""),
            facts=facts,
            evidence=evidence,
            open_questions=open_questions,
            confidence=response.get("confidence", 0.5),
        )

    def _parse_followup_response(
        self,
        response: dict[str, Any],
        issue_id: str,
        scope: dict[str, Any],
    ) -> FollowupAnswer:
        """Parse LLM response into FollowupAnswer.

        Args:
            response: Raw LLM JSON response.
            issue_id: The issue ID.
            scope: The analysis scope.

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

        return FollowupAnswer(
            issue_id=issue_id,
            scope=scope,
            answer=response.get("answer", ""),
            facts=facts,
            evidence=evidence,
            confidence=response.get("confidence", 0.5),
        )

    def _create_error_result(
        self,
        payload: DocChunkPayload,
        chunk_kind: ChunkKind,
        error: str,
    ) -> ChunkResult:
        """Create a minimal result when analysis fails.

        Args:
            payload: The chunk payload.
            chunk_kind: The chunk kind.
            error: Error description.

        Returns:
            Minimal ChunkResult with error noted.
        """
        return ChunkResult(
            job_id=payload.job_id,
            artifact_id=payload.artifact_ref.artifact_id if payload.artifact_ref else "",
            artifact_version=payload.artifact_ref.artifact_version if payload.artifact_ref else "",
            chunk_id=payload.chunk_id,
            chunk_locator={
                "start_line": payload.chunk_locator.start_line,
                "end_line": payload.chunk_locator.end_line,
            },
            chunk_kind=chunk_kind.value,
            summary=f"Analysis failed: {error}",
            facts=ChunkFacts(),
            open_questions=[
                OpenQuestion(
                    question="Analysis failed, manual review required",
                    context_needed=error,
                )
            ],
            confidence=0.0,
            metadata={"error": error},
        )
