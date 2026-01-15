"""Scribe worker for chunk analysis.

The Scribe analyzes individual chunks of source code and produces
structured ChunkResult artifacts with facts, evidence, and open questions.

Key Responsibilities:
- Extract symbols, calls, I/O operations, error handling patterns
- Record evidence with line references
- Document open questions when context is insufficient
"""

from abc import abstractmethod
from typing import Any

from atlas.workers.base import Worker
from atlas.models.work_item import WorkItem, DocChunkPayload, DocFollowupPayload
from atlas.models.results import ChunkResult, FollowupAnswer, ChunkFacts
from atlas.models.enums import WorkItemType


class Scribe(Worker):
    """Worker for analyzing source code chunks.

    Scribes claim DOC_CHUNK work items and produce ChunkResult artifacts.
    They also handle DOC_FOLLOWUP items for targeted follow-up analysis.

    Design Principles:
        - Extract structured facts (symbols, calls, I/O, error handling)
        - Record evidence with line number references
        - Document open questions when context is insufficient
        - Never guess - record unknowns explicitly

    Output Schema (ChunkResult):
        - summary: Short narrative summary
        - facts: Structured, mergeable facts
        - evidence: Source references
        - open_questions: Explicit unknowns
        - confidence: 0.0 to 1.0

    Example Implementation:
        >>> class COBOLScribe(Scribe):
        ...     async def analyze_chunk(self, content, chunk_spec, manifest):
        ...         # Build analysis prompt
        ...         messages = self._build_prompt(content, chunk_spec)
        ...         # Get LLM analysis
        ...         response = await self.llm.complete_json(messages, CHUNK_SCHEMA)
        ...         # Create result
        ...         return ChunkResult(**response)

    TODO: Implement concrete scribes for:
        - COBOL programs
        - COBOL copybooks
        - JCL scripts
    """

    @property
    def supported_work_types(self) -> list[WorkItemType]:
        """Scribe supports chunk and follow-up analysis."""
        return [WorkItemType.DOC_CHUNK, WorkItemType.DOC_FOLLOWUP]

    async def process(self, work_item: WorkItem) -> Any:
        """Process a chunk or follow-up work item.

        Args:
            work_item: The work item to process.

        Returns:
            ChunkResult or FollowupAnswer depending on work type.
        """
        if work_item.work_type == WorkItemType.DOC_CHUNK:
            return await self._process_chunk(work_item)
        elif work_item.work_type == WorkItemType.DOC_FOLLOWUP:
            return await self._process_followup(work_item)
        else:
            raise ValueError(f"Unsupported work type: {work_item.work_type}")

    async def _process_chunk(self, work_item: WorkItem) -> ChunkResult:
        """Process a DOC_CHUNK work item.

        Args:
            work_item: The chunk work item.

        Returns:
            ChunkResult artifact.
        """
        payload = work_item.payload
        if not isinstance(payload, DocChunkPayload):
            raise ValueError("Expected DocChunkPayload")

        # Load manifest
        manifest = await self.artifact_store.read_json(payload.manifest_uri)

        # Load chunk content
        chunk_content = await self._load_chunk_content(
            payload.artifact_ref.artifact_uri,
            payload.chunk_locator,
        )

        # Analyze the chunk
        result = await self.analyze_chunk(
            chunk_content,
            payload,
            manifest,
        )

        # Write result artifact
        await self.artifact_store.write_json(
            payload.result_uri,
            result.model_dump(),
        )

        return result

    async def _process_followup(self, work_item: WorkItem) -> FollowupAnswer:
        """Process a DOC_FOLLOWUP work item.

        Args:
            work_item: The follow-up work item.

        Returns:
            FollowupAnswer artifact.
        """
        payload = work_item.payload
        if not isinstance(payload, DocFollowupPayload):
            raise ValueError("Expected DocFollowupPayload")

        # Load relevant inputs
        inputs = []
        for input_uri in payload.inputs:
            input_data = await self.artifact_store.read_json(input_uri)
            inputs.append(input_data)

        # Answer the follow-up question
        answer = await self.answer_followup(
            payload.issue_id,
            payload.scope,
            inputs,
        )

        # Write result artifact
        await self.artifact_store.write_json(
            payload.output_uri,
            answer.model_dump(),
        )

        return answer

    @abstractmethod
    async def analyze_chunk(
        self,
        content: str,
        payload: DocChunkPayload,
        manifest: dict[str, Any],
    ) -> ChunkResult:
        """Analyze a chunk of source code.

        Args:
            content: The chunk source code.
            payload: Chunk work item payload.
            manifest: The workflow manifest.

        Returns:
            ChunkResult with structured analysis.

        TODO: Implement language-specific analysis logic.
        """
        pass

    @abstractmethod
    async def answer_followup(
        self,
        issue_id: str,
        scope: dict[str, Any],
        inputs: list[dict[str, Any]],
    ) -> FollowupAnswer:
        """Answer a follow-up question.

        Args:
            issue_id: The issue being addressed.
            scope: The analysis scope.
            inputs: Relevant input artifacts.

        Returns:
            FollowupAnswer with the response.

        TODO: Implement follow-up analysis logic.
        """
        pass

    async def _load_chunk_content(
        self,
        artifact_uri: str,
        chunk_locator: Any,
    ) -> str:
        """Load chunk content from the artifact.

        Args:
            artifact_uri: URI of the source artifact.
            chunk_locator: Chunk location specification.

        Returns:
            The chunk content as a string.
        """
        # Load full source
        source = await self.artifact_store.read_text(artifact_uri)
        lines = source.splitlines()

        # Extract chunk lines
        start = chunk_locator.start_line - 1 if chunk_locator.start_line else 0
        end = chunk_locator.end_line if chunk_locator.end_line else len(lines)

        chunk_lines = lines[start:end]
        return "\n".join(chunk_lines)

    async def _output_exists(self, work_item: WorkItem) -> bool:
        """Check if output already exists."""
        output_uri = self._get_output_uri(work_item)
        if output_uri:
            return await self.artifact_store.exists(output_uri)
        return False
