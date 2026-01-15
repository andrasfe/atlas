"""Patch Merge worker implementation.

The Patch Merge worker handles DOC_PATCH_MERGE work items, applying
follow-up answers to existing documentation to produce updated
documentation and documentation models.

This implementation:
- Loads base documentation and model
- Applies follow-up answers to update content
- Handles fact corrections and additions
- Updates section source references
- Produces updated doc and doc model artifacts
"""

import json
import logging
from datetime import datetime
from typing import Any

from atlas.adapters.artifact_store import ArtifactStoreAdapter
from atlas.adapters.llm import LLMAdapter, LLMMessage, LLMRole
from atlas.adapters.ticket_system import TicketSystemAdapter
from atlas.models.enums import WorkItemStatus, WorkItemType
from atlas.models.results import (
    DocIndex,
    DocumentationModel,
    Section,
)
from atlas.models.work_item import DocPatchMergePayload, WorkItem
from atlas.workers.base import Worker

logger = logging.getLogger(__name__)


PATCH_MERGE_SYSTEM_PROMPT = """You are an expert documentation editor applying updates from follow-up analysis.

You will be given:
1. Base documentation (the current documentation text)
2. Base documentation model (machine-readable structure with sections)
3. Follow-up answers addressing specific issues

Your task is to:
1. Integrate the follow-up answers into the documentation
2. Update or add sections as needed
3. Ensure consistency and proper flow
4. Preserve existing content that is not being updated

Produce valid JSON with this structure:
{
  "updated_content": "The full updated documentation text with follow-up answers integrated",
  "sections_modified": ["section_id_1", "section_id_2"],
  "sections_added": ["new_section_id_1"],
  "changes_summary": "Brief description of what was updated",
  "new_sections": [
    {
      "section_id": "...",
      "title": "...",
      "content": "...",
      "source_refs": ["chunk_id_1", "chunk_id_2"],
      "after_section_id": "optional - id of section this should appear after"
    }
  ]
}

Guidelines:
- Integrate answers naturally into the documentation flow
- Update existing sections when the answer clarifies or corrects them
- Add new sections for substantial new information
- Preserve source references and traceability
- Maintain document structure and formatting
- If multiple answers relate to the same topic, consolidate them
"""

PATCH_MERGE_USER_PROMPT = """Apply the following follow-up answers to update the documentation.

Artifact: {artifact_id}
Job ID: {job_id}

---BASE DOCUMENTATION---
{base_doc}
---END BASE DOCUMENTATION---

---BASE DOCUMENTATION MODEL---
{base_model_json}
---END BASE DOCUMENTATION MODEL---

---FOLLOW-UP ANSWERS ({answer_count} answers)---
{answers_json}
---END FOLLOW-UP ANSWERS---

Update the documentation to incorporate all follow-up answers. Ensure the changes flow naturally
and maintain the document's overall structure and quality.
"""


class PatchMergeResult:
    """Result of a patch merge operation.

    Attributes:
        updated_doc: The updated documentation text.
        updated_model: The updated documentation model.
        sections_modified: IDs of sections that were modified.
        sections_added: IDs of new sections that were added.
        changes_summary: Brief description of changes made.
        success: Whether the patch merge succeeded.
        error: Error message if failed.
    """

    def __init__(
        self,
        updated_doc: str = "",
        updated_model: DocumentationModel | None = None,
        sections_modified: list[str] | None = None,
        sections_added: list[str] | None = None,
        changes_summary: str = "",
        success: bool = True,
        error: str | None = None,
    ):
        self.updated_doc = updated_doc
        self.updated_model = updated_model
        self.sections_modified = sections_modified or []
        self.sections_added = sections_added or []
        self.changes_summary = changes_summary
        self.success = success
        self.error = error


class PatchMergeWorker(Worker):
    """Concrete implementation of the Patch Merge worker.

    Applies follow-up answers to documentation to produce updated
    documentation that addresses challenger issues.

    Design Principles:
        - Preserves existing documentation structure
        - Integrates answers naturally into the document flow
        - Updates source references for traceability
        - Maintains consistency between doc and doc model

    Example:
        >>> worker = PatchMergeWorker(
        ...     worker_id="patch-merge-1",
        ...     ticket_system=ticket_system,
        ...     artifact_store=artifact_store,
        ...     llm=llm,
        ... )
        >>> result = await worker.process(work_item)
    """

    def __init__(
        self,
        worker_id: str,
        ticket_system: TicketSystemAdapter,
        artifact_store: ArtifactStoreAdapter,
        llm: LLMAdapter,
    ):
        """Initialize the patch merge worker.

        Args:
            worker_id: Unique identifier for this worker instance.
            ticket_system: Ticket system adapter.
            artifact_store: Artifact store adapter.
            llm: LLM adapter for generating merged documentation.
        """
        super().__init__(worker_id, ticket_system, artifact_store, llm)

    @property
    def supported_work_types(self) -> list[WorkItemType]:
        """Patch merge worker handles DOC_PATCH_MERGE items."""
        return [WorkItemType.DOC_PATCH_MERGE]

    async def process(self, work_item: WorkItem) -> PatchMergeResult:
        """Process a patch merge work item.

        Loads the base documentation and follow-up answers, then
        produces updated documentation incorporating the answers.

        Args:
            work_item: The patch merge work item.

        Returns:
            PatchMergeResult with updated doc and model.

        Raises:
            ValueError: If payload is not DocPatchMergePayload.
        """
        payload = work_item.payload
        if not isinstance(payload, DocPatchMergePayload):
            raise ValueError("Expected DocPatchMergePayload")

        logger.info(
            f"Processing patch merge for job {payload.job_id}, "
            f"applying {len(payload.inputs)} follow-up answers"
        )

        try:
            # Check if output already exists (idempotency)
            if await self._output_exists(work_item):
                logger.info(f"Patch merge output already exists for {work_item.work_id}")
                return await self._load_existing_result(payload)

            # Load base documentation
            base_doc = await self._load_base_doc(payload.base_doc_uri)
            base_model = await self._load_base_model(payload.base_doc_model_uri)

            # Load follow-up answers
            answers = await self._load_followup_answers(payload.inputs)

            if not answers:
                logger.warning(f"No follow-up answers to apply for {work_item.work_id}")
                # Just copy base doc to output
                result = PatchMergeResult(
                    updated_doc=base_doc,
                    updated_model=base_model,
                    changes_summary="No follow-up answers to apply",
                )
            else:
                # Apply patches
                result = await self._apply_patches(
                    payload=payload,
                    base_doc=base_doc,
                    base_model=base_model,
                    answers=answers,
                )

            # Write output artifacts
            await self._write_outputs(payload, result)

            logger.info(
                f"Patch merge complete for {work_item.work_id}: "
                f"modified {len(result.sections_modified)} sections, "
                f"added {len(result.sections_added)} sections"
            )

            return result

        except Exception as e:
            logger.exception(f"Patch merge failed for {work_item.work_id}")
            return PatchMergeResult(
                success=False,
                error=str(e),
            )

    async def _load_base_doc(self, uri: str) -> str:
        """Load base documentation text.

        Args:
            uri: URI of the base documentation.

        Returns:
            Documentation text content.
        """
        try:
            return await self.artifact_store.read_text(uri)
        except Exception as e:
            logger.error(f"Could not load base doc from {uri}: {e}")
            raise

    async def _load_base_model(self, uri: str) -> DocumentationModel:
        """Load base documentation model.

        Args:
            uri: URI of the base documentation model.

        Returns:
            DocumentationModel object.
        """
        try:
            data = await self.artifact_store.read_json(uri)
            return DocumentationModel(**data)
        except Exception as e:
            logger.error(f"Could not load base model from {uri}: {e}")
            # Return empty model if can't load
            return DocumentationModel(doc_uri=uri)

    async def _load_followup_answers(
        self, answer_uris: list[str]
    ) -> list[dict[str, Any]]:
        """Load follow-up answer artifacts.

        Args:
            answer_uris: URIs of follow-up answer artifacts.

        Returns:
            List of answer dictionaries.
        """
        answers: list[dict[str, Any]] = []

        for uri in answer_uris:
            try:
                data = await self.artifact_store.read_json(uri)
                answers.append(data)
                logger.debug(f"Loaded follow-up answer from {uri}")
            except Exception as e:
                logger.warning(f"Could not load follow-up answer from {uri}: {e}")

        return answers

    async def _apply_patches(
        self,
        payload: DocPatchMergePayload,
        base_doc: str,
        base_model: DocumentationModel,
        answers: list[dict[str, Any]],
    ) -> PatchMergeResult:
        """Apply follow-up answers to documentation.

        Uses LLM to intelligently integrate answers into the documentation
        while maintaining structure and flow.

        Args:
            payload: The patch merge payload.
            base_doc: Base documentation text.
            base_model: Base documentation model.
            answers: Follow-up answer artifacts.

        Returns:
            PatchMergeResult with updated content.
        """
        # Get artifact ID from payload
        artifact_id = payload.artifact_ref.artifact_id if payload.artifact_ref else "unknown"

        # Prepare model JSON for prompt (simplified)
        base_model_json = json.dumps(
            {
                "doc_uri": base_model.doc_uri,
                "sections": [
                    {
                        "section_id": s.section_id,
                        "title": s.title,
                        "source_refs": s.source_refs,
                    }
                    for s in base_model.sections
                ],
            },
            indent=2,
        )

        # Prepare answers JSON
        answers_summary = [
            {
                "issue_id": a.get("issue_id", "unknown"),
                "scope": a.get("scope", {}),
                "answer": a.get("answer", "")[:2000],  # Truncate long answers
                "confidence": a.get("confidence", 0.5),
            }
            for a in answers
        ]
        answers_json = json.dumps(answers_summary, indent=2)

        # Build prompt
        user_prompt = PATCH_MERGE_USER_PROMPT.format(
            artifact_id=artifact_id,
            job_id=payload.job_id,
            base_doc=base_doc[:10000],  # Truncate if too long
            base_model_json=base_model_json[:3000],
            answer_count=len(answers),
            answers_json=answers_json[:5000],
        )

        messages = [
            LLMMessage(role=LLMRole.SYSTEM, content=PATCH_MERGE_SYSTEM_PROMPT),
            LLMMessage(role=LLMRole.USER, content=user_prompt),
        ]

        try:
            response = await self.llm.complete_json(messages)

            # Parse response
            updated_content = response.get("updated_content", base_doc)
            sections_modified = response.get("sections_modified", [])
            sections_added = response.get("sections_added", [])
            changes_summary = response.get("changes_summary", "")
            new_sections = response.get("new_sections", [])

            # Update documentation model
            updated_model = self._update_doc_model(
                base_model=base_model,
                output_doc_uri=payload.output_doc_uri,
                sections_modified=sections_modified,
                new_sections=new_sections,
                answers=answers,
            )

            return PatchMergeResult(
                updated_doc=updated_content,
                updated_model=updated_model,
                sections_modified=sections_modified,
                sections_added=sections_added,
                changes_summary=changes_summary,
            )

        except Exception as e:
            logger.error(f"LLM patch merge failed: {e}")
            # Fall back to simple concatenation
            return self._fallback_merge(base_doc, base_model, payload, answers)

    def _fallback_merge(
        self,
        base_doc: str,
        base_model: DocumentationModel,
        payload: DocPatchMergePayload,
        answers: list[dict[str, Any]],
    ) -> PatchMergeResult:
        """Fallback merge when LLM fails.

        Simply appends follow-up answers to the documentation.

        Args:
            base_doc: Base documentation text.
            base_model: Base documentation model.
            payload: The patch merge payload.
            answers: Follow-up answer artifacts.

        Returns:
            PatchMergeResult with appended content.
        """
        updated_content = base_doc
        sections_added: list[str] = []

        # Append each answer
        for answer in answers:
            issue_id = answer.get("issue_id", "unknown")
            answer_text = answer.get("answer", "")
            scope = answer.get("scope", {})

            if answer_text:
                section_id = f"followup_{issue_id}"
                updated_content += f"\n\n## Follow-up: {issue_id}\n\n"
                updated_content += answer_text
                sections_added.append(section_id)

        # Update model
        updated_model = self._update_doc_model(
            base_model=base_model,
            output_doc_uri=payload.output_doc_uri,
            sections_modified=[],
            new_sections=[],
            answers=answers,
        )

        return PatchMergeResult(
            updated_doc=updated_content,
            updated_model=updated_model,
            sections_modified=[],
            sections_added=sections_added,
            changes_summary=f"Appended {len(answers)} follow-up answers (fallback merge)",
        )

    def _update_doc_model(
        self,
        base_model: DocumentationModel,
        output_doc_uri: str,
        sections_modified: list[str],
        new_sections: list[dict[str, Any]],
        answers: list[dict[str, Any]],
    ) -> DocumentationModel:
        """Update documentation model with patch merge changes.

        Args:
            base_model: Original documentation model.
            output_doc_uri: URI for the updated documentation.
            sections_modified: IDs of modified sections.
            new_sections: New sections to add.
            answers: Follow-up answers for additional indexing.

        Returns:
            Updated DocumentationModel.
        """
        # Copy existing sections
        sections = list(base_model.sections)

        # Add new sections from LLM response
        for ns in new_sections:
            section = Section(
                section_id=ns.get("section_id", f"new_{len(sections)}"),
                title=ns.get("title", "Untitled"),
                content=ns.get("content"),
                source_refs=ns.get("source_refs", []),
            )
            sections.append(section)

        # Add sections for follow-up answers that weren't incorporated
        for answer in answers:
            issue_id = answer.get("issue_id", "unknown")
            section_id = f"followup_{issue_id}"

            # Check if section already exists
            existing_ids = {s.section_id for s in sections}
            if section_id not in existing_ids:
                chunk_ids = answer.get("scope", {}).get("chunk_ids", [])
                sections.append(
                    Section(
                        section_id=section_id,
                        title=f"Follow-up: {issue_id}",
                        content=answer.get("answer", ""),
                        source_refs=chunk_ids,
                    )
                )

        # Rebuild index
        updated_index = self._rebuild_index(sections, answers)

        return DocumentationModel(
            doc_uri=output_doc_uri,
            sections=sections,
            index=updated_index,
            metadata={
                **base_model.metadata,
                "patch_merge_timestamp": datetime.utcnow().isoformat(),
                "sections_modified": sections_modified,
                "followup_count": len(answers),
            },
        )

    def _rebuild_index(
        self,
        sections: list[Section],
        answers: list[dict[str, Any]],
    ) -> DocIndex:
        """Rebuild documentation index after patch merge.

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
                # Map section title words as potential symbol references
                # This is simplified - production would parse content
                pass

        # Index from follow-up answers
        for answer in answers:
            facts = answer.get("facts", {})
            chunk_ids = answer.get("scope", {}).get("chunk_ids", [])

            # Index symbols
            for sym in facts.get("symbols_defined", []):
                name = sym.get("name", "")
                if name:
                    if name not in symbol_to_chunks:
                        symbol_to_chunks[name] = []
                    symbol_to_chunks[name].extend(chunk_ids)
                    # Deduplicate
                    symbol_to_chunks[name] = list(set(symbol_to_chunks[name]))

            # Index paragraphs
            for para in facts.get("paragraphs_defined", []):
                if para and chunk_ids:
                    paragraph_to_chunk[para] = chunk_ids[0]

        return DocIndex(
            symbol_to_chunks=symbol_to_chunks,
            paragraph_to_chunk=paragraph_to_chunk,
            file_to_chunks=file_to_chunks,
        )

    async def _write_outputs(
        self,
        payload: DocPatchMergePayload,
        result: PatchMergeResult,
    ) -> None:
        """Write output artifacts.

        Args:
            payload: The patch merge payload with output URIs.
            result: The patch merge result to write.
        """
        # Write updated documentation
        await self.artifact_store.write_text(
            payload.output_doc_uri,
            result.updated_doc,
        )
        logger.debug(f"Wrote updated doc to {payload.output_doc_uri}")

        # Write updated documentation model
        if result.updated_model:
            await self.artifact_store.write_json(
                payload.output_doc_model_uri,
                result.updated_model.model_dump(),
            )
            logger.debug(f"Wrote updated doc model to {payload.output_doc_model_uri}")

    async def _output_exists(self, work_item: WorkItem) -> bool:
        """Check if output already exists (idempotency check).

        Args:
            work_item: The work item to check.

        Returns:
            True if both output artifacts exist.
        """
        if not isinstance(work_item.payload, DocPatchMergePayload):
            return False

        payload = work_item.payload
        doc_exists = await self.artifact_store.exists(payload.output_doc_uri)
        model_exists = await self.artifact_store.exists(payload.output_doc_model_uri)

        return doc_exists and model_exists

    async def _load_existing_result(
        self, payload: DocPatchMergePayload
    ) -> PatchMergeResult:
        """Load existing result for idempotent reprocessing.

        Args:
            payload: The patch merge payload.

        Returns:
            PatchMergeResult loaded from existing artifacts.
        """
        try:
            updated_doc = await self.artifact_store.read_text(payload.output_doc_uri)
            model_data = await self.artifact_store.read_json(payload.output_doc_model_uri)
            updated_model = DocumentationModel(**model_data)

            return PatchMergeResult(
                updated_doc=updated_doc,
                updated_model=updated_model,
                changes_summary="Loaded from existing output (idempotent)",
            )
        except Exception as e:
            logger.error(f"Could not load existing result: {e}")
            raise
