"""Aggregator worker for merging chunk results.

The Aggregator merges chunk results or child merges into higher-level
summaries, building up the documentation hierarchically.

Key Responsibilities:
- Merge facts from multiple inputs
- Identify conflicts and inconsistencies
- Produce consolidated documentation sections
"""

from abc import abstractmethod
from typing import Any

from atlas.workers.base import Worker
from atlas.models.work_item import WorkItem, DocMergePayload, DocPatchMergePayload
from atlas.models.results import MergeResult, DocumentationModel
from atlas.models.enums import WorkItemType


class Aggregator(Worker):
    """Worker for merging analysis results.

    Aggregators claim DOC_MERGE work items and produce MergeResult artifacts.
    They also handle DOC_PATCH_MERGE items for applying follow-up answers.

    Design Principles:
        - Merge structured facts from inputs
        - Identify and record conflicts between inputs
        - Produce consolidated documentation sections
        - Track coverage (what was included vs. missing)

    Output Schema (MergeResult):
        - coverage: Which inputs were included/missing
        - consolidated_facts: Merged facts
        - conflicts: Disagreements between inputs
        - narrative_sections: Optional documentation fragments

    Example Implementation:
        >>> class DocumentAggregator(Aggregator):
        ...     async def merge_results(self, inputs, payload, manifest):
        ...         # Combine facts from all inputs
        ...         consolidated = self._merge_facts(inputs)
        ...         # Detect conflicts
        ...         conflicts = self._find_conflicts(inputs)
        ...         # Build narrative
        ...         narrative = await self._generate_narrative(consolidated)
        ...         return MergeResult(...)

    TODO: Implement concrete aggregators for:
        - Hierarchical document merging
        - Patch merge with follow-up answers
    """

    @property
    def supported_work_types(self) -> list[WorkItemType]:
        """Aggregator supports merge and patch merge."""
        return [WorkItemType.DOC_MERGE, WorkItemType.DOC_PATCH_MERGE]

    async def process(self, work_item: WorkItem) -> Any:
        """Process a merge or patch merge work item.

        Args:
            work_item: The work item to process.

        Returns:
            MergeResult or updated DocumentationModel.
        """
        if work_item.work_type == WorkItemType.DOC_MERGE:
            return await self._process_merge(work_item)
        elif work_item.work_type == WorkItemType.DOC_PATCH_MERGE:
            return await self._process_patch_merge(work_item)
        else:
            raise ValueError(f"Unsupported work type: {work_item.work_type}")

    async def _process_merge(self, work_item: WorkItem) -> MergeResult:
        """Process a DOC_MERGE work item.

        Args:
            work_item: The merge work item.

        Returns:
            MergeResult artifact.
        """
        payload = work_item.payload
        if not isinstance(payload, DocMergePayload):
            raise ValueError("Expected DocMergePayload")

        # Load manifest
        manifest = await self.artifact_store.read_json(payload.manifest_uri)

        # Load input results
        inputs = []
        for input_uri in payload.input_uris:
            try:
                input_data = await self.artifact_store.read_json(input_uri)
                inputs.append(input_data)
            except Exception:
                # Track missing inputs
                pass

        # Perform merge
        result = await self.merge_results(
            inputs,
            payload,
            manifest,
        )

        # Write result artifact
        await self.artifact_store.write_json(
            payload.output_uri,
            result.model_dump(),
        )

        return result

    async def _process_patch_merge(
        self,
        work_item: WorkItem,
    ) -> tuple[str, DocumentationModel]:
        """Process a DOC_PATCH_MERGE work item.

        Args:
            work_item: The patch merge work item.

        Returns:
            Tuple of (updated doc URI, updated doc model).
        """
        payload = work_item.payload
        if not isinstance(payload, DocPatchMergePayload):
            raise ValueError("Expected DocPatchMergePayload")

        # Load base documentation
        base_doc = await self.artifact_store.read_text(payload.base_doc_uri)
        base_model = await self.artifact_store.read_json(payload.base_doc_model_uri)

        # Load follow-up answers
        answers = []
        for input_uri in payload.inputs:
            answer = await self.artifact_store.read_json(input_uri)
            answers.append(answer)

        # Apply patches
        updated_doc, updated_model = await self.apply_patches(
            base_doc,
            DocumentationModel(**base_model),
            answers,
        )

        # Write updated artifacts
        await self.artifact_store.write_text(
            payload.output_doc_uri,
            updated_doc,
        )
        await self.artifact_store.write_json(
            payload.output_doc_model_uri,
            updated_model.model_dump(),
        )

        return payload.output_doc_uri, updated_model

    @abstractmethod
    async def merge_results(
        self,
        inputs: list[dict[str, Any]],
        payload: DocMergePayload,
        manifest: dict[str, Any],
    ) -> MergeResult:
        """Merge multiple input results.

        Args:
            inputs: List of input result artifacts.
            payload: Merge work item payload.
            manifest: The workflow manifest.

        Returns:
            MergeResult with consolidated facts.

        TODO: Implement merge logic.
        """
        pass

    @abstractmethod
    async def apply_patches(
        self,
        base_doc: str,
        base_model: DocumentationModel,
        answers: list[dict[str, Any]],
    ) -> tuple[str, DocumentationModel]:
        """Apply follow-up answers to documentation.

        Args:
            base_doc: The base documentation text.
            base_model: The base documentation model.
            answers: List of follow-up answer artifacts.

        Returns:
            Tuple of (updated doc, updated model).

        TODO: Implement patch application logic.
        """
        pass

    async def _output_exists(self, work_item: WorkItem) -> bool:
        """Check if output already exists."""
        output_uri = self._get_output_uri(work_item)
        if output_uri:
            return await self.artifact_store.exists(output_uri)
        return False
