"""Mock LLM adapter for integration testing.

Provides a configurable mock LLM that returns predictable responses
for testing the full workflow pipeline.

Usage:
    >>> mock_llm = MockLLMAdapter()
    >>> mock_llm.configure_chunk_response({"summary": "Test summary"})
    >>> response = await mock_llm.complete_json(messages)
"""

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from atlas.adapters.llm import LLMAdapter, LLMMessage, LLMResponse
from atlas.models.results import (
    ChunkResult,
    ChunkFacts,
    MergeResult,
    ConsolidatedFacts,
    MergeCoverage,
    ChallengeResult,
    Issue,
    ResolutionPlan,
    FollowupTask,
    FollowupAnswer,
    Evidence,
    SymbolDef,
    IOOperation,
)
from atlas.models.enums import IssueSeverity


@dataclass
class MockLLMConfig:
    """Configuration for mock LLM behavior.

    Attributes:
        chunk_responses: Responses for chunk analysis.
        merge_responses: Responses for merge analysis.
        challenge_responses: Responses for challenge review.
        followup_responses: Responses for follow-up questions.
        default_confidence: Default confidence score.
        raise_issues: Whether challenger should raise issues.
        issue_count: Number of issues to raise.
        issue_severity: Severity of raised issues.
    """

    chunk_responses: list[dict[str, Any]] = field(default_factory=list)
    merge_responses: list[dict[str, Any]] = field(default_factory=list)
    challenge_responses: list[dict[str, Any]] = field(default_factory=list)
    followup_responses: list[dict[str, Any]] = field(default_factory=list)
    default_confidence: float = 0.9
    raise_issues: bool = True
    issue_count: int = 2
    issue_severity: IssueSeverity = IssueSeverity.MAJOR


class MockLLMAdapter(LLMAdapter):
    """Mock LLM adapter for integration testing.

    Provides configurable, predictable responses for testing
    the complete workflow pipeline.

    Features:
    - Configurable response content
    - Call tracking for verification
    - Support for all work item types
    - Controllable issue raising

    Example:
        >>> mock_llm = MockLLMAdapter()
        >>> mock_llm.set_chunk_response({"summary": "Test"})
        >>> response = await mock_llm.complete(messages)
    """

    def __init__(self, config: MockLLMConfig | None = None) -> None:
        """Initialize mock LLM.

        Args:
            config: Optional configuration.
        """
        self.config = config or MockLLMConfig()
        self._calls: list[dict[str, Any]] = []
        self._chunk_index = 0
        self._merge_index = 0
        self._challenge_index = 0
        self._followup_index = 0
        self._custom_handler: Callable[[list[LLMMessage]], str] | None = None

    def set_custom_handler(
        self,
        handler: Callable[[list[LLMMessage]], str],
    ) -> None:
        """Set a custom handler for generating responses.

        Args:
            handler: Function that takes messages and returns response content.
        """
        self._custom_handler = handler

    def set_chunk_responses(self, responses: list[dict[str, Any]]) -> None:
        """Set chunk analysis responses.

        Args:
            responses: List of response dictionaries.
        """
        self.config.chunk_responses = responses

    def set_merge_responses(self, responses: list[dict[str, Any]]) -> None:
        """Set merge responses.

        Args:
            responses: List of response dictionaries.
        """
        self.config.merge_responses = responses

    def set_challenge_responses(self, responses: list[dict[str, Any]]) -> None:
        """Set challenge responses.

        Args:
            responses: List of response dictionaries.
        """
        self.config.challenge_responses = responses

    def set_followup_responses(self, responses: list[dict[str, Any]]) -> None:
        """Set follow-up responses.

        Args:
            responses: List of response dictionaries.
        """
        self.config.followup_responses = responses

    def configure_no_issues(self) -> None:
        """Configure challenger to raise no issues."""
        self.config.raise_issues = False
        self.config.issue_count = 0

    def configure_issues(
        self,
        count: int = 2,
        severity: IssueSeverity = IssueSeverity.MAJOR,
    ) -> None:
        """Configure challenger to raise issues.

        Args:
            count: Number of issues to raise.
            severity: Severity level for issues.
        """
        self.config.raise_issues = True
        self.config.issue_count = count
        self.config.issue_severity = severity

    @property
    def call_count(self) -> int:
        """Get total number of LLM calls."""
        return len(self._calls)

    @property
    def calls(self) -> list[dict[str, Any]]:
        """Get all recorded calls."""
        return self._calls.copy()

    def reset(self) -> None:
        """Reset call tracking and indexes."""
        self._calls.clear()
        self._chunk_index = 0
        self._merge_index = 0
        self._challenge_index = 0
        self._followup_index = 0

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
        stop_sequences: list[str] | None = None,
    ) -> LLMResponse:
        """Generate a mock completion.

        Args:
            messages: Conversation messages.
            temperature: Ignored.
            max_tokens: Ignored.
            json_mode: If True, generate JSON response.
            stop_sequences: Ignored.

        Returns:
            Mock LLMResponse.
        """
        # Record call
        self._calls.append({
            "messages": [{"role": m.role.value, "content": m.content} for m in messages],
            "json_mode": json_mode,
            "temperature": temperature,
        })

        # Generate response
        if self._custom_handler:
            content = self._custom_handler(messages)
        else:
            content = self._generate_response(messages, json_mode)

        return LLMResponse(
            content=content,
            usage={
                "prompt_tokens": sum(len(m.content) // 4 for m in messages),
                "completion_tokens": len(content) // 4,
                "total_tokens": sum(len(m.content) // 4 for m in messages) + len(content) // 4,
            },
            model="mock-llm",
        )

    async def complete_json(
        self,
        messages: list[LLMMessage],
        schema: dict[str, Any] | None = None,
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Generate structured JSON output.

        Args:
            messages: Conversation messages.
            schema: Optional JSON schema (ignored by mock).
            temperature: Ignored.
            max_tokens: Ignored.

        Returns:
            Parsed JSON response.
        """
        response = await self.complete(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
        )
        return json.loads(response.content)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text to count.

        Returns:
            Approximate token count (~4 chars per token).
        """
        return len(text) // 4

    def count_message_tokens(self, messages: list[LLMMessage]) -> int:
        """Count tokens in messages.

        Args:
            messages: Messages to count.

        Returns:
            Total token count.
        """
        total = sum(self.count_tokens(m.content) for m in messages)
        return total + len(messages) * 4  # Overhead per message

    def get_context_limit(self) -> int:
        """Get context window size.

        Returns:
            Mock context limit of 8000 tokens.
        """
        return 8000

    @property
    def model_name(self) -> str:
        """Get model name.

        Returns:
            Mock model name.
        """
        return "mock-llm"

    def _generate_response(
        self,
        messages: list[LLMMessage],
        json_mode: bool,
    ) -> str:
        """Generate appropriate response based on message content.

        Args:
            messages: Input messages.
            json_mode: Whether JSON output is expected.

        Returns:
            Response content string.
        """
        # Analyze messages to determine work type
        message_text = " ".join(m.content.lower() for m in messages)

        if "chunk" in message_text and "analyze" in message_text:
            return self._generate_chunk_response()
        elif "merge" in message_text:
            return self._generate_merge_response()
        elif "challenge" in message_text or "review" in message_text:
            return self._generate_challenge_response()
        elif "followup" in message_text or "follow-up" in message_text:
            return self._generate_followup_response()
        else:
            # Default response
            return json.dumps({"response": "Mock response"})

    def _generate_chunk_response(self) -> str:
        """Generate chunk analysis response."""
        if self.config.chunk_responses:
            response = self.config.chunk_responses[
                self._chunk_index % len(self.config.chunk_responses)
            ]
            self._chunk_index += 1
            return json.dumps(response)

        # Default chunk response
        result = ChunkResult(
            job_id="mock-job",
            artifact_id="mock-artifact",
            artifact_version="mock-version",
            chunk_id=f"chunk-{self._chunk_index}",
            chunk_locator={"start_line": 1, "end_line": 100},
            chunk_kind="procedure_part",
            summary="This chunk contains processing logic for handling input records.",
            facts=ChunkFacts(
                symbols_defined=[
                    SymbolDef(name="WS-COUNTER", kind="variable", line_number=10),
                    SymbolDef(name="WS-STATUS", kind="variable", line_number=15),
                ],
                symbols_used=["WS-COUNTER", "WS-STATUS", "INPUT-RECORD"],
                entrypoints=["MAIN-PROCESS"],
                paragraphs_defined=["MAIN-PROCESS", "PROCESS-RECORD"],
                io_operations=[
                    IOOperation(
                        operation="READ",
                        file_name="INPUT-FILE",
                        line_number=50,
                        status_check=True,
                    )
                ],
            ),
            evidence=[
                Evidence(
                    evidence_type="line_range",
                    start_line=1,
                    end_line=100,
                    note="Main processing logic",
                )
            ],
            confidence=self.config.default_confidence,
        )
        self._chunk_index += 1
        return json.dumps(result.model_dump())

    def _generate_merge_response(self) -> str:
        """Generate merge response."""
        if self.config.merge_responses:
            response = self.config.merge_responses[
                self._merge_index % len(self.config.merge_responses)
            ]
            self._merge_index += 1
            return json.dumps(response)

        # Default merge response
        result = MergeResult(
            job_id="mock-job",
            artifact_id="mock-artifact",
            artifact_version="mock-version",
            merge_node_id=f"merge-{self._merge_index}",
            coverage=MergeCoverage(
                included_input_ids=["chunk-1", "chunk-2"],
                missing_input_ids=[],
            ),
            consolidated_facts=ConsolidatedFacts(
                call_graph_edges=[
                    {"from": "MAIN-PROCESS", "to": "PROCESS-RECORD"},
                ],
                io_map=[
                    IOOperation(
                        operation="READ",
                        file_name="INPUT-FILE",
                        status_check=True,
                    )
                ],
            ),
        )
        self._merge_index += 1
        return json.dumps(result.model_dump())

    def _generate_challenge_response(self) -> str:
        """Generate challenge response."""
        if self.config.challenge_responses:
            response = self.config.challenge_responses[
                self._challenge_index % len(self.config.challenge_responses)
            ]
            self._challenge_index += 1
            return json.dumps(response)

        # Generate issues based on config
        issues = []
        if self.config.raise_issues:
            for i in range(self.config.issue_count):
                issues.append(
                    Issue(
                        issue_id=f"issue-{self._challenge_index}-{i}",
                        severity=self.config.issue_severity,
                        question=f"Mock issue {i}: Please clarify the error handling logic.",
                        doc_section_refs=["section-1"],
                        suspected_scopes=["chunk-1"],
                        routing_hints={"paragraphs": ["PROCESS-RECORD"]},
                    )
                )

        # Build resolution plan
        followup_tasks = []
        for issue in issues:
            if issue.severity in [IssueSeverity.BLOCKER, IssueSeverity.MAJOR]:
                followup_tasks.append(
                    FollowupTask(
                        issue_id=issue.issue_id,
                        scope={"chunk_ids": issue.suspected_scopes},
                        description=issue.question,
                    )
                )

        result = ChallengeResult(
            job_id="mock-job",
            artifact_id="mock-artifact",
            artifact_version="mock-version",
            issues=issues,
            resolution_plan=ResolutionPlan(
                followup_tasks=followup_tasks,
                requires_patch_merge=len(followup_tasks) > 0,
            ),
        )
        self._challenge_index += 1
        return json.dumps(result.model_dump())

    def _generate_followup_response(self) -> str:
        """Generate follow-up response."""
        if self.config.followup_responses:
            response = self.config.followup_responses[
                self._followup_index % len(self.config.followup_responses)
            ]
            self._followup_index += 1
            return json.dumps(response)

        # Default follow-up response
        result = FollowupAnswer(
            issue_id=f"issue-{self._followup_index}",
            scope={"chunk_ids": ["chunk-1"]},
            answer="The error handling is implemented in the FILE-STATUS-CHECK paragraph. "
                   "The program checks FILE-STATUS after each I/O operation and abends if the status is not '00'.",
            facts=ChunkFacts(
                symbols_defined=[],
                symbols_used=["FILE-STATUS"],
            ),
            evidence=[
                Evidence(
                    evidence_type="line_range",
                    start_line=100,
                    end_line=110,
                    note="Error handling implementation",
                )
            ],
            confidence=0.95,
        )
        self._followup_index += 1
        return json.dumps(result.model_dump())


# Convenience factory functions

def create_mock_llm_with_no_issues() -> MockLLMAdapter:
    """Create mock LLM configured to raise no issues.

    Returns:
        MockLLMAdapter that produces clean challenge results.
    """
    mock = MockLLMAdapter()
    mock.configure_no_issues()
    return mock


def create_mock_llm_with_issues(
    count: int = 2,
    severity: IssueSeverity = IssueSeverity.MAJOR,
) -> MockLLMAdapter:
    """Create mock LLM configured to raise issues.

    Args:
        count: Number of issues to raise.
        severity: Issue severity level.

    Returns:
        MockLLMAdapter that produces issues.
    """
    mock = MockLLMAdapter()
    mock.configure_issues(count=count, severity=severity)
    return mock


def create_mock_llm_with_blockers() -> MockLLMAdapter:
    """Create mock LLM configured to raise blocker issues.

    Returns:
        MockLLMAdapter that produces blocker issues.
    """
    return create_mock_llm_with_issues(count=1, severity=IssueSeverity.BLOCKER)
