"""Mock agent simulation harness for integration testing.

Provides a comprehensive infrastructure for simulating agent behavior
in integration tests with configurable delays, failures, and responses.

Features:
- Configurable response delays to simulate real-world latency
- Controllable failure modes (random, specific, intermittent)
- Detailed call tracking and statistics
- Support for all agent types (Scribe, Aggregator, Challenger)
- Reproducible behavior with seed-based randomness

Usage:
    >>> harness = MockAgentHarness()
    >>> harness.configure_delay(min_ms=10, max_ms=50)
    >>> harness.configure_failure_rate(0.1)  # 10% failure rate
    >>> mock_llm = harness.create_mock_llm()
"""

import asyncio
import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable
from enum import Enum

from atlas.adapters.llm import LLMAdapter, LLMMessage, LLMResponse
from atlas.models.enums import IssueSeverity, WorkItemStatus, WorkItemType
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


class FailureMode(str, Enum):
    """Types of simulated failures."""

    NONE = "none"
    RANDOM = "random"
    SPECIFIC = "specific"
    INTERMITTENT = "intermittent"
    TIMEOUT = "timeout"


class AgentType(str, Enum):
    """Types of agents that can be simulated."""

    SCRIBE = "scribe"
    AGGREGATOR = "aggregator"
    CHALLENGER = "challenger"
    FOLLOWUP = "followup"


@dataclass
class AgentStats:
    """Statistics for agent behavior tracking.

    Attributes:
        total_calls: Total number of calls made.
        successful_calls: Number of successful calls.
        failed_calls: Number of failed calls.
        total_latency_ms: Total latency across all calls.
        calls_by_type: Call count by agent type.
        responses_by_type: Response content by agent type.
    """

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_latency_ms: float = 0.0
    calls_by_type: dict[AgentType, int] = field(default_factory=dict)
    responses_by_type: dict[AgentType, list[dict]] = field(default_factory=dict)

    @property
    def average_latency_ms(self) -> float:
        """Calculate average latency per call."""
        if self.total_calls == 0:
            return 0.0
        return self.total_latency_ms / self.total_calls

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_calls == 0:
            return 1.0
        return self.successful_calls / self.total_calls


@dataclass
class DelayConfig:
    """Configuration for simulated delays.

    Attributes:
        enabled: Whether delays are enabled.
        min_ms: Minimum delay in milliseconds.
        max_ms: Maximum delay in milliseconds.
        per_type: Per-agent-type delay overrides.
    """

    enabled: bool = False
    min_ms: float = 0.0
    max_ms: float = 0.0
    per_type: dict[AgentType, tuple[float, float]] = field(default_factory=dict)


@dataclass
class FailureConfig:
    """Configuration for simulated failures.

    Attributes:
        mode: Type of failure simulation.
        rate: Probability of failure (0.0 to 1.0).
        specific_calls: Specific call numbers to fail.
        error_messages: Custom error messages by call number.
    """

    mode: FailureMode = FailureMode.NONE
    rate: float = 0.0
    specific_calls: list[int] = field(default_factory=list)
    error_messages: dict[int, str] = field(default_factory=dict)


@dataclass
class ResponseConfig:
    """Configuration for response generation.

    Attributes:
        chunk_responses: Pre-configured chunk responses.
        merge_responses: Pre-configured merge responses.
        challenge_responses: Pre-configured challenge responses.
        followup_responses: Pre-configured follow-up responses.
        default_confidence: Default confidence score.
        raise_issues: Whether challenger should raise issues.
        issue_count: Number of issues to raise.
        issue_severity: Severity of raised issues.
        cross_cutting_issues: Whether to create cross-cutting issues.
    """

    chunk_responses: list[dict[str, Any]] = field(default_factory=list)
    merge_responses: list[dict[str, Any]] = field(default_factory=list)
    challenge_responses: list[dict[str, Any]] = field(default_factory=list)
    followup_responses: list[dict[str, Any]] = field(default_factory=list)
    default_confidence: float = 0.9
    raise_issues: bool = True
    issue_count: int = 2
    issue_severity: IssueSeverity = IssueSeverity.MAJOR
    cross_cutting_issues: bool = False


class MockAgentHarness:
    """Comprehensive mock agent simulation harness.

    Provides a central point for configuring and managing mock agents
    in integration tests.

    Example:
        >>> harness = MockAgentHarness(seed=42)
        >>> harness.configure_delay(min_ms=10, max_ms=100)
        >>> harness.configure_failure_rate(0.05)
        >>> mock_llm = harness.create_mock_llm()
        >>> # Run tests...
        >>> print(harness.stats.success_rate)
    """

    def __init__(self, seed: int | None = None) -> None:
        """Initialize the mock agent harness.

        Args:
            seed: Random seed for reproducible behavior.
        """
        self._seed = seed
        self._rng = random.Random(seed)
        self._delay_config = DelayConfig()
        self._failure_config = FailureConfig()
        self._response_config = ResponseConfig()
        self._stats = AgentStats()
        self._call_history: list[dict[str, Any]] = []
        self._custom_handlers: dict[AgentType, Callable] = {}

    @property
    def stats(self) -> AgentStats:
        """Get agent statistics."""
        return self._stats

    @property
    def call_history(self) -> list[dict[str, Any]]:
        """Get complete call history."""
        return self._call_history.copy()

    def reset(self) -> None:
        """Reset all statistics and call history."""
        self._stats = AgentStats()
        self._call_history.clear()
        if self._seed is not None:
            self._rng = random.Random(self._seed)

    def configure_delay(
        self,
        min_ms: float = 0.0,
        max_ms: float = 0.0,
        enabled: bool = True,
    ) -> None:
        """Configure simulated delays.

        Args:
            min_ms: Minimum delay in milliseconds.
            max_ms: Maximum delay in milliseconds.
            enabled: Whether to enable delays.
        """
        self._delay_config.enabled = enabled
        self._delay_config.min_ms = min_ms
        self._delay_config.max_ms = max_ms

    def configure_delay_for_type(
        self,
        agent_type: AgentType,
        min_ms: float,
        max_ms: float,
    ) -> None:
        """Configure delays for a specific agent type.

        Args:
            agent_type: The agent type.
            min_ms: Minimum delay in milliseconds.
            max_ms: Maximum delay in milliseconds.
        """
        self._delay_config.per_type[agent_type] = (min_ms, max_ms)

    def configure_failure_rate(self, rate: float) -> None:
        """Configure random failure rate.

        Args:
            rate: Probability of failure (0.0 to 1.0).
        """
        self._failure_config.mode = FailureMode.RANDOM
        self._failure_config.rate = rate

    def configure_specific_failures(self, call_numbers: list[int]) -> None:
        """Configure specific calls to fail.

        Args:
            call_numbers: List of call numbers (1-indexed) to fail.
        """
        self._failure_config.mode = FailureMode.SPECIFIC
        self._failure_config.specific_calls = call_numbers

    def configure_intermittent_failures(
        self,
        rate: float,
        call_numbers: list[int],
    ) -> None:
        """Configure intermittent failures on specific calls.

        Args:
            rate: Probability of failure on specified calls.
            call_numbers: Calls subject to intermittent failure.
        """
        self._failure_config.mode = FailureMode.INTERMITTENT
        self._failure_config.rate = rate
        self._failure_config.specific_calls = call_numbers

    def configure_no_failures(self) -> None:
        """Disable all failure simulation."""
        self._failure_config.mode = FailureMode.NONE
        self._failure_config.rate = 0.0
        self._failure_config.specific_calls.clear()

    def configure_responses(
        self,
        raise_issues: bool = True,
        issue_count: int = 2,
        issue_severity: IssueSeverity = IssueSeverity.MAJOR,
        cross_cutting: bool = False,
    ) -> None:
        """Configure response generation.

        Args:
            raise_issues: Whether challenger raises issues.
            issue_count: Number of issues to raise.
            issue_severity: Severity of issues.
            cross_cutting: Whether issues are cross-cutting.
        """
        self._response_config.raise_issues = raise_issues
        self._response_config.issue_count = issue_count
        self._response_config.issue_severity = issue_severity
        self._response_config.cross_cutting_issues = cross_cutting

    def configure_no_issues(self) -> None:
        """Configure challenger to raise no issues."""
        self._response_config.raise_issues = False
        self._response_config.issue_count = 0

    def set_chunk_responses(self, responses: list[dict[str, Any]]) -> None:
        """Set pre-configured chunk responses.

        Args:
            responses: List of response dictionaries.
        """
        self._response_config.chunk_responses = responses

    def set_merge_responses(self, responses: list[dict[str, Any]]) -> None:
        """Set pre-configured merge responses.

        Args:
            responses: List of response dictionaries.
        """
        self._response_config.merge_responses = responses

    def set_challenge_responses(self, responses: list[dict[str, Any]]) -> None:
        """Set pre-configured challenge responses.

        Args:
            responses: List of response dictionaries.
        """
        self._response_config.challenge_responses = responses

    def set_followup_responses(self, responses: list[dict[str, Any]]) -> None:
        """Set pre-configured follow-up responses.

        Args:
            responses: List of response dictionaries.
        """
        self._response_config.followup_responses = responses

    def set_custom_handler(
        self,
        agent_type: AgentType,
        handler: Callable[[list[LLMMessage]], str],
    ) -> None:
        """Set a custom response handler for an agent type.

        Args:
            agent_type: The agent type.
            handler: Function that takes messages and returns response.
        """
        self._custom_handlers[agent_type] = handler

    def create_mock_llm(self) -> "HarnessLLMAdapter":
        """Create a mock LLM adapter tied to this harness.

        Returns:
            HarnessLLMAdapter instance.
        """
        return HarnessLLMAdapter(self)

    async def _apply_delay(self, agent_type: AgentType | None = None) -> float:
        """Apply configured delay.

        Args:
            agent_type: Optional agent type for type-specific delays.

        Returns:
            Actual delay applied in milliseconds.
        """
        if not self._delay_config.enabled:
            return 0.0

        # Check for type-specific delay
        if agent_type and agent_type in self._delay_config.per_type:
            min_ms, max_ms = self._delay_config.per_type[agent_type]
        else:
            min_ms = self._delay_config.min_ms
            max_ms = self._delay_config.max_ms

        if max_ms <= 0:
            return 0.0

        delay_ms = self._rng.uniform(min_ms, max_ms)
        await asyncio.sleep(delay_ms / 1000.0)
        return delay_ms

    def _should_fail(self, call_number: int) -> bool:
        """Determine if this call should fail.

        Args:
            call_number: Current call number (1-indexed).

        Returns:
            True if call should fail.
        """
        mode = self._failure_config.mode

        if mode == FailureMode.NONE:
            return False
        elif mode == FailureMode.RANDOM:
            return self._rng.random() < self._failure_config.rate
        elif mode == FailureMode.SPECIFIC:
            return call_number in self._failure_config.specific_calls
        elif mode == FailureMode.INTERMITTENT:
            if call_number in self._failure_config.specific_calls:
                return self._rng.random() < self._failure_config.rate
            return False

        return False

    def _record_call(
        self,
        agent_type: AgentType,
        messages: list[LLMMessage],
        response: str | None,
        success: bool,
        latency_ms: float,
    ) -> None:
        """Record a call for statistics and history.

        Args:
            agent_type: Type of agent called.
            messages: Input messages.
            response: Response content (if successful).
            success: Whether call succeeded.
            latency_ms: Call latency.
        """
        self._stats.total_calls += 1
        if success:
            self._stats.successful_calls += 1
        else:
            self._stats.failed_calls += 1

        self._stats.total_latency_ms += latency_ms

        if agent_type not in self._stats.calls_by_type:
            self._stats.calls_by_type[agent_type] = 0
        self._stats.calls_by_type[agent_type] += 1

        self._call_history.append({
            "call_number": self._stats.total_calls,
            "agent_type": agent_type.value,
            "success": success,
            "latency_ms": latency_ms,
            "message_count": len(messages),
            "response_length": len(response) if response else 0,
        })


class HarnessLLMAdapter(LLMAdapter):
    """LLM adapter tied to a MockAgentHarness.

    This adapter uses the harness configuration for delays,
    failures, and response generation.
    """

    def __init__(self, harness: MockAgentHarness) -> None:
        """Initialize the harness LLM adapter.

        Args:
            harness: The parent harness.
        """
        self._harness = harness
        self._chunk_index = 0
        self._merge_index = 0
        self._challenge_index = 0
        self._followup_index = 0

    def _detect_agent_type(self, messages: list[LLMMessage]) -> AgentType:
        """Detect agent type from message content.

        Args:
            messages: Input messages.

        Returns:
            Detected agent type.
        """
        message_text = " ".join(m.content.lower() for m in messages)

        if "followup" in message_text or "follow-up" in message_text:
            return AgentType.FOLLOWUP
        elif "challenge" in message_text or "review" in message_text:
            return AgentType.CHALLENGER
        elif "merge" in message_text:
            return AgentType.AGGREGATOR
        else:
            return AgentType.SCRIBE

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

        Raises:
            RuntimeError: If configured to fail.
        """
        start_time = time.perf_counter()
        agent_type = self._detect_agent_type(messages)
        call_number = self._harness.stats.total_calls + 1

        # Apply delay
        delay_ms = await self._harness._apply_delay(agent_type)

        # Check for failure
        if self._harness._should_fail(call_number):
            error_msg = self._harness._failure_config.error_messages.get(
                call_number,
                f"Simulated failure on call {call_number}",
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._harness._record_call(
                agent_type, messages, None, False, elapsed_ms
            )
            raise RuntimeError(error_msg)

        # Generate response
        content = self._generate_response(messages, agent_type)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._harness._record_call(
            agent_type, messages, content, True, elapsed_ms
        )

        return LLMResponse(
            content=content,
            usage={
                "prompt_tokens": sum(len(m.content) // 4 for m in messages),
                "completion_tokens": len(content) // 4,
                "total_tokens": (
                    sum(len(m.content) // 4 for m in messages) + len(content) // 4
                ),
            },
            model="mock-harness-llm",
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
            schema: Optional JSON schema (ignored).
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
            Approximate token count.
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
        return total + len(messages) * 4

    def get_context_limit(self) -> int:
        """Get context window size.

        Returns:
            Mock context limit.
        """
        return 8000

    @property
    def model_name(self) -> str:
        """Get model name.

        Returns:
            Mock model name.
        """
        return "mock-harness-llm"

    def _generate_response(
        self,
        messages: list[LLMMessage],
        agent_type: AgentType,
    ) -> str:
        """Generate appropriate response based on agent type.

        Args:
            messages: Input messages.
            agent_type: Detected agent type.

        Returns:
            Response content string.
        """
        # Check for custom handler
        if agent_type in self._harness._custom_handlers:
            return self._harness._custom_handlers[agent_type](messages)

        config = self._harness._response_config

        if agent_type == AgentType.SCRIBE:
            return self._generate_chunk_response(config)
        elif agent_type == AgentType.AGGREGATOR:
            return self._generate_merge_response(config)
        elif agent_type == AgentType.CHALLENGER:
            return self._generate_challenge_response(config)
        elif agent_type == AgentType.FOLLOWUP:
            return self._generate_followup_response(config)

        return json.dumps({"response": "Mock response"})

    def _generate_chunk_response(self, config: ResponseConfig) -> str:
        """Generate chunk analysis response."""
        if config.chunk_responses:
            response = config.chunk_responses[
                self._chunk_index % len(config.chunk_responses)
            ]
            self._chunk_index += 1
            return json.dumps(response)

        result = ChunkResult(
            job_id="mock-job",
            artifact_id="mock-artifact",
            artifact_version="mock-version",
            chunk_id=f"chunk-{self._chunk_index}",
            chunk_locator={"start_line": 1 + self._chunk_index * 50, "end_line": 50 + self._chunk_index * 50},
            chunk_kind="procedure_part",
            summary=f"Chunk {self._chunk_index} contains processing logic.",
            facts=ChunkFacts(
                symbols_defined=[
                    SymbolDef(name=f"WS-VAR-{self._chunk_index}", kind="variable", line_number=10),
                ],
                symbols_used=[f"WS-VAR-{self._chunk_index}", "INPUT-RECORD"],
                entrypoints=[f"PROCESS-{self._chunk_index}"],
                paragraphs_defined=[f"PROCESS-{self._chunk_index}"],
                io_operations=[
                    IOOperation(
                        operation="READ",
                        file_name="INPUT-FILE",
                        line_number=30 + self._chunk_index * 50,
                        status_check=True,
                    )
                ],
            ),
            evidence=[
                Evidence(
                    evidence_type="line_range",
                    start_line=1 + self._chunk_index * 50,
                    end_line=50 + self._chunk_index * 50,
                    note=f"Processing logic for chunk {self._chunk_index}",
                )
            ],
            confidence=config.default_confidence,
        )
        self._chunk_index += 1
        return json.dumps(result.model_dump())

    def _generate_merge_response(self, config: ResponseConfig) -> str:
        """Generate merge response."""
        if config.merge_responses:
            response = config.merge_responses[
                self._merge_index % len(config.merge_responses)
            ]
            self._merge_index += 1
            return json.dumps(response)

        result = MergeResult(
            job_id="mock-job",
            artifact_id="mock-artifact",
            artifact_version="mock-version",
            merge_node_id=f"merge-{self._merge_index}",
            coverage=MergeCoverage(
                included_input_ids=[f"chunk-{i}" for i in range(self._merge_index + 1)],
                missing_input_ids=[],
            ),
            consolidated_facts=ConsolidatedFacts(
                call_graph_edges=[
                    {"from": "MAIN-PROCESS", "to": f"PROCESS-{i}"}
                    for i in range(self._merge_index + 1)
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

    def _generate_challenge_response(self, config: ResponseConfig) -> str:
        """Generate challenge response."""
        if config.challenge_responses:
            response = config.challenge_responses[
                self._challenge_index % len(config.challenge_responses)
            ]
            self._challenge_index += 1
            return json.dumps(response)

        issues = []
        if config.raise_issues:
            for i in range(config.issue_count):
                issue = Issue(
                    issue_id=f"issue-{self._challenge_index}-{i}",
                    severity=config.issue_severity,
                    question=f"Issue {i}: Clarify the error handling logic.",
                    doc_section_refs=["section-1"],
                    suspected_scopes=[] if config.cross_cutting_issues else [f"chunk-{i}"],
                    routing_hints={"paragraphs": [f"PROCESS-{i}"]} if not config.cross_cutting_issues else {},
                )
                issues.append(issue)

        followup_tasks = []
        for issue in issues:
            if issue.severity in [IssueSeverity.BLOCKER, IssueSeverity.MAJOR]:
                followup_tasks.append(
                    FollowupTask(
                        issue_id=issue.issue_id,
                        scope={"chunk_ids": issue.suspected_scopes} if issue.suspected_scopes else {},
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

    def _generate_followup_response(self, config: ResponseConfig) -> str:
        """Generate follow-up response."""
        if config.followup_responses:
            response = config.followup_responses[
                self._followup_index % len(config.followup_responses)
            ]
            self._followup_index += 1
            return json.dumps(response)

        result = FollowupAnswer(
            issue_id=f"issue-{self._followup_index}",
            scope={"chunk_ids": [f"chunk-{self._followup_index}"]},
            answer=(
                f"Follow-up answer {self._followup_index}: "
                "Error handling is implemented via FILE-STATUS checks."
            ),
            facts=ChunkFacts(
                symbols_defined=[],
                symbols_used=["FILE-STATUS"],
            ),
            evidence=[
                Evidence(
                    evidence_type="line_range",
                    start_line=100 + self._followup_index * 10,
                    end_line=110 + self._followup_index * 10,
                    note="Error handling implementation",
                )
            ],
            confidence=0.95,
        )
        self._followup_index += 1
        return json.dumps(result.model_dump())


# Convenience factory functions

def create_harness_with_delays(
    min_ms: float = 10,
    max_ms: float = 50,
    seed: int | None = None,
) -> MockAgentHarness:
    """Create harness with simulated delays.

    Args:
        min_ms: Minimum delay in milliseconds.
        max_ms: Maximum delay in milliseconds.
        seed: Random seed.

    Returns:
        Configured MockAgentHarness.
    """
    harness = MockAgentHarness(seed=seed)
    harness.configure_delay(min_ms=min_ms, max_ms=max_ms)
    return harness


def create_harness_with_failures(
    failure_rate: float = 0.1,
    seed: int | None = None,
) -> MockAgentHarness:
    """Create harness with simulated failures.

    Args:
        failure_rate: Probability of failure.
        seed: Random seed.

    Returns:
        Configured MockAgentHarness.
    """
    harness = MockAgentHarness(seed=seed)
    harness.configure_failure_rate(failure_rate)
    return harness


def create_harness_no_issues(seed: int | None = None) -> MockAgentHarness:
    """Create harness configured for no challenger issues.

    Args:
        seed: Random seed.

    Returns:
        Configured MockAgentHarness.
    """
    harness = MockAgentHarness(seed=seed)
    harness.configure_no_issues()
    return harness


def create_harness_with_cross_cutting_issues(
    issue_count: int = 2,
    seed: int | None = None,
) -> MockAgentHarness:
    """Create harness with cross-cutting challenger issues.

    Args:
        issue_count: Number of issues.
        seed: Random seed.

    Returns:
        Configured MockAgentHarness.
    """
    harness = MockAgentHarness(seed=seed)
    harness.configure_responses(
        raise_issues=True,
        issue_count=issue_count,
        cross_cutting=True,
    )
    return harness
