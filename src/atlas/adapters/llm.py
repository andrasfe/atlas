"""Abstract base class for LLM adapters.

LLM adapters provide integration with language model providers
(OpenAI, Anthropic, local models, etc.) for the analysis workflow.

Key Requirements:
- Structured output support (JSON mode)
- Token counting for context budget enforcement
- Async execution for concurrent processing
- Error handling and retries
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class LLMRole(str, Enum):
    """Role for LLM messages."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class LLMMessage:
    """A message in an LLM conversation.

    Attributes:
        role: Message role (system, user, assistant).
        content: Message content.
        name: Optional name for the message author.
    """

    role: LLMRole
    content: str
    name: str | None = None


@dataclass
class LLMResponse:
    """Response from an LLM.

    Attributes:
        content: The generated content.
        finish_reason: Why generation stopped (stop, length, etc.).
        usage: Token usage information.
        model: Model that generated the response.
        raw_response: Raw response from the provider (for debugging).
    """

    content: str
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)
    model: str = ""
    raw_response: dict[str, Any] = field(default_factory=dict)

    @property
    def prompt_tokens(self) -> int:
        """Get prompt token count."""
        return self.usage.get("prompt_tokens", 0)

    @property
    def completion_tokens(self) -> int:
        """Get completion token count."""
        return self.usage.get("completion_tokens", 0)

    @property
    def total_tokens(self) -> int:
        """Get total token count."""
        return self.usage.get("total_tokens", 0)


class LLMAdapter(ABC):
    """Abstract interface for LLM provider integration.

    Implementations should handle:
    - Provider-specific API calls
    - Structured output (JSON mode) support
    - Token counting and context budget management
    - Rate limiting and retries

    Design Principle:
        Bounded context per task. No task may require reading more
        content than fits the configured context budget.

    Example Implementation:
        >>> class OpenAIAdapter(LLMAdapter):
        ...     async def complete(self, messages, **kwargs):
        ...         response = await self.client.chat.completions.create(
        ...             model=self.model,
        ...             messages=[{"role": m.role, "content": m.content} for m in messages],
        ...             **kwargs
        ...         )
        ...         return LLMResponse(
        ...             content=response.choices[0].message.content,
        ...             usage=response.usage.dict()
        ...         )

    TODO: Implement concrete adapters for:
        - OpenAI (GPT-4, GPT-3.5)
        - Anthropic (Claude)
        - Azure OpenAI
        - Local models (Ollama, vLLM)
    """

    @abstractmethod
    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
        stop_sequences: list[str] | None = None,
    ) -> LLMResponse:
        """Generate a completion from the LLM.

        Args:
            messages: Conversation messages.
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum tokens to generate.
            json_mode: If True, request structured JSON output.
            stop_sequences: Sequences that stop generation.

        Returns:
            LLMResponse with generated content and metadata.

        Raises:
            LLMError: If generation fails.
            ContextLengthExceededError: If input exceeds context limit.

        TODO: Implement with proper error handling and retries.
        """
        pass

    @abstractmethod
    async def complete_json(
        self,
        messages: list[LLMMessage],
        schema: dict[str, Any] | None = None,
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Generate structured JSON output.

        Convenience method that parses the response as JSON.

        Args:
            messages: Conversation messages.
            schema: Optional JSON schema for validation.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            Parsed JSON response.

        Raises:
            LLMError: If generation fails.
            JSONDecodeError: If response is not valid JSON.
            ValidationError: If response doesn't match schema.

        TODO: Implement with schema validation.
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Used for context budget enforcement.

        Args:
            text: Text to count tokens for.

        Returns:
            Token count.

        TODO: Implement with provider-specific tokenizer.
        """
        pass

    @abstractmethod
    def count_message_tokens(self, messages: list[LLMMessage]) -> int:
        """Count tokens in a message list.

        Accounts for message formatting overhead.

        Args:
            messages: Messages to count.

        Returns:
            Total token count including overhead.

        TODO: Implement with provider-specific accounting.
        """
        pass

    @abstractmethod
    def get_context_limit(self) -> int:
        """Get the context window size for this model.

        Returns:
            Maximum context window in tokens.

        TODO: Implement based on model configuration.
        """
        pass

    def fits_context(
        self,
        messages: list[LLMMessage],
        reserved_for_output: int = 1000,
    ) -> bool:
        """Check if messages fit within context limit.

        Args:
            messages: Messages to check.
            reserved_for_output: Tokens to reserve for output.

        Returns:
            True if messages fit, False otherwise.
        """
        used = self.count_message_tokens(messages)
        limit = self.get_context_limit()
        return used + reserved_for_output <= limit

    def truncate_to_fit(
        self,
        text: str,
        max_tokens: int,
    ) -> str:
        """Truncate text to fit within token limit.

        Args:
            text: Text to truncate.
            max_tokens: Maximum allowed tokens.

        Returns:
            Truncated text.

        Note:
            Default implementation is approximate. Override for
            accurate truncation with proper tokenization.
        """
        current_tokens = self.count_tokens(text)
        if current_tokens <= max_tokens:
            return text

        # Approximate truncation (should be overridden)
        ratio = max_tokens / current_tokens
        target_chars = int(len(text) * ratio * 0.9)  # 10% safety margin
        return text[:target_chars] + "..."

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name/identifier.

        Returns:
            Model name string.
        """
        pass

    @property
    def supports_json_mode(self) -> bool:
        """Check if model supports native JSON mode.

        Returns:
            True if JSON mode is supported.
        """
        return True  # Override if not supported

    @property
    def supports_function_calling(self) -> bool:
        """Check if model supports function calling.

        Returns:
            True if function calling is supported.
        """
        return False  # Override if supported
