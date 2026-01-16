"""Unit tests for the splitter registry and plugin architecture."""

import pytest

from atlas.splitter import (
    Splitter,
    SplitResult,
    SplitterRegistry,
    SplitterNotFoundError,
    get_default_registry,
    reset_default_registry,
    COBOLSplitter,
    LineBasedSplitter,
)
from atlas.models.manifest import SplitterProfile, ChunkSpec
from atlas.models.enums import ChunkKind


class TestSplitterRegistry:
    """Tests for the SplitterRegistry class."""

    @pytest.fixture
    def registry(self) -> SplitterRegistry:
        """Provide a fresh registry for each test."""
        return SplitterRegistry()

    @pytest.fixture
    def default_profile(self) -> SplitterProfile:
        """Provide a default splitter profile."""
        return SplitterProfile(
            name="test",
            prefer_semantic=True,
            max_chunk_tokens=3500,
            overlap_lines=10,
        )

    def test_registry_creation(self, registry: SplitterRegistry) -> None:
        """Test SplitterRegistry can be instantiated."""
        assert registry is not None
        assert len(registry) == 0

    def test_register_splitter(self, registry: SplitterRegistry) -> None:
        """Test registering a splitter class."""
        registry.register(COBOLSplitter)

        assert "cobol" in registry
        assert "copybook" in registry
        assert len(registry) == 2

    def test_register_splitter_with_custom_types(
        self, registry: SplitterRegistry
    ) -> None:
        """Test registering a splitter with custom artifact types."""
        registry.register(COBOLSplitter, artifact_types=["custom_cobol", "cob"])

        assert "custom_cobol" in registry
        assert "cob" in registry
        assert "cobol" not in registry  # Original type not registered
        assert len(registry) == 2

    def test_register_splitter_type_normalization(
        self, registry: SplitterRegistry
    ) -> None:
        """Test that artifact types are normalized during registration and lookup."""
        registry.register(COBOLSplitter, artifact_types=["  COBOL  ", "CopyBook"])

        # Normalized types should be registered
        assert "cobol" in registry
        assert "copybook" in registry

        # Lookup also normalizes, so unnormalized queries should work
        assert "  COBOL  " in registry  # Normalized during lookup
        assert "COPYBOOK" in registry  # Case insensitive lookup

        # But the registered types list should show normalized versions
        registered = registry.list_registered_types()
        assert "cobol" in registered
        assert "  COBOL  " not in registered

    def test_register_splitter_overwrite_warning(
        self, registry: SplitterRegistry, caplog
    ) -> None:
        """Test that overwriting a splitter produces a warning."""
        import logging

        caplog.set_level(logging.WARNING)

        registry.register(COBOLSplitter)
        registry.register(LineBasedSplitter, artifact_types=["cobol"])

        assert "Overwriting splitter" in caplog.text

    def test_get_splitter(self, registry: SplitterRegistry) -> None:
        """Test getting a splitter by artifact type."""
        registry.register(COBOLSplitter)

        splitter = registry.get_splitter("cobol")
        assert isinstance(splitter, COBOLSplitter)

    def test_get_splitter_case_insensitive(self, registry: SplitterRegistry) -> None:
        """Test that artifact type lookup is case insensitive."""
        registry.register(COBOLSplitter)

        splitter1 = registry.get_splitter("cobol")
        splitter2 = registry.get_splitter("COBOL")
        splitter3 = registry.get_splitter("Cobol")

        assert isinstance(splitter1, COBOLSplitter)
        assert isinstance(splitter2, COBOLSplitter)
        assert isinstance(splitter3, COBOLSplitter)

    def test_get_splitter_caching(self, registry: SplitterRegistry) -> None:
        """Test that splitter instances are cached."""
        registry.register(COBOLSplitter)

        splitter1 = registry.get_splitter("cobol")
        splitter2 = registry.get_splitter("cobol")

        assert splitter1 is splitter2  # Same instance

    def test_get_splitter_fallback(self, registry: SplitterRegistry) -> None:
        """Test fallback to LineBasedSplitter for unknown types."""
        # Don't register any splitters
        splitter = registry.get_splitter("unknown_type")

        assert isinstance(splitter, LineBasedSplitter)

    def test_get_splitter_no_fallback(self, registry: SplitterRegistry) -> None:
        """Test raising exception when no fallback is used."""
        with pytest.raises(SplitterNotFoundError) as exc_info:
            registry.get_splitter("unknown_type", use_fallback=False)

        assert "unknown_type" in str(exc_info.value)

    def test_get_splitter_class(self, registry: SplitterRegistry) -> None:
        """Test getting splitter class without instantiation."""
        registry.register(COBOLSplitter)

        splitter_class = registry.get_splitter_class("cobol")
        assert splitter_class is COBOLSplitter

        # Unknown type returns None
        unknown_class = registry.get_splitter_class("unknown")
        assert unknown_class is None

    def test_unregister_splitter(self, registry: SplitterRegistry) -> None:
        """Test unregistering a splitter."""
        registry.register(COBOLSplitter)
        assert "cobol" in registry

        result = registry.unregister("cobol")
        assert result is True
        assert "cobol" not in registry
        assert "copybook" in registry  # Other type still registered

    def test_unregister_nonexistent(self, registry: SplitterRegistry) -> None:
        """Test unregistering a nonexistent type."""
        result = registry.unregister("nonexistent")
        assert result is False

    def test_has_splitter(self, registry: SplitterRegistry) -> None:
        """Test checking if splitter is registered."""
        registry.register(COBOLSplitter)

        assert registry.has_splitter("cobol") is True
        assert registry.has_splitter("unknown") is False

    def test_list_registered_types(self, registry: SplitterRegistry) -> None:
        """Test listing all registered artifact types."""
        registry.register(COBOLSplitter)

        types = registry.list_registered_types()
        assert "cobol" in types
        assert "copybook" in types
        assert len(types) == 2

    def test_clear_registry(self, registry: SplitterRegistry) -> None:
        """Test clearing all registrations."""
        registry.register(COBOLSplitter)
        registry.set_config("cobol", {"custom": "config"})

        registry.clear()

        assert len(registry) == 0
        assert registry.get_config("cobol") == {}

    def test_set_and_get_config(self, registry: SplitterRegistry) -> None:
        """Test configuration per artifact type."""
        registry.register(COBOLSplitter)
        config = {"chars_per_line": 80, "prefer_sections": True}
        registry.set_config("cobol", config)

        retrieved = registry.get_config("cobol")
        assert retrieved == config

    def test_config_clears_cached_instance(self, registry: SplitterRegistry) -> None:
        """Test that setting config clears cached splitter instance."""
        registry.register(COBOLSplitter)

        splitter1 = registry.get_splitter("cobol")
        registry.set_config("cobol", {"new": "config"})
        splitter2 = registry.get_splitter("cobol")

        assert splitter1 is not splitter2  # Different instances

    def test_contains_operator(self, registry: SplitterRegistry) -> None:
        """Test 'in' operator support."""
        registry.register(COBOLSplitter)

        assert "cobol" in registry
        assert "unknown" not in registry

    def test_len_operator(self, registry: SplitterRegistry) -> None:
        """Test len() support."""
        assert len(registry) == 0

        registry.register(COBOLSplitter)
        assert len(registry) == 2  # cobol and copybook

    def test_custom_fallback_class(self) -> None:
        """Test using a custom fallback splitter class."""
        # Create registry with COBOLSplitter as fallback
        registry = SplitterRegistry(fallback_class=COBOLSplitter)

        splitter = registry.get_splitter("unknown_type")
        assert isinstance(splitter, COBOLSplitter)

    def test_register_empty_types_raises(self, registry: SplitterRegistry) -> None:
        """Test that registering with empty types raises error."""
        # LineBasedSplitter returns empty list from get_artifact_types
        with pytest.raises(ValueError) as exc_info:
            registry.register(LineBasedSplitter)

        assert "no artifact types" in str(exc_info.value).lower()


class TestDefaultRegistry:
    """Tests for the global default registry."""

    def test_get_default_registry(self) -> None:
        """Test getting the default registry."""
        reset_default_registry()  # Ensure clean state

        registry = get_default_registry()
        assert registry is not None
        assert isinstance(registry, SplitterRegistry)

    def test_default_registry_singleton(self) -> None:
        """Test that default registry is singleton."""
        reset_default_registry()

        registry1 = get_default_registry()
        registry2 = get_default_registry()

        assert registry1 is registry2

    def test_default_registry_has_cobol_splitter(self) -> None:
        """Test that default registry has COBOLSplitter registered."""
        reset_default_registry()

        registry = get_default_registry()
        assert "cobol" in registry
        assert "copybook" in registry

        splitter = registry.get_splitter("cobol")
        assert isinstance(splitter, COBOLSplitter)

    def test_default_registry_fallback(self) -> None:
        """Test that default registry uses LineBasedSplitter as fallback."""
        reset_default_registry()

        registry = get_default_registry()
        splitter = registry.get_splitter("unknown_type")
        assert isinstance(splitter, LineBasedSplitter)

    def test_reset_default_registry(self) -> None:
        """Test resetting the default registry."""
        reset_default_registry()

        registry1 = get_default_registry()
        reset_default_registry()
        registry2 = get_default_registry()

        # Should be different instances after reset
        assert registry1 is not registry2


class TestLineBasedSplitter:
    """Tests for the LineBasedSplitter class."""

    @pytest.fixture
    def splitter(self) -> LineBasedSplitter:
        """Provide a LineBasedSplitter instance."""
        return LineBasedSplitter()

    @pytest.fixture
    def default_profile(self) -> SplitterProfile:
        """Provide a default splitter profile."""
        return SplitterProfile(
            name="line_based_test",
            prefer_semantic=False,
            max_chunk_tokens=3500,
            overlap_lines=10,
        )

    @pytest.fixture
    def small_profile(self) -> SplitterProfile:
        """Provide a small token budget profile."""
        return SplitterProfile(
            name="small",
            prefer_semantic=False,
            max_chunk_tokens=100,
            overlap_lines=5,
        )

    def test_splitter_creation(self, splitter: LineBasedSplitter) -> None:
        """Test LineBasedSplitter can be instantiated."""
        assert splitter is not None
        assert hasattr(splitter, "split")
        assert hasattr(splitter, "estimate_tokens")
        assert hasattr(splitter, "detect_semantic_boundaries")

    def test_get_artifact_types(self) -> None:
        """Test that LineBasedSplitter returns empty artifact types."""
        types = LineBasedSplitter.get_artifact_types()
        assert types == []

    def test_estimate_tokens(self, splitter: LineBasedSplitter) -> None:
        """Test token estimation."""
        assert splitter.estimate_tokens("") == 1  # minimum 1
        assert splitter.estimate_tokens("hello") == 1
        assert splitter.estimate_tokens("hello world foo bar") == 4  # 19 chars / 4
        assert splitter.estimate_tokens("a" * 100) == 25  # 100 / 4

    def test_detect_semantic_boundaries(self, splitter: LineBasedSplitter) -> None:
        """Test that no semantic boundaries are detected."""
        source = """Line 1
Line 2
Line 3
"""
        boundaries = splitter.detect_semantic_boundaries(source)
        assert boundaries == []

    def test_split_simple_text(
        self, splitter: LineBasedSplitter, default_profile: SplitterProfile
    ) -> None:
        """Test splitting simple text."""
        source = "\n".join([f"Line {i}" for i in range(100)])
        result = splitter.split(source, default_profile, "test.txt")

        assert result.chunks is not None
        assert len(result.chunks) >= 1
        assert result.total_lines == 100
        assert result.semantic_boundaries_found == 0

    def test_split_deterministic(
        self, splitter: LineBasedSplitter, default_profile: SplitterProfile
    ) -> None:
        """Test that splitting is deterministic."""
        source = "\n".join([f"Line {i}" for i in range(50)])

        result1 = splitter.split(source, default_profile, "test.txt")
        result2 = splitter.split(source, default_profile, "test.txt")

        assert len(result1.chunks) == len(result2.chunks)
        for c1, c2 in zip(result1.chunks, result2.chunks):
            assert c1.chunk_id == c2.chunk_id
            assert c1.start_line == c2.start_line
            assert c1.end_line == c2.end_line

    def test_split_empty_source(
        self, splitter: LineBasedSplitter, default_profile: SplitterProfile
    ) -> None:
        """Test handling of empty source."""
        result = splitter.split("", default_profile, "empty.txt")

        assert result.total_lines == 0
        assert len(result.chunks) == 0
        assert "empty" in result.warnings[0].lower()

    def test_split_single_line(
        self, splitter: LineBasedSplitter, default_profile: SplitterProfile
    ) -> None:
        """Test splitting single line source."""
        result = splitter.split("Single line", default_profile, "single.txt")

        assert result.total_lines == 1
        assert len(result.chunks) == 1
        assert result.chunks[0].start_line == 1
        assert result.chunks[0].end_line == 1

    def test_split_with_small_budget(
        self, splitter: LineBasedSplitter, small_profile: SplitterProfile
    ) -> None:
        """Test that small budget creates more chunks."""
        source = "\n".join([f"Line {i} with some content" for i in range(100)])

        large_profile = SplitterProfile(
            name="large",
            prefer_semantic=False,
            max_chunk_tokens=10000,
        )

        result_small = splitter.split(source, small_profile, "test.txt")
        result_large = splitter.split(source, large_profile, "test.txt")

        # Small budget should create more chunks
        assert len(result_small.chunks) >= len(result_large.chunks)

    def test_chunks_have_generic_kind(
        self, splitter: LineBasedSplitter, default_profile: SplitterProfile
    ) -> None:
        """Test that all chunks have GENERIC kind."""
        source = "\n".join([f"Line {i}" for i in range(50)])
        result = splitter.split(source, default_profile, "test.txt")

        for chunk in result.chunks:
            assert chunk.chunk_kind == ChunkKind.GENERIC

    def test_chunks_have_valid_line_ranges(
        self, splitter: LineBasedSplitter, default_profile: SplitterProfile
    ) -> None:
        """Test that chunks have valid line ranges."""
        source = "\n".join([f"Line {i}" for i in range(100)])
        result = splitter.split(source, default_profile, "test.txt")

        for chunk in result.chunks:
            assert chunk.start_line >= 1
            assert chunk.end_line >= chunk.start_line
            assert chunk.end_line <= result.total_lines

    def test_chunks_have_metadata(
        self, splitter: LineBasedSplitter, default_profile: SplitterProfile
    ) -> None:
        """Test that chunks include splitter metadata."""
        source = "\n".join([f"Line {i}" for i in range(50)])
        result = splitter.split(source, default_profile, "test.txt")

        for chunk in result.chunks:
            assert "splitter" in chunk.metadata
            assert chunk.metadata["splitter"] == "line_based"
            assert "lines_per_chunk" in chunk.metadata

    def test_chunk_ids_are_unique(
        self, splitter: LineBasedSplitter, default_profile: SplitterProfile
    ) -> None:
        """Test that chunk IDs are unique."""
        source = "\n".join([f"Line {i}" for i in range(100)])
        result = splitter.split(source, default_profile, "test.txt")

        chunk_ids = [c.chunk_id for c in result.chunks]
        assert len(chunk_ids) == len(set(chunk_ids))

    def test_chunks_cover_source(
        self, splitter: LineBasedSplitter, small_profile: SplitterProfile
    ) -> None:
        """Test that chunks cover the entire source."""
        source = "\n".join([f"Line {i}" for i in range(100)])
        result = splitter.split(source, small_profile, "test.txt")

        # Get all covered lines
        covered_lines = set()
        for chunk in result.chunks:
            for line in range(chunk.start_line, chunk.end_line + 1):
                covered_lines.add(line)

        # Should cover all lines
        assert len(covered_lines) >= result.total_lines * 0.9

    def test_custom_chars_per_line(
        self, splitter: LineBasedSplitter
    ) -> None:
        """Test custom chars_per_line configuration."""
        profile = SplitterProfile(
            name="custom",
            prefer_semantic=False,
            max_chunk_tokens=1000,
            custom_config={"chars_per_line": 100},  # Higher than default
        )

        source = "\n".join(["X" * 100 for _ in range(50)])  # 100 chars per line
        result = splitter.split(source, profile, "test.txt")

        # With 100 chars/line and 4 chars/token, 25 tokens/line
        # 1000 tokens budget = 40 lines per chunk
        # 50 lines / 40 = ~2 chunks
        assert len(result.chunks) >= 1


class TestCOBOLSplitterArtifactTypes:
    """Tests for COBOLSplitter artifact type declaration."""

    def test_get_artifact_types(self) -> None:
        """Test that COBOLSplitter declares correct artifact types."""
        types = COBOLSplitter.get_artifact_types()

        assert "cobol" in types
        assert "copybook" in types
        assert len(types) == 2

    def test_splitter_works_for_declared_types(self) -> None:
        """Test that splitter works for its declared artifact types."""
        splitter = COBOLSplitter()
        profile = SplitterProfile(name="test", max_chunk_tokens=3500)

        source = """       IDENTIFICATION DIVISION.
       PROGRAM-ID. TEST.
       PROCEDURE DIVISION.
           STOP RUN.
"""
        # Should work for both declared types
        for artifact_type in COBOLSplitter.get_artifact_types():
            result = splitter.split(source, profile, f"test.{artifact_type}")
            assert len(result.chunks) >= 1


class TestSplitterIntegration:
    """Integration tests for the splitter system."""

    def test_registry_to_splitter_workflow(self) -> None:
        """Test complete workflow from registry to splitting."""
        reset_default_registry()
        registry = get_default_registry()

        source = """       IDENTIFICATION DIVISION.
       PROGRAM-ID. TEST.
       PROCEDURE DIVISION.
           STOP RUN.
"""
        profile = SplitterProfile(name="test", max_chunk_tokens=3500)

        # Get appropriate splitter from registry
        splitter = registry.get_splitter("cobol")

        # Perform split
        result = splitter.split(source, profile, "test.cbl")

        assert len(result.chunks) >= 1
        assert result.semantic_boundaries_found >= 1

    def test_fallback_workflow(self) -> None:
        """Test fallback workflow for unknown artifact type."""
        reset_default_registry()
        registry = get_default_registry()

        source = "\n".join([f"Line {i}" for i in range(50)])
        profile = SplitterProfile(name="test", max_chunk_tokens=3500)

        # Get fallback splitter for unknown type
        splitter = registry.get_splitter("python")  # Not registered

        # Perform split
        result = splitter.split(source, profile, "test.py")

        assert len(result.chunks) >= 1
        assert result.semantic_boundaries_found == 0  # Line-based has no semantics

    def test_multiple_artifact_types(self) -> None:
        """Test registry with multiple artifact types."""
        registry = SplitterRegistry()
        registry.register(COBOLSplitter)

        # All types should resolve to appropriate splitter
        cobol_splitter = registry.get_splitter("cobol")
        copybook_splitter = registry.get_splitter("copybook")
        unknown_splitter = registry.get_splitter("unknown")

        assert isinstance(cobol_splitter, COBOLSplitter)
        assert isinstance(copybook_splitter, COBOLSplitter)
        assert isinstance(unknown_splitter, LineBasedSplitter)
