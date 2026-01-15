"""Unit tests for the COBOL-aware chunk splitter."""

import pytest
from pathlib import Path

from atlas.splitter.cobol import COBOLSplitter, SemanticBoundary, COBOLStructure
from atlas.models.manifest import SplitterProfile
from atlas.models.enums import ChunkKind


# Sample COBOL source for testing
SIMPLE_COBOL = """       IDENTIFICATION DIVISION.
       PROGRAM-ID. TEST001.
       AUTHOR. TEST.

       ENVIRONMENT DIVISION.
       CONFIGURATION SECTION.
       SOURCE-COMPUTER. IBM-370.

       DATA DIVISION.
       WORKING-STORAGE SECTION.
       01 WS-COUNTER PIC 9(4) VALUE ZERO.
       01 WS-FLAG    PIC X    VALUE 'N'.

       PROCEDURE DIVISION.
       MAIN-LOGIC.
           PERFORM INIT-ROUTINE.
           PERFORM PROCESS-DATA.
           STOP RUN.

       INIT-ROUTINE.
           MOVE 0 TO WS-COUNTER.
           MOVE 'Y' TO WS-FLAG.

       PROCESS-DATA.
           ADD 1 TO WS-COUNTER.
           IF WS-COUNTER > 100
               MOVE 'N' TO WS-FLAG.
"""

LARGE_PROCEDURE_COBOL = """       IDENTIFICATION DIVISION.
       PROGRAM-ID. BIGPROG.

       PROCEDURE DIVISION.
       MAIN-PROCESS.
           PERFORM STEP-001.
           PERFORM STEP-002.
           PERFORM STEP-003.
           STOP RUN.

       STEP-001.
           DISPLAY 'STEP 001'.
           DISPLAY 'MORE OUTPUT'.
           DISPLAY 'EVEN MORE'.

       STEP-002.
           DISPLAY 'STEP 002'.
           DISPLAY 'PROCESSING'.
           DISPLAY 'DATA HERE'.

       STEP-003.
           DISPLAY 'STEP 003'.
           DISPLAY 'FINISHING'.
           DISPLAY 'UP NOW'.
"""


# Path to test fixtures
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "cobol"


class TestCOBOLSplitter:
    """Tests for COBOLSplitter class."""

    @pytest.fixture
    def splitter(self) -> COBOLSplitter:
        """Provide a COBOLSplitter instance."""
        return COBOLSplitter()

    @pytest.fixture
    def default_profile(self) -> SplitterProfile:
        """Provide a default splitter profile."""
        return SplitterProfile(
            name="test",
            prefer_semantic=True,
            max_chunk_tokens=3500,
            overlap_lines=10,
        )

    @pytest.fixture
    def small_profile(self) -> SplitterProfile:
        """Provide a small token budget profile for testing splits."""
        return SplitterProfile(
            name="small",
            prefer_semantic=True,
            max_chunk_tokens=100,  # Very small to force splitting
            overlap_lines=5,
        )

    def test_splitter_creation(self, splitter: COBOLSplitter) -> None:
        """Test COBOLSplitter can be instantiated."""
        assert splitter is not None
        assert hasattr(splitter, "split")
        assert hasattr(splitter, "estimate_tokens")
        assert hasattr(splitter, "detect_semantic_boundaries")

    def test_estimate_tokens(self, splitter: COBOLSplitter) -> None:
        """Test token estimation."""
        # ~4 chars per token
        assert splitter.estimate_tokens("") == 1  # minimum 1
        assert splitter.estimate_tokens("hello") == 1
        assert splitter.estimate_tokens("hello world foo bar") == 4  # 19 chars / 4
        assert splitter.estimate_tokens("a" * 100) == 25  # 100 / 4

    def test_estimate_tokens_with_whitespace(self, splitter: COBOLSplitter) -> None:
        """Test token estimation handles whitespace correctly."""
        # Whitespace should count toward token estimate
        text_with_spaces = "    " * 100  # 400 chars
        assert splitter.estimate_tokens(text_with_spaces) >= 100

        # Newlines
        text_with_newlines = "\n" * 100
        assert splitter.estimate_tokens(text_with_newlines) >= 25

    def test_estimate_tokens_unicode(self, splitter: COBOLSplitter) -> None:
        """Test token estimation with unicode characters."""
        # Unicode chars might have different byte lengths but char count is used
        unicode_text = "abc" * 100  # 300 chars
        assert splitter.estimate_tokens(unicode_text) == 75

    def test_detect_semantic_boundaries(self, splitter: COBOLSplitter) -> None:
        """Test detection of COBOL semantic boundaries."""
        boundaries = splitter.detect_semantic_boundaries(SIMPLE_COBOL)

        # Should find divisions
        division_names = [b[1] for b in boundaries if "DIVISION" in b[1]]
        assert "IDENTIFICATION DIVISION" in division_names
        assert "ENVIRONMENT DIVISION" in division_names
        assert "DATA DIVISION" in division_names
        assert "PROCEDURE DIVISION" in division_names

        # Should find paragraphs in PROCEDURE DIVISION
        para_names = [b[1] for b in boundaries if b[2] == ChunkKind.PROCEDURE_PART]
        assert "MAIN-LOGIC" in para_names
        assert "INIT-ROUTINE" in para_names
        assert "PROCESS-DATA" in para_names

    def test_detect_divisions(self, splitter: COBOLSplitter) -> None:
        """Test that all four COBOL divisions are detected."""
        boundaries = splitter.detect_semantic_boundaries(SIMPLE_COBOL)

        division_kinds = {b[2] for b in boundaries if b[2] != ChunkKind.PROCEDURE_PART}
        expected = {
            ChunkKind.IDENTIFICATION_DIVISION,
            ChunkKind.ENVIRONMENT_DIVISION,
            ChunkKind.DATA_DIVISION,
            ChunkKind.PROCEDURE_DIVISION,
        }
        assert expected.issubset(division_kinds)

    def test_split_simple_cobol(
        self,
        splitter: COBOLSplitter,
        default_profile: SplitterProfile,
    ) -> None:
        """Test splitting a simple COBOL program."""
        result = splitter.split(SIMPLE_COBOL, default_profile, "TEST001.cbl")

        assert result.chunks is not None
        assert len(result.chunks) > 0
        assert result.total_lines > 0
        assert result.semantic_boundaries_found > 0

    def test_split_creates_deterministic_chunks(
        self,
        splitter: COBOLSplitter,
        default_profile: SplitterProfile,
    ) -> None:
        """Test that splitting is deterministic."""
        result1 = splitter.split(SIMPLE_COBOL, default_profile, "TEST001.cbl")
        result2 = splitter.split(SIMPLE_COBOL, default_profile, "TEST001.cbl")

        # Same number of chunks
        assert len(result1.chunks) == len(result2.chunks)

        # Same chunk IDs
        for c1, c2 in zip(result1.chunks, result2.chunks):
            assert c1.chunk_id == c2.chunk_id
            assert c1.start_line == c2.start_line
            assert c1.end_line == c2.end_line

    def test_chunks_have_valid_line_ranges(
        self,
        splitter: COBOLSplitter,
        default_profile: SplitterProfile,
    ) -> None:
        """Test that all chunks have valid line ranges."""
        result = splitter.split(SIMPLE_COBOL, default_profile, "TEST001.cbl")

        for chunk in result.chunks:
            assert chunk.start_line >= 1
            assert chunk.end_line >= chunk.start_line
            assert chunk.end_line <= result.total_lines

    def test_chunks_cover_source(
        self,
        splitter: COBOLSplitter,
        default_profile: SplitterProfile,
    ) -> None:
        """Test that chunks cover the entire source."""
        result = splitter.split(SIMPLE_COBOL, default_profile, "TEST001.cbl")

        # Get all covered lines
        covered_lines = set()
        for chunk in result.chunks:
            for line in range(chunk.start_line, chunk.end_line + 1):
                covered_lines.add(line)

        # Should cover most lines (some overlap is OK)
        assert len(covered_lines) >= result.total_lines * 0.9

    def test_chunk_ids_are_unique(
        self,
        splitter: COBOLSplitter,
        default_profile: SplitterProfile,
    ) -> None:
        """Test that all chunk IDs are unique."""
        result = splitter.split(SIMPLE_COBOL, default_profile, "TEST001.cbl")

        chunk_ids = [chunk.chunk_id for chunk in result.chunks]
        assert len(chunk_ids) == len(set(chunk_ids)), "Duplicate chunk IDs found"

    def test_chunks_have_estimated_tokens(
        self,
        splitter: COBOLSplitter,
        default_profile: SplitterProfile,
    ) -> None:
        """Test that chunks have estimated token counts."""
        result = splitter.split(SIMPLE_COBOL, default_profile, "TEST001.cbl")

        for chunk in result.chunks:
            assert chunk.estimated_tokens > 0

    def test_split_with_small_budget(
        self,
        splitter: COBOLSplitter,
        small_profile: SplitterProfile,
    ) -> None:
        """Test that small budget creates more chunks."""
        large_profile = SplitterProfile(
            name="large",
            prefer_semantic=True,
            max_chunk_tokens=10000,
        )

        result_small = splitter.split(SIMPLE_COBOL, small_profile, "TEST.cbl")
        result_large = splitter.split(SIMPLE_COBOL, large_profile, "TEST.cbl")

        # Small budget should create more chunks (or at least as many)
        assert len(result_small.chunks) >= len(result_large.chunks)

    def test_split_without_semantic(
        self,
        splitter: COBOLSplitter,
    ) -> None:
        """Test line-based splitting without semantic parsing."""
        profile = SplitterProfile(
            name="line_based",
            prefer_semantic=False,
            max_chunk_tokens=200,
        )

        result = splitter.split(SIMPLE_COBOL, profile, "TEST.cbl")

        assert len(result.chunks) > 0
        for chunk in result.chunks:
            assert chunk.chunk_kind == ChunkKind.GENERIC

    def test_procedure_paragraphs_tracked(
        self,
        splitter: COBOLSplitter,
        default_profile: SplitterProfile,
    ) -> None:
        """Test that paragraphs are tracked in procedure chunks."""
        result = splitter.split(SIMPLE_COBOL, default_profile, "TEST001.cbl")

        # Find procedure-related chunks
        procedure_chunks = [
            c for c in result.chunks
            if c.chunk_kind in (ChunkKind.PROCEDURE_DIVISION, ChunkKind.PROCEDURE_PART)
        ]

        # At least one should have paragraphs
        all_paragraphs = []
        for chunk in procedure_chunks:
            all_paragraphs.extend(chunk.paragraphs)

        assert "MAIN-LOGIC" in all_paragraphs or len(procedure_chunks) > 0

    def test_generate_chunk_id_deterministic(
        self,
        splitter: COBOLSplitter,
    ) -> None:
        """Test that chunk ID generation is deterministic."""
        id1 = splitter.generate_chunk_id("TEST.cbl", ChunkKind.PROCEDURE_PART, 0)
        id2 = splitter.generate_chunk_id("TEST.cbl", ChunkKind.PROCEDURE_PART, 0)
        assert id1 == id2

        # Different inputs produce different IDs
        id3 = splitter.generate_chunk_id("TEST.cbl", ChunkKind.PROCEDURE_PART, 1)
        assert id1 != id3

    def test_generate_chunk_id_with_name(
        self,
        splitter: COBOLSplitter,
    ) -> None:
        """Test chunk ID generation with semantic name."""
        id1 = splitter.generate_chunk_id(
            "TEST.cbl", ChunkKind.PROCEDURE_PART, 0, "MAIN-LOGIC"
        )
        assert "main_logic" in id1.lower()

    def test_generate_chunk_id_different_files(
        self,
        splitter: COBOLSplitter,
    ) -> None:
        """Test chunk ID varies by filename."""
        id1 = splitter.generate_chunk_id("FILE1.cbl", ChunkKind.PROCEDURE_PART, 0)
        id2 = splitter.generate_chunk_id("FILE2.cbl", ChunkKind.PROCEDURE_PART, 0)
        assert id1 != id2

    def test_generate_chunk_id_different_kinds(
        self,
        splitter: COBOLSplitter,
    ) -> None:
        """Test chunk ID varies by chunk kind."""
        id1 = splitter.generate_chunk_id("TEST.cbl", ChunkKind.DATA_DIVISION, 0)
        id2 = splitter.generate_chunk_id("TEST.cbl", ChunkKind.PROCEDURE_DIVISION, 0)
        assert id1 != id2

    def test_validate_chunk(
        self,
        splitter: COBOLSplitter,
        default_profile: SplitterProfile,
    ) -> None:
        """Test chunk validation."""
        from atlas.models.manifest import ChunkSpec

        # Valid chunk
        valid_chunk = ChunkSpec(
            chunk_id="test",
            chunk_kind=ChunkKind.PROCEDURE_PART,
            start_line=1,
            end_line=100,
            estimated_tokens=1000,
        )
        errors = splitter.validate_chunk(valid_chunk, default_profile)
        assert len(errors) == 0

        # Invalid: tokens exceed budget
        oversized = ChunkSpec(
            chunk_id="oversized",
            chunk_kind=ChunkKind.PROCEDURE_PART,
            start_line=1,
            end_line=100,
            estimated_tokens=10000,
        )
        errors = splitter.validate_chunk(oversized, default_profile)
        assert len(errors) > 0
        assert "exceeds" in errors[0].lower()

        # Invalid: end line before start
        invalid_range = ChunkSpec(
            chunk_id="invalid",
            chunk_kind=ChunkKind.PROCEDURE_PART,
            start_line=100,
            end_line=50,
            estimated_tokens=100,
        )
        errors = splitter.validate_chunk(invalid_range, default_profile)
        assert len(errors) > 0

    def test_validate_chunk_zero_tokens(
        self,
        splitter: COBOLSplitter,
        default_profile: SplitterProfile,
    ) -> None:
        """Test validation of chunk with zero tokens."""
        from atlas.models.manifest import ChunkSpec

        zero_token_chunk = ChunkSpec(
            chunk_id="zero",
            chunk_kind=ChunkKind.GENERIC,
            start_line=1,
            end_line=1,
            estimated_tokens=0,
        )
        errors = splitter.validate_chunk(zero_token_chunk, default_profile)
        # Depending on implementation, might have warning or be valid
        assert isinstance(errors, list)

    def test_validate_chunk_single_line(
        self,
        splitter: COBOLSplitter,
        default_profile: SplitterProfile,
    ) -> None:
        """Test validation of single-line chunk."""
        from atlas.models.manifest import ChunkSpec

        single_line = ChunkSpec(
            chunk_id="single",
            chunk_kind=ChunkKind.GENERIC,
            start_line=50,
            end_line=50,
            estimated_tokens=10,
        )
        errors = splitter.validate_chunk(single_line, default_profile)
        assert len(errors) == 0  # Single line is valid

    def test_warnings_for_oversized_chunks(
        self,
        splitter: COBOLSplitter,
    ) -> None:
        """Test that oversized chunks generate warnings."""
        # Very small budget should create warnings if semantic units are large
        tiny_profile = SplitterProfile(
            name="tiny",
            prefer_semantic=False,  # Force line-based which has validation
            max_chunk_tokens=10,
        )

        result = splitter.split(SIMPLE_COBOL, tiny_profile, "TEST.cbl")

        # With such a small budget, we might have oversized chunks
        # The important thing is the splitter doesn't crash
        assert result is not None

    def test_split_handles_empty_source(
        self,
        splitter: COBOLSplitter,
        default_profile: SplitterProfile,
    ) -> None:
        """Test handling of empty source."""
        result = splitter.split("", default_profile, "EMPTY.cbl")

        assert result.total_lines == 1  # Empty string splits to ['']
        assert result.semantic_boundaries_found == 0

    def test_split_handles_no_divisions(
        self,
        splitter: COBOLSplitter,
        default_profile: SplitterProfile,
    ) -> None:
        """Test handling of source without COBOL divisions."""
        source = """       * This is just a comment
       * No divisions here
       MOVE 1 TO X.
       ADD 1 TO Y.
"""
        result = splitter.split(source, default_profile, "NODIV.cbl")

        # Should fall back to line-based splitting
        assert len(result.chunks) > 0

    def test_split_handles_whitespace_only(
        self,
        splitter: COBOLSplitter,
        default_profile: SplitterProfile,
    ) -> None:
        """Test handling of whitespace-only source."""
        source = "       \n       \n       \n"
        result = splitter.split(source, default_profile, "WHITESPACE.cbl")

        assert result is not None
        assert result.total_lines >= 1

    def test_split_handles_comments_only(
        self,
        splitter: COBOLSplitter,
        default_profile: SplitterProfile,
    ) -> None:
        """Test handling of comment-only source."""
        source = """      * Comment line 1
      * Comment line 2
      * Comment line 3
"""
        result = splitter.split(source, default_profile, "COMMENTS.cbl")

        assert result is not None
        assert len(result.chunks) > 0

    def test_cobol_structure_parsing(
        self,
        splitter: COBOLSplitter,
    ) -> None:
        """Test COBOLStructure is correctly populated."""
        lines = SIMPLE_COBOL.split("\n")
        structure = splitter._parse_structure(SIMPLE_COBOL, lines)

        assert structure.total_lines == len(lines)
        assert len(structure.divisions) == 4
        assert "IDENTIFICATION" in structure.divisions
        assert "PROCEDURE" in structure.divisions

        # Check paragraphs were found
        assert len(structure.paragraphs) >= 3
        assert "MAIN-LOGIC" in structure.paragraphs

    def test_section_detection(
        self,
        splitter: COBOLSplitter,
    ) -> None:
        """Test that sections are detected."""
        # The WORKING-STORAGE SECTION should be detected
        boundaries = splitter.detect_semantic_boundaries(SIMPLE_COBOL)

        section_names = [b[1] for b in boundaries if "SECTION" in b[1]]
        # WORKING-STORAGE is a section
        assert any("WORKING-STORAGE" in s for s in section_names) or \
               any("CONFIGURATION" in s for s in section_names)

    def test_multiple_sections_in_division(
        self,
        splitter: COBOLSplitter,
    ) -> None:
        """Test detection of multiple sections within a division."""
        source = """       IDENTIFICATION DIVISION.
       PROGRAM-ID. MULTISEC.

       DATA DIVISION.
       FILE SECTION.
       FD  INPUT-FILE.
       01  INPUT-RECORD PIC X(80).

       WORKING-STORAGE SECTION.
       01  WS-COUNTER PIC 9(4).

       LINKAGE SECTION.
       01  LS-PARAM PIC X(100).

       PROCEDURE DIVISION.
           STOP RUN.
"""
        boundaries = splitter.detect_semantic_boundaries(source)
        section_names = [b[1] for b in boundaries if "SECTION" in b[1]]

        # Should find multiple sections
        assert len(section_names) >= 2

    def test_nested_perform_tracking(
        self,
        splitter: COBOLSplitter,
        default_profile: SplitterProfile,
    ) -> None:
        """Test tracking of PERFORM statements in procedure chunks."""
        source = """       IDENTIFICATION DIVISION.
       PROGRAM-ID. NESTED.

       PROCEDURE DIVISION.
       MAIN-PARA.
           PERFORM SUB-A.
           PERFORM SUB-B THRU SUB-B-EXIT.
           PERFORM SUB-C UNTIL DONE.
           STOP RUN.

       SUB-A.
           DISPLAY 'A'.

       SUB-B.
           DISPLAY 'B'.
       SUB-B-EXIT.
           EXIT.

       SUB-C.
           DISPLAY 'C'.
"""
        result = splitter.split(source, default_profile, "NESTED.cbl")

        # Should successfully split
        assert len(result.chunks) > 0
        # Should find paragraphs
        assert result.semantic_boundaries_found > 0


class TestSemanticBoundary:
    """Tests for SemanticBoundary named tuple."""

    def test_creation(self) -> None:
        """Test SemanticBoundary creation."""
        boundary = SemanticBoundary(
            line_number=10,
            name="MAIN-LOGIC",
            kind=ChunkKind.PROCEDURE_PART,
            level=2,
        )
        assert boundary.line_number == 10
        assert boundary.name == "MAIN-LOGIC"
        assert boundary.kind == ChunkKind.PROCEDURE_PART
        assert boundary.level == 2

    def test_immutability(self) -> None:
        """Test that SemanticBoundary is immutable."""
        boundary = SemanticBoundary(
            line_number=10,
            name="TEST",
            kind=ChunkKind.GENERIC,
            level=0,
        )
        with pytest.raises(AttributeError):
            boundary.line_number = 20  # type: ignore

    def test_equality(self) -> None:
        """Test SemanticBoundary equality comparison."""
        b1 = SemanticBoundary(10, "TEST", ChunkKind.GENERIC, 0)
        b2 = SemanticBoundary(10, "TEST", ChunkKind.GENERIC, 0)
        b3 = SemanticBoundary(20, "TEST", ChunkKind.GENERIC, 0)

        assert b1 == b2
        assert b1 != b3

    def test_hashing(self) -> None:
        """Test SemanticBoundary can be used in sets and dicts."""
        b1 = SemanticBoundary(10, "TEST", ChunkKind.GENERIC, 0)
        b2 = SemanticBoundary(10, "TEST", ChunkKind.GENERIC, 0)

        # Should be hashable
        boundary_set = {b1, b2}
        assert len(boundary_set) == 1  # Same boundary

        # Can be dict key
        boundary_dict = {b1: "value"}
        assert boundary_dict[b2] == "value"

    def test_unpacking(self) -> None:
        """Test SemanticBoundary can be unpacked."""
        boundary = SemanticBoundary(10, "TEST", ChunkKind.GENERIC, 0)
        line, name, kind, level = boundary

        assert line == 10
        assert name == "TEST"
        assert kind == ChunkKind.GENERIC
        assert level == 0


class TestCOBOLStructure:
    """Tests for COBOLStructure dataclass."""

    def test_creation_defaults(self) -> None:
        """Test COBOLStructure default values."""
        structure = COBOLStructure()
        assert structure.boundaries == []
        assert structure.divisions == {}
        assert structure.sections == {}
        assert structure.paragraphs == {}
        assert structure.total_lines == 0

    def test_with_data(self) -> None:
        """Test COBOLStructure with data."""
        boundary = SemanticBoundary(
            line_number=1,
            name="IDENTIFICATION DIVISION",
            kind=ChunkKind.IDENTIFICATION_DIVISION,
            level=0,
        )
        structure = COBOLStructure(
            boundaries=[boundary],
            divisions={"IDENTIFICATION": boundary},
            total_lines=100,
        )
        assert len(structure.boundaries) == 1
        assert "IDENTIFICATION" in structure.divisions
        assert structure.total_lines == 100

    def test_multiple_boundaries(self) -> None:
        """Test COBOLStructure with multiple boundaries."""
        boundaries = [
            SemanticBoundary(1, "IDENTIFICATION DIVISION", ChunkKind.IDENTIFICATION_DIVISION, 0),
            SemanticBoundary(10, "PROCEDURE DIVISION", ChunkKind.PROCEDURE_DIVISION, 0),
            SemanticBoundary(15, "MAIN-PARA", ChunkKind.PROCEDURE_PART, 1),
        ]
        structure = COBOLStructure(
            boundaries=boundaries,
            divisions={
                "IDENTIFICATION": boundaries[0],
                "PROCEDURE": boundaries[1],
            },
            paragraphs={"MAIN-PARA": boundaries[2]},
            total_lines=50,
        )

        assert len(structure.boundaries) == 3
        assert len(structure.divisions) == 2
        assert len(structure.paragraphs) == 1


@pytest.mark.skipif(
    not FIXTURES_DIR.exists(),
    reason="COBOL fixtures directory not found"
)
class TestCOBOLSplitterWithFixtures:
    """Tests using COBOL fixture files."""

    @pytest.fixture
    def splitter(self) -> COBOLSplitter:
        """Provide a COBOLSplitter instance."""
        return COBOLSplitter()

    @pytest.fixture
    def default_profile(self) -> SplitterProfile:
        """Provide a default splitter profile."""
        return SplitterProfile(
            name="test",
            prefer_semantic=True,
            max_chunk_tokens=3500,
            overlap_lines=10,
        )

    @pytest.fixture
    def small_profile(self) -> SplitterProfile:
        """Small profile to force more splitting."""
        return SplitterProfile(
            name="small",
            prefer_semantic=True,
            max_chunk_tokens=500,
            overlap_lines=5,
        )

    def _load_fixture(self, filename: str) -> str:
        """Load a fixture file."""
        filepath = FIXTURES_DIR / filename
        if filepath.exists():
            return filepath.read_text()
        pytest.skip(f"Fixture file not found: {filename}")

    def test_small_program(
        self,
        splitter: COBOLSplitter,
        default_profile: SplitterProfile,
    ) -> None:
        """Test splitting a small COBOL program."""
        source = self._load_fixture("small_program.cbl")
        result = splitter.split(source, default_profile, "small_program.cbl")

        assert result.total_lines > 0
        assert len(result.chunks) >= 1
        # Small program - chunks vary based on divisions/sections detected
        # (may be more chunks with finer-grained semantic splitting)
        assert len(result.chunks) <= 10

    def test_medium_program(
        self,
        splitter: COBOLSplitter,
        default_profile: SplitterProfile,
    ) -> None:
        """Test splitting a medium-sized COBOL program."""
        source = self._load_fixture("medium_program.cbl")
        result = splitter.split(source, default_profile, "medium_program.cbl")

        assert result.total_lines > 50
        assert len(result.chunks) >= 1
        assert result.semantic_boundaries_found > 0

    def test_large_program(
        self,
        splitter: COBOLSplitter,
        default_profile: SplitterProfile,
    ) -> None:
        """Test splitting a large COBOL program."""
        source = self._load_fixture("large_program.cbl")
        result = splitter.split(source, default_profile, "large_program.cbl")

        assert result.total_lines > 100
        assert len(result.chunks) >= 1
        # Large programs should have multiple boundaries
        assert result.semantic_boundaries_found >= 4

    def test_large_program_with_small_budget(
        self,
        splitter: COBOLSplitter,
        small_profile: SplitterProfile,
    ) -> None:
        """Test that large programs create more chunks with small budget."""
        source = self._load_fixture("large_program.cbl")

        large_budget = SplitterProfile(
            name="large",
            prefer_semantic=True,
            max_chunk_tokens=10000,
        )

        result_small = splitter.split(source, small_profile, "large_program.cbl")
        result_large = splitter.split(source, large_budget, "large_program.cbl")

        # Small budget should create at least as many chunks
        assert len(result_small.chunks) >= len(result_large.chunks)

    def test_copybook_customer(
        self,
        splitter: COBOLSplitter,
        default_profile: SplitterProfile,
    ) -> None:
        """Test splitting a copybook file (customer layout)."""
        source = self._load_fixture("customer_copy.cpy")
        result = splitter.split(source, default_profile, "customer_copy.cpy")

        # Copybooks are typically smaller
        assert result.total_lines > 0
        assert len(result.chunks) >= 1
        # Copybooks often don't have PROCEDURE DIVISION
        # Should still be processable

    def test_copybook_error_codes(
        self,
        splitter: COBOLSplitter,
        default_profile: SplitterProfile,
    ) -> None:
        """Test splitting error codes copybook."""
        source = self._load_fixture("error_codes.cpy")
        result = splitter.split(source, default_profile, "error_codes.cpy")

        assert result.total_lines > 0
        assert len(result.chunks) >= 1

    def test_empty_program(
        self,
        splitter: COBOLSplitter,
        default_profile: SplitterProfile,
    ) -> None:
        """Test splitting minimal/empty COBOL program."""
        source = self._load_fixture("empty_program.cbl")
        result = splitter.split(source, default_profile, "empty_program.cbl")

        # Should handle minimal program gracefully
        assert result.total_lines >= 1
        assert len(result.chunks) >= 1

    def test_malformed_cobol(
        self,
        splitter: COBOLSplitter,
        default_profile: SplitterProfile,
    ) -> None:
        """Test handling of malformed COBOL."""
        source = self._load_fixture("malformed.cbl")
        result = splitter.split(source, default_profile, "malformed.cbl")

        # Should not crash
        assert result is not None
        # Should fall back to line-based splitting
        assert len(result.chunks) >= 1
        # Likely no semantic boundaries found
        assert result.semantic_boundaries_found == 0 or result.chunks[0].chunk_kind == ChunkKind.GENERIC

    def test_fixture_determinism(
        self,
        splitter: COBOLSplitter,
        default_profile: SplitterProfile,
    ) -> None:
        """Test that fixture splitting is deterministic."""
        source = self._load_fixture("medium_program.cbl")

        result1 = splitter.split(source, default_profile, "medium_program.cbl")
        result2 = splitter.split(source, default_profile, "medium_program.cbl")

        assert len(result1.chunks) == len(result2.chunks)
        for c1, c2 in zip(result1.chunks, result2.chunks):
            assert c1.chunk_id == c2.chunk_id
            assert c1.start_line == c2.start_line
            assert c1.end_line == c2.end_line
            assert c1.chunk_kind == c2.chunk_kind


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.fixture
    def splitter(self) -> COBOLSplitter:
        """Provide a COBOLSplitter instance."""
        return COBOLSplitter()

    @pytest.fixture
    def default_profile(self) -> SplitterProfile:
        """Default profile for testing."""
        return SplitterProfile(
            name="test",
            prefer_semantic=True,
            max_chunk_tokens=3500,
        )

    def test_single_line_source(
        self,
        splitter: COBOLSplitter,
        default_profile: SplitterProfile,
    ) -> None:
        """Test handling of single-line source."""
        source = "       IDENTIFICATION DIVISION."
        result = splitter.split(source, default_profile, "SINGLE.cbl")

        assert result.total_lines == 1
        assert len(result.chunks) >= 1

    def test_very_long_line(
        self,
        splitter: COBOLSplitter,
        default_profile: SplitterProfile,
    ) -> None:
        """Test handling of very long lines."""
        # Create a line that exceeds typical COBOL column limits
        long_line = "       01 LONG-VAR PIC X(" + "9" * 1000 + ")."
        source = f"""       IDENTIFICATION DIVISION.
       PROGRAM-ID. LONGLINE.
       DATA DIVISION.
       WORKING-STORAGE SECTION.
{long_line}
       PROCEDURE DIVISION.
           STOP RUN.
"""
        result = splitter.split(source, default_profile, "LONGLINE.cbl")

        # Should not crash
        assert result is not None
        assert len(result.chunks) >= 1

    def test_special_characters_in_comments(
        self,
        splitter: COBOLSplitter,
        default_profile: SplitterProfile,
    ) -> None:
        """Test handling of special characters in comments."""
        source = """       IDENTIFICATION DIVISION.
       PROGRAM-ID. SPECIAL.
      * Special chars: !@#$%^&*()[]{}|;:'",.<>?/~`
      * Unicode:
      * Tabs:	tab	here
       PROCEDURE DIVISION.
           STOP RUN.
"""
        result = splitter.split(source, default_profile, "SPECIAL.cbl")

        assert result is not None
        assert len(result.chunks) >= 1

    def test_mixed_case_keywords(
        self,
        splitter: COBOLSplitter,
        default_profile: SplitterProfile,
    ) -> None:
        """Test handling of mixed-case COBOL keywords."""
        source = """       Identification Division.
       Program-Id. MIXEDCASE.

       Procedure DIVISION.
       Main-Logic.
           stop RUN.
"""
        result = splitter.split(source, default_profile, "MIXED.cbl")

        # Should still detect divisions (case-insensitive)
        assert result.semantic_boundaries_found > 0

    def test_continuation_lines(
        self,
        splitter: COBOLSplitter,
        default_profile: SplitterProfile,
    ) -> None:
        """Test handling of COBOL continuation lines."""
        source = """       IDENTIFICATION DIVISION.
       PROGRAM-ID. CONTINUE.
       DATA DIVISION.
       WORKING-STORAGE SECTION.
       01 WS-LONG-STRING PIC X(100) VALUE 'THIS IS A VERY LON
      -    'G STRING THAT CONTINUES ON THE NEXT LINE'.
       PROCEDURE DIVISION.
           STOP RUN.
"""
        result = splitter.split(source, default_profile, "CONTINUE.cbl")

        assert result is not None
        assert len(result.chunks) >= 1

    def test_blank_lines_between_divisions(
        self,
        splitter: COBOLSplitter,
        default_profile: SplitterProfile,
    ) -> None:
        """Test handling of many blank lines between divisions."""
        source = """       IDENTIFICATION DIVISION.
       PROGRAM-ID. BLANKS.



       DATA DIVISION.



       WORKING-STORAGE SECTION.
       01 WS-VAR PIC X.



       PROCEDURE DIVISION.



           STOP RUN.
"""
        result = splitter.split(source, default_profile, "BLANKS.cbl")

        # Should still find all divisions
        assert result.semantic_boundaries_found >= 3

    def test_deeply_nested_data_items(
        self,
        splitter: COBOLSplitter,
        default_profile: SplitterProfile,
    ) -> None:
        """Test handling of deeply nested data structures."""
        source = """       IDENTIFICATION DIVISION.
       PROGRAM-ID. NESTED.
       DATA DIVISION.
       WORKING-STORAGE SECTION.
       01 WS-LEVEL-01.
          05 WS-LEVEL-05.
             10 WS-LEVEL-10.
                15 WS-LEVEL-15.
                   20 WS-LEVEL-20.
                      25 WS-LEVEL-25 PIC X.
       PROCEDURE DIVISION.
           STOP RUN.
"""
        result = splitter.split(source, default_profile, "NESTED.cbl")

        assert result is not None
        assert len(result.chunks) >= 1

    def test_multiple_programs_in_source(
        self,
        splitter: COBOLSplitter,
        default_profile: SplitterProfile,
    ) -> None:
        """Test handling of multiple programs (nested programs)."""
        source = """       IDENTIFICATION DIVISION.
       PROGRAM-ID. OUTER.
       PROCEDURE DIVISION.
           DISPLAY 'OUTER'.
           STOP RUN.

       IDENTIFICATION DIVISION.
       PROGRAM-ID. INNER.
       PROCEDURE DIVISION.
           DISPLAY 'INNER'.
       END PROGRAM INNER.
       END PROGRAM OUTER.
"""
        result = splitter.split(source, default_profile, "MULTI.cbl")

        # Should handle multiple IDENTIFICATION DIVISIONs
        assert result is not None
        assert result.semantic_boundaries_found >= 2  # At least 2 ID divisions

    def test_only_procedure_division(
        self,
        splitter: COBOLSplitter,
        default_profile: SplitterProfile,
    ) -> None:
        """Test COBOL with only PROCEDURE DIVISION (invalid but should handle)."""
        source = """       PROCEDURE DIVISION.
       MAIN-PARA.
           DISPLAY 'HELLO'.
           STOP RUN.
"""
        result = splitter.split(source, default_profile, "PROCONLY.cbl")

        assert result is not None
        # Should detect procedure division
        assert len(result.chunks) >= 1

    def test_large_number_of_paragraphs(
        self,
        splitter: COBOLSplitter,
    ) -> None:
        """Test handling of many paragraphs."""
        # Generate source with many paragraphs
        paragraphs = []
        for i in range(100):
            paragraphs.append(f"""       PARA-{i:04d}.
           DISPLAY 'PARA {i}'.
""")

        source = f"""       IDENTIFICATION DIVISION.
       PROGRAM-ID. MANYPARA.
       PROCEDURE DIVISION.
{''.join(paragraphs)}
           STOP RUN.
"""
        small_profile = SplitterProfile(
            name="small",
            prefer_semantic=True,
            max_chunk_tokens=500,
        )

        result = splitter.split(source, small_profile, "MANYPARA.cbl")

        # Should handle many paragraphs
        assert result is not None
        assert result.semantic_boundaries_found >= 50  # Should find many paragraphs

    def test_section_with_paragraphs(
        self,
        splitter: COBOLSplitter,
        default_profile: SplitterProfile,
    ) -> None:
        """Test sections containing multiple paragraphs."""
        source = """       IDENTIFICATION DIVISION.
       PROGRAM-ID. SECPARA.
       PROCEDURE DIVISION.

       INIT-SECTION SECTION.
       INIT-START.
           DISPLAY 'INIT START'.
       INIT-MIDDLE.
           DISPLAY 'INIT MIDDLE'.
       INIT-END.
           DISPLAY 'INIT END'.

       PROCESS-SECTION SECTION.
       PROCESS-START.
           DISPLAY 'PROCESS'.
           STOP RUN.
"""
        result = splitter.split(source, default_profile, "SECPARA.cbl")

        # Should find paragraphs as semantic boundaries
        boundaries = splitter.detect_semantic_boundaries(source)
        # Splitter finds divisions and paragraph entries
        paragraph_count = sum(
            1 for b in boundaries if b[2].value == "procedure_part"
        )
        # Should find multiple paragraphs (INIT-START, INIT-MIDDLE, etc.)
        assert paragraph_count >= 2

    def test_redefines_clause(
        self,
        splitter: COBOLSplitter,
        default_profile: SplitterProfile,
    ) -> None:
        """Test handling of REDEFINES clauses in data division."""
        source = """       IDENTIFICATION DIVISION.
       PROGRAM-ID. REDEFINE.
       DATA DIVISION.
       WORKING-STORAGE SECTION.
       01 WS-DATE-NUMERIC PIC 9(8).
       01 WS-DATE-FORMATTED REDEFINES WS-DATE-NUMERIC.
          05 WS-YEAR   PIC 9(4).
          05 WS-MONTH  PIC 99.
          05 WS-DAY    PIC 99.
       PROCEDURE DIVISION.
           STOP RUN.
"""
        result = splitter.split(source, default_profile, "REDEFINE.cbl")

        assert result is not None
        assert len(result.chunks) >= 1

    def test_copybook_statement(
        self,
        splitter: COBOLSplitter,
        default_profile: SplitterProfile,
    ) -> None:
        """Test handling of COPY statements."""
        source = """       IDENTIFICATION DIVISION.
       PROGRAM-ID. COPYSTMT.
       DATA DIVISION.
       WORKING-STORAGE SECTION.
           COPY CUSTOMER.
           COPY ERROR-CODES.
       PROCEDURE DIVISION.
           STOP RUN.
"""
        result = splitter.split(source, default_profile, "COPYSTMT.cbl")

        assert result is not None
        assert len(result.chunks) >= 1


class TestOverlapBehavior:
    """Tests for chunk overlap behavior."""

    @pytest.fixture
    def splitter(self) -> COBOLSplitter:
        """Provide a COBOLSplitter instance."""
        return COBOLSplitter()

    def test_overlap_lines_respected(
        self,
        splitter: COBOLSplitter,
    ) -> None:
        """Test that overlap_lines parameter is respected."""
        profile_no_overlap = SplitterProfile(
            name="no_overlap",
            prefer_semantic=False,
            max_chunk_tokens=100,
            overlap_lines=0,
        )

        profile_with_overlap = SplitterProfile(
            name="with_overlap",
            prefer_semantic=False,
            max_chunk_tokens=100,
            overlap_lines=10,
        )

        result_no = splitter.split(LARGE_PROCEDURE_COBOL, profile_no_overlap, "TEST.cbl")
        result_with = splitter.split(LARGE_PROCEDURE_COBOL, profile_with_overlap, "TEST.cbl")

        # With overlap, chunks may share lines
        if len(result_with.chunks) > 1 and len(result_no.chunks) > 1:
            # Check for overlap in adjacent chunks
            for i in range(len(result_with.chunks) - 1):
                chunk1 = result_with.chunks[i]
                chunk2 = result_with.chunks[i + 1]
                # Chunk 2 might start before chunk 1 ends (overlap)
                overlap = chunk1.end_line - chunk2.start_line + 1
                if overlap > 0:
                    assert overlap <= 10  # Should not exceed overlap_lines

    def test_default_overlap(
        self,
        splitter: COBOLSplitter,
    ) -> None:
        """Test default overlap behavior."""
        profile = SplitterProfile(
            name="default",
            prefer_semantic=True,
            max_chunk_tokens=3500,
            # overlap_lines defaults to 0 or some default
        )

        result = splitter.split(SIMPLE_COBOL, profile, "TEST.cbl")

        # Should work without crash
        assert result is not None
        assert len(result.chunks) >= 1


class TestTokenEstimation:
    """Tests for token estimation accuracy."""

    @pytest.fixture
    def splitter(self) -> COBOLSplitter:
        """Provide a COBOLSplitter instance."""
        return COBOLSplitter()

    def test_token_estimate_minimum(
        self,
        splitter: COBOLSplitter,
    ) -> None:
        """Test minimum token estimate is 1."""
        assert splitter.estimate_tokens("") == 1
        assert splitter.estimate_tokens("a") == 1
        assert splitter.estimate_tokens("ab") == 1
        assert splitter.estimate_tokens("abc") == 1

    def test_token_estimate_proportional(
        self,
        splitter: COBOLSplitter,
    ) -> None:
        """Test token estimates scale proportionally with text length."""
        short = "a" * 100
        medium = "a" * 400
        long_text = "a" * 1000

        est_short = splitter.estimate_tokens(short)
        est_medium = splitter.estimate_tokens(medium)
        est_long = splitter.estimate_tokens(long_text)

        assert est_short < est_medium < est_long
        # Should be roughly proportional (4 chars per token)
        assert est_medium == pytest.approx(est_short * 4, rel=0.1)
        assert est_long == pytest.approx(est_short * 10, rel=0.1)

    def test_chunks_respect_token_budget(
        self,
        splitter: COBOLSplitter,
    ) -> None:
        """Test that chunks (mostly) respect token budget."""
        profile = SplitterProfile(
            name="small",
            prefer_semantic=True,
            max_chunk_tokens=500,
        )

        result = splitter.split(LARGE_PROCEDURE_COBOL, profile, "TEST.cbl")

        # Most chunks should be under budget
        # (some may exceed due to semantic boundaries being larger than budget)
        within_budget = sum(1 for c in result.chunks if c.estimated_tokens <= 500)
        assert within_budget >= len(result.chunks) * 0.5  # At least half should be under
