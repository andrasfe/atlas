"""Mock LLM response fixtures for testing Atlas workers.

This module provides pre-canned LLM responses for testing without
making actual API calls. Fixtures cover:
- Chunk analysis (Scribe worker)
- Merge operations (Aggregator worker)
- Challenge reviews (Challenger worker)
- Follow-up investigations (Scribe/Followup worker)

Usage:
    >>> from tests.fixtures.llm_responses import load_fixture, get_mock_response
    >>> response = get_mock_response("chunk_analysis", "procedure_division")
    >>> mock_llm = MockLLM(responses=[json.dumps(response)])
"""

import json
from pathlib import Path
from typing import Any

FIXTURES_DIR = Path(__file__).parent


def load_fixture(fixture_name: str) -> dict[str, Any]:
    """Load a fixture file by name.

    Args:
        fixture_name: Name of the fixture file (without .json extension).
            Valid names: chunk_analysis, merge_responses,
            challenge_responses, followup_responses

    Returns:
        The parsed fixture data.

    Raises:
        FileNotFoundError: If fixture file doesn't exist.
        json.JSONDecodeError: If fixture is not valid JSON.

    Example:
        >>> fixture = load_fixture("chunk_analysis")
        >>> print(fixture["responses"].keys())
    """
    fixture_path = FIXTURES_DIR / f"{fixture_name}.json"
    with open(fixture_path) as f:
        return json.load(f)


def get_mock_response(fixture_name: str, response_key: str) -> dict[str, Any]:
    """Get a specific mock response from a fixture.

    Args:
        fixture_name: Name of the fixture file.
        response_key: Key of the response within the fixture's "responses" dict.

    Returns:
        The mock response dictionary.

    Raises:
        KeyError: If response_key doesn't exist in the fixture.

    Example:
        >>> response = get_mock_response("chunk_analysis", "procedure_division")
        >>> print(response["summary"])
    """
    fixture = load_fixture(fixture_name)
    return fixture["responses"][response_key]


def get_all_responses(fixture_name: str) -> dict[str, dict[str, Any]]:
    """Get all responses from a fixture.

    Args:
        fixture_name: Name of the fixture file.

    Returns:
        Dictionary of all responses keyed by response name.

    Example:
        >>> responses = get_all_responses("challenge_responses")
        >>> for name, response in responses.items():
        ...     print(f"{name}: {len(response.get('issues', []))} issues")
    """
    fixture = load_fixture(fixture_name)
    return fixture["responses"]


def list_fixtures() -> list[str]:
    """List all available fixture files.

    Returns:
        List of fixture names (without .json extension).
    """
    return [f.stem for f in FIXTURES_DIR.glob("*.json")]


def list_responses(fixture_name: str) -> list[str]:
    """List all response keys in a fixture.

    Args:
        fixture_name: Name of the fixture file.

    Returns:
        List of response keys.
    """
    fixture = load_fixture(fixture_name)
    return list(fixture["responses"].keys())


# Pre-load common responses for convenience
CHUNK_PROCEDURE_DIVISION = "chunk_analysis:procedure_division"
CHUNK_DATA_DIVISION = "chunk_analysis:data_division"
CHUNK_FILE_CONTROL = "chunk_analysis:file_control"
CHUNK_COPYBOOK = "chunk_analysis:copybook"
CHUNK_ERROR = "chunk_analysis:error_result"

MERGE_SIMPLE = "merge_responses:simple_merge"
MERGE_WITH_CONFLICTS = "merge_responses:merge_with_conflicts"
MERGE_PARTIAL = "merge_responses:partial_merge"

CHALLENGE_NO_ISSUES = "challenge_responses:no_issues"
CHALLENGE_MINOR = "challenge_responses:minor_issues_only"
CHALLENGE_MAJOR = "challenge_responses:major_issues"
CHALLENGE_BLOCKER = "challenge_responses:blocker_issues"
CHALLENGE_MIXED = "challenge_responses:mixed_severity"

FOLLOWUP_SUCCESS = "followup_responses:successful_investigation"
FOLLOWUP_PARTIAL = "followup_responses:partial_answer"
FOLLOWUP_UNABLE = "followup_responses:unable_to_answer"
FOLLOWUP_CROSS_CUTTING = "followup_responses:cross_cutting_investigation"


def get_response_by_id(response_id: str) -> dict[str, Any]:
    """Get a response using its full ID.

    Args:
        response_id: Response ID in format "fixture_name:response_key"

    Returns:
        The mock response dictionary.

    Example:
        >>> response = get_response_by_id(CHUNK_PROCEDURE_DIVISION)
        >>> print(response["summary"])
    """
    fixture_name, response_key = response_id.split(":", 1)
    return get_mock_response(fixture_name, response_key)
