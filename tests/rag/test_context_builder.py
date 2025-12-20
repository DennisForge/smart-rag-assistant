"""Tests for ContextBuilder service."""

from __future__ import annotations

from app.rag.models.documents import DocumentChunk, ScoredDocumentChunk
from app.rag.services.context_builder import ContextBuilder


def test_build_empty_input():
    """Test that building with no chunks returns empty string."""
    builder = ContextBuilder()
    result = builder.build([])
    assert result == ""


def test_build_ordering_and_separator():
    """Test that chunks are joined in order with correct separator."""
    builder = ContextBuilder()

    chunks = [
        ScoredDocumentChunk(
            chunk=DocumentChunk(
                id="doc1::chunk:0",
                document_id="doc1",
                content="First chunk content",
                index=0,
                metadata={},
            ),
            score=0.1,
        ),
        ScoredDocumentChunk(
            chunk=DocumentChunk(
                id="doc2::chunk:0",
                document_id="doc2",
                content="Second chunk content",
                index=0,
                metadata={},
            ),
            score=0.2,
        ),
    ]

    result = builder.build(chunks)

    # Assert ordering
    assert result.startswith("First chunk content")
    assert result.endswith("Second chunk content")

    # Assert separator
    assert "\n\n---\n\n" in result
    assert result == "First chunk content\n\n---\n\nSecond chunk content"


def test_build_max_chars_enforced():
    """Test that max_chars cap is enforced and stops at chunk boundaries."""
    builder = ContextBuilder()

    # Create chunks with known sizes
    # Each content is 100 chars, separator is 7 chars
    chunks = [
        ScoredDocumentChunk(
            chunk=DocumentChunk(
                id=f"doc{i}::chunk:0",
                document_id=f"doc{i}",
                content="x" * 100,  # 100 chars each
                index=0,
                metadata={},
            ),
            score=0.1 * i,
        )
        for i in range(1, 101)  # 100 chunks
    ]

    # With max_chars=500, we can fit:
    # chunk1: 100 chars
    # separator: 7 chars (total: 107)
    # chunk2: 100 chars (total: 207)
    # separator: 7 chars (total: 214)
    # chunk3: 100 chars (total: 314)
    # separator: 7 chars (total: 321)
    # chunk4: 100 chars (total: 421)
    # separator: 7 chars (total: 428)
    # chunk5: 100 chars would make total 528 > 500, so stop

    result = builder.build(chunks, max_chars=500)

    # Assert length constraint
    assert len(result) <= 500

    # Assert we got exactly 4 chunks (stops before overflow)
    expected = "\n\n---\n\n".join(["x" * 100] * 4)
    assert result == expected
    assert len(result) == 421  # 4*100 + 3*7


def test_build_single_chunk():
    """Test building with a single chunk (no separator)."""
    builder = ContextBuilder()

    chunk = ScoredDocumentChunk(
        chunk=DocumentChunk(
            id="doc1::chunk:0",
            document_id="doc1",
            content="Only chunk",
            index=0,
            metadata={},
        ),
        score=0.5,
    )

    result = builder.build([chunk])
    assert result == "Only chunk"
    assert "\n\n---\n\n" not in result


def test_build_respects_exact_limit():
    """Test that builder stops exactly at boundary when limit is tight."""
    builder = ContextBuilder()

    # Create two chunks where both fit exactly at the limit
    chunks = [
        ScoredDocumentChunk(
            chunk=DocumentChunk(
                id="doc1::chunk:0",
                document_id="doc1",
                content="A" * 50,
                index=0,
                metadata={},
            ),
            score=0.1,
        ),
        ScoredDocumentChunk(
            chunk=DocumentChunk(
                id="doc2::chunk:0",
                document_id="doc2",
                content="B" * 50,
                index=0,
                metadata={},
            ),
            score=0.2,
        ),
    ]

    # 50 + 7 (separator) + 50 = 107 chars total
    result = builder.build(chunks, max_chars=107)
    assert len(result) == 107
    assert result.count("A") == 50
    assert result.count("B") == 50
