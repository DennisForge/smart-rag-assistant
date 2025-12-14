"""
Type aliases for RAG module.

These NewType definitions provide semantic clarity while maintaining full type compatibility.
"""

from typing import NewType

DocumentId = NewType("DocumentId", str)
ChunkId = NewType("ChunkId", str)
CollectionName = NewType("CollectionName", str)
