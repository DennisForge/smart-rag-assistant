from __future__ import annotations

from typing import List
from pydantic import BaseModel


class EmbeddingVector(BaseModel):
    """
    Embedding vector representation.
    
    The dimension (dim) is derived from len(vector), so it's not stored separately.
    """

    vector: List[float]