from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from contracts.schemas import ChunkingStrategy
# shared.schemas.enums import ChunkingStrategy

class ChunkerPredictRequest(BaseModel):
    """Request to the AI Cluster Chunker.

    Supports three input modes:
    - text: inline text content
    - file_url: direct URL to a file
    - read_url + write_url: pre-signed MinIO URLs for stateless processing
    """
    source: Optional[str] = None
    text: Optional[str] = None
    file_url: Optional[str] = None
    read_url: Optional[str] = None
    write_url: Optional[str] = None
    strategy: ChunkingStrategy = ChunkingStrategy.FIXED
    chunk_size: int = Field(ge=1)
    chunk_overlap: int = Field(ge=0)


class ChunkResult(BaseModel):
    chunk_index: int
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkerPredictResponse(BaseModel):
    chunks: list[ChunkResult]