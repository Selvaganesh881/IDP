from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TextChunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: int = Field(
        ...,
        ge=0,
        description="Unique structural ID or page number within the document spectrum",
    )
    text: str = Field(..., min_length=1, description="Extracted Markdown content block")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Contextual tracing information"
    )


class BaseIngestor(ABC):
    @abstractmethod
    async def ingest(self, file_path: str) -> list[TextChunk]:
        """
        Asynchronously read and extract clean structured text elements from a raw document.
        Must be implemented as a coroutine to ensure non-blocking ingestion pipelines.
        """
        return []
