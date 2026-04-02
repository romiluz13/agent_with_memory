"""
Minimal LangChain-compatible Voyage embeddings adapter.
"""

import asyncio
import os

import voyageai
from langchain_core.embeddings import Embeddings


class VoyageEmbeddingsAdapter(Embeddings):
    """Bridge Voyage AI's SDK to the LangChain Embeddings interface."""

    def __init__(
        self,
        model: str = "voyage-3",
        output_dimension: int = 1024,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.output_dimension = output_dimension
        self.api_key = api_key or os.getenv("VOYAGE_API_KEY")
        if not self.api_key:
            raise ValueError("VOYAGE_API_KEY is required for Voyage embeddings")
        self.client = voyageai.Client(api_key=self.api_key)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        result = self.client.embed(
            texts=texts,
            model=self.model,
            input_type="document",
            output_dimension=self.output_dimension,
        )
        return list(result.embeddings)

    def embed_query(self, text: str) -> list[float]:
        result = self.client.embed(
            texts=[text],
            model=self.model,
            input_type="query",
            output_dimension=self.output_dimension,
        )
        return list(result.embeddings[0])

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_documents, texts)

    async def aembed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embed_query, text)
