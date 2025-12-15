"""
Voyage AI Embeddings Service
Ported from MongoDB's get-embeddings.js
Maintains exact configuration for compatibility
"""

import os
import logging
import hashlib
import asyncio
from typing import List, Optional, Dict, Any, Literal
from dataclasses import dataclass
import voyageai
from tenacity import retry, stop_after_attempt, wait_exponential
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class EmbeddingConfig(BaseModel):
    """Configuration for Voyage AI embeddings."""
    api_key: str = Field(..., description="Voyage AI API key")
    model: str = Field(default="voyage-3", description="Model to use")
    dimensions: int = Field(default=1024, description="Embedding dimensions")
    batch_size: int = Field(default=10, description="Batch size for embedding generation")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    cache_enabled: bool = Field(default=True, description="Enable embedding cache")


@dataclass
class EmbeddingResult:
    """Result from embedding generation."""
    embedding: List[float]
    text: str
    model: str
    usage: Optional[Dict[str, Any]] = None


class VoyageEmbeddingService:
    """
    Manages embedding generation using Voyage AI.
    Follows patterns from MongoDB's examples.
    """
    
    # Model configurations - keep these exact
    MODELS = {
        "voyage-3": {
            "dimensions": 1024,
            "context_length": 32000,
            "best_for": "general"
        },
        "voyage-3-lite": {
            "dimensions": 512,
            "context_length": 16000,
            "best_for": "lightweight"
        },
        "voyage-code-3": {
            "dimensions": 1024,
            "context_length": 16000,
            "best_for": "code"
        },
    }
    
    def __init__(self, config: EmbeddingConfig):
        """Initialize with configuration."""
        self.config = config
        self.client = voyageai.Client(api_key=config.api_key)
        self._cache: Dict[str, List[float]] = {}
        
        # Validate model configuration
        if config.model not in self.MODELS:
            raise ValueError(f"Invalid model: {config.model}. Choose from: {list(self.MODELS.keys())}")
        
        # Ensure dimensions match model
        expected_dims = self.MODELS[config.model]["dimensions"]
        if config.dimensions != expected_dims:
            logger.warning(f"Adjusting dimensions from {config.dimensions} to {expected_dims} for model {config.model}")
            self.config.dimensions = expected_dims
    
    def _get_cache_key(self, text: str, model: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(f"{text}:{model}".encode()).hexdigest()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def generate_embedding(
        self, 
        text: str,
        input_type: Literal["document", "query"] = "document"
    ) -> EmbeddingResult:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            input_type: Type of input (document or query)
            
        Returns:
            Embedding result
        """
        # Check cache
        if self.config.cache_enabled:
            cache_key = self._get_cache_key(text, self.config.model)
            if cache_key in self._cache:
                logger.debug(f"Using cached embedding for text: {text[:50]}...")
                return EmbeddingResult(
                    embedding=self._cache[cache_key],
                    text=text,
                    model=self.config.model
                )
        
        try:
            # Generate embedding (voyage SDK is sync, wrap in thread)
            result = await asyncio.to_thread(
                self.client.embed,
                texts=[text],
                model=self.config.model,
                input_type=input_type
            )
            
            embedding = result.embeddings[0]
            
            # Cache the result
            if self.config.cache_enabled:
                self._cache[cache_key] = embedding
            
            logger.debug(f"Generated embedding for text: {text[:50]}...")
            
            return EmbeddingResult(
                embedding=embedding,
                text=text,
                model=self.config.model,
                usage=result.usage if hasattr(result, 'usage') else None
            )
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
    
    async def generate_embeddings_batch(
        self,
        texts: List[str],
        input_type: Literal["document", "query"] = "document"
    ) -> List[EmbeddingResult]:
        """
        Generate embeddings for multiple texts in batches.
        
        Args:
            texts: List of texts to embed
            input_type: Type of input (document or query)
            
        Returns:
            List of embedding results
        """
        results = []
        
        # Process in batches
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            
            # Check cache for each text
            cached_results = []
            uncached_texts = []
            uncached_indices = []
            
            if self.config.cache_enabled:
                for j, text in enumerate(batch):
                    cache_key = self._get_cache_key(text, self.config.model)
                    if cache_key in self._cache:
                        cached_results.append(EmbeddingResult(
                            embedding=self._cache[cache_key],
                            text=text,
                            model=self.config.model
                        ))
                    else:
                        uncached_texts.append(text)
                        uncached_indices.append(j)
            else:
                uncached_texts = batch
            
            # Generate embeddings for uncached texts
            if uncached_texts:
                try:
                    batch_result = await asyncio.to_thread(
                        self.client.embed,
                        texts=uncached_texts,
                        model=self.config.model,
                        input_type=input_type
                    )
                    
                    # Create results and update cache
                    for text, embedding in zip(uncached_texts, batch_result.embeddings):
                        result = EmbeddingResult(
                            embedding=embedding,
                            text=text,
                            model=self.config.model,
                            usage=batch_result.usage if hasattr(batch_result, 'usage') else None
                        )
                        
                        if self.config.cache_enabled:
                            cache_key = self._get_cache_key(text, self.config.model)
                            self._cache[cache_key] = embedding
                        
                        results.append(result)
                    
                except Exception as e:
                    logger.error(f"Failed to generate batch embeddings: {e}")
                    raise
            
            # Add cached results
            results.extend(cached_results)
            
            logger.info(f"Processed batch {i//self.config.batch_size + 1}: {len(batch)} texts")
        
        return results
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self._cache.clear()
        logger.info("Embedding cache cleared")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "model": self.config.model,
            "dimensions": self.config.dimensions,
            "context_length": self.MODELS[self.config.model]["context_length"],
            "best_for": self.MODELS[self.config.model]["best_for"],
            "cache_size": len(self._cache)
        }
    
    @staticmethod
    def select_model_for_content(content: str) -> str:
        """
        Select the best model based on content type.
        
        Args:
            content: Content to analyze
            
        Returns:
            Recommended model name
        """
        # Simple heuristic - can be made more sophisticated
        code_indicators = ["def ", "class ", "function ", "import ", "const ", "var ", "let "]
        
        if any(indicator in content for indicator in code_indicators):
            return "voyage-code-3"
        elif len(content) < 1000:
            return "voyage-3-lite"
        else:
            return "voyage-3"


# Singleton instance
_embedding_service: Optional[VoyageEmbeddingService] = None


def get_embedding_service() -> VoyageEmbeddingService:
    """Get the global embedding service instance."""
    global _embedding_service
    
    if _embedding_service is None:
        config = EmbeddingConfig(
            api_key=os.getenv("VOYAGE_API_KEY"),
            model=os.getenv("VOYAGE_EMBEDDING_MODEL", "voyage-3"),
            dimensions=int(os.getenv("VOYAGE_EMBEDDING_DIMENSION", "1024"))
        )
        _embedding_service = VoyageEmbeddingService(config)
    
    return _embedding_service


# Example usage
if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv
    
    load_dotenv()
    
    async def main():
        # Initialize service
        service = get_embedding_service()
        
        # Test single embedding
        result = await service.generate_embedding(
            "This is a test document about AI agents.",
            input_type="document"
        )
        print(f"Embedding shape: {len(result.embedding)}")
        print(f"Model: {result.model}")
        
        # Test batch embeddings
        texts = [
            "AI agents are autonomous systems",
            "Memory systems help agents remember",
            "LangGraph orchestrates agent workflows"
        ]
        results = await service.generate_embeddings_batch(texts)
        print(f"Generated {len(results)} embeddings")
        
        # Model info
        info = service.get_model_info()
        print(f"Model info: {info}")
    
    asyncio.run(main())
