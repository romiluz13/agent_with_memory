"""
Document Ingestion Module
Based on MongoDB's atlas-search-playground-chatbot-starter/ingest-data.js
Handles PDF processing, chunking, embedding generation, and storage
"""

import os
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from langchain_community.document_loaders import PyPDFLoader, TextLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_voyageai import VoyageAIEmbeddings
from pymongo import MongoClient
import tiktoken

logger = logging.getLogger(__name__)


@dataclass
class IngestionConfig:
    """Configuration for document ingestion (from MongoDB starter)."""
    chunk_size: int = 1000  # MongoDB's recommended chunk size
    chunk_overlap: int = 200  # MongoDB's recommended overlap
    embedding_model: str = "voyage-3-large"
    embedding_dimensions: int = 1024
    batch_size: int = 10  # For batch embedding generation


class MongoDBDocumentIngestion:
    """
    Document ingestion system based on MongoDB's patterns.
    Follows exact patterns from atlas-search-playground-chatbot-starter.
    """
    
    def __init__(
        self,
        mongodb_uri: str,
        database_name: str = "ai_agent_boilerplate",
        collection_name: str = "documents",
        config: Optional[IngestionConfig] = None
    ):
        """
        Initialize the ingestion system.
        
        Args:
            mongodb_uri: MongoDB connection string
            database_name: Database name
            collection_name: Collection name for documents
            config: Ingestion configuration
        """
        self.mongodb_uri = mongodb_uri
        self.database_name = database_name
        self.collection_name = collection_name
        self.config = config or IngestionConfig()
        
        # Initialize MongoDB client
        self.client = MongoClient(mongodb_uri)
        self.db = self.client[database_name]
        self.collection = self.db[collection_name]
        
        # Initialize embeddings
        self.embedding_model = VoyageAIEmbeddings(
            model=self.config.embedding_model,
            output_dimension=self.config.embedding_dimensions
        )
        
        # Initialize tokenizer for accurate chunk sizing
        self.encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
        
        logger.info(f"Initialized MongoDB Document Ingestion for {database_name}.{collection_name}")
    
    def get_token_count(self, text: str) -> int:
        """Get token count for text (from MongoDB starter)."""
        return len(self.encoding.encode(text))
    
    async def ingest_pdf(
        self,
        pdf_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Ingest a PDF document (based on MongoDB's ingest-data.js).
        
        Args:
            pdf_path: Path to PDF file
            metadata: Additional metadata for the document
            
        Returns:
            Ingestion results
        """
        try:
            # Load PDF (MongoDB pattern)
            loader = PyPDFLoader(pdf_path)
            data = await loader.aload()
            
            # Configure text splitter with token counter (MongoDB pattern)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                length_function=self.get_token_count,
                separators=["\n\n", "\n", " ", ""]
            )
            
            # Split documents
            docs = await text_splitter.asplit_documents(data)
            logger.info(f"Successfully chunked PDF into {len(docs)} documents")
            
            # Clear existing documents if needed
            logger.info("Clearing collection of any pre-existing data")
            delete_result = self.collection.delete_many({})
            logger.info(f"Deleted {delete_result.deleted_count} documents")
            
            # Generate embeddings in batches
            logger.info("Generating embeddings and inserting documents...")
            
            # Extract text from documents
            texts = [doc.page_content for doc in docs]
            
            # Generate embeddings in batches
            all_embeddings = []
            for i in range(0, len(texts), self.config.batch_size):
                batch_texts = texts[i:i + self.config.batch_size]
                batch_embeddings = await self.embedding_model.aembed_documents(batch_texts)
                all_embeddings.extend(batch_embeddings)
            
            # Prepare documents for insertion (MongoDB pattern)
            insert_documents = []
            for index, (doc, embedding) in enumerate(zip(docs, all_embeddings)):
                document = {
                    "_id": index,
                    "text": doc.page_content,
                    "vector_embeddings": embedding,
                    "page_number": doc.metadata.get("page", 0),
                    "source": pdf_path,
                    "metadata": {**doc.metadata, **(metadata or {})}
                }
                insert_documents.append(document)
            
            # Bulk insert with ordered=False for better performance
            options = {"ordered": False}
            result = self.collection.insert_many(insert_documents, **options)
            
            logger.info(f"Count of documents inserted: {len(result.inserted_ids)}")
            
            return {
                "status": "success",
                "documents_processed": len(docs),
                "documents_inserted": len(result.inserted_ids),
                "source": pdf_path
            }
            
        except Exception as e:
            logger.error(f"Failed to ingest PDF: {e}")
            return {
                "status": "error",
                "error": str(e),
                "source": pdf_path
            }
    
    async def ingest_text(
        self,
        text_content: str,
        source_name: str = "text_input",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Ingest raw text content.
        
        Args:
            text_content: Text to ingest
            source_name: Name of the source
            metadata: Additional metadata
            
        Returns:
            Ingestion results
        """
        try:
            # Create text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                length_function=self.get_token_count
            )
            
            # Split text
            chunks = text_splitter.split_text(text_content)
            logger.info(f"Split text into {len(chunks)} chunks")
            
            # Generate embeddings
            embeddings = await self.embedding_model.aembed_documents(chunks)
            
            # Prepare documents
            insert_documents = []
            for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                document = {
                    "_id": f"{source_name}_{index}",
                    "text": chunk,
                    "vector_embeddings": embedding,
                    "source": source_name,
                    "chunk_index": index,
                    "metadata": metadata or {}
                }
                insert_documents.append(document)
            
            # Insert documents
            result = self.collection.insert_many(insert_documents, ordered=False)
            
            return {
                "status": "success",
                "chunks_created": len(chunks),
                "documents_inserted": len(result.inserted_ids),
                "source": source_name
            }
            
        except Exception as e:
            logger.error(f"Failed to ingest text: {e}")
            return {
                "status": "error",
                "error": str(e),
                "source": source_name
            }
    
    async def retrieve_documents(
        self,
        query: str,
        num_candidates: int = 100,  # MongoDB's recommended value
        limit: int = 5,
        exact: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents using vector search (from retrieve-documents.js).
        
        Args:
            query: Search query
            num_candidates: Number of candidates to consider
            limit: Number of results to return
            exact: Whether to use exact search
            
        Returns:
            List of retrieved documents
        """
        try:
            # Generate query embedding
            query_embedding = await self.embedding_model.aembed_query(query)
            
            # MongoDB vector search aggregation pipeline (exact pattern from starter)
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "vector_index",
                        "queryVector": query_embedding,
                        "path": "vector_embeddings",
                        "numCandidates": num_candidates,
                        "exact": exact,
                        "limit": limit
                    }
                },
                {
                    "$project": {
                        "vector_embeddings": 0,  # Exclude embeddings from results
                        "text": 1,
                        "page_number": 1,
                        "source": 1,
                        "metadata": 1,
                        "score": {"$meta": "vectorSearchScore"}
                    }
                }
            ]
            
            # Execute aggregation
            results = list(self.collection.aggregate(pipeline))
            
            logger.debug(f"Retrieved {len(results)} documents for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Failed to retrieve documents: {e}")
            return []
    
    def create_vector_index(self):
        """
        Create vector search index (from build-vector-index.js).
        Uses exact configuration from MongoDB starter.
        """
        try:
            # MongoDB's exact index configuration
            index_definition = {
                "name": "vector_index",
                "type": "vectorSearch",
                "definition": {
                    "fields": [
                        {
                            "type": "vector",
                            "path": "vector_embeddings",
                            "similarity": "cosine",
                            "numDimensions": self.config.embedding_dimensions
                        }
                    ]
                }
            }
            
            # Create the index
            result = self.collection.create_search_index(index_definition)
            logger.info(f"Successfully created vector index: {result}")
            return result
            
        except Exception as e:
            if "already exists" in str(e):
                logger.info("Vector index already exists")
            else:
                logger.error(f"Failed to create vector index: {e}")
                raise


# RAG Response Generator (from generate-response.js)
class MongoDBRAGGenerator:
    """
    RAG response generator based on MongoDB's patterns.
    """
    
    def __init__(
        self,
        ingestion_system: MongoDBDocumentIngestion,
        llm_provider: str = "openai",
        model_name: str = "gpt-4o"
    ):
        """
        Initialize RAG generator.
        
        Args:
            ingestion_system: Document ingestion system
            llm_provider: LLM provider
            model_name: Model name
        """
        self.ingestion = ingestion_system
        
        # Initialize LLM
        if llm_provider == "openai":
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(model=model_name)
        elif llm_provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            self.llm = ChatAnthropic(model=model_name)
        else:
            raise ValueError(f"Unknown provider: {llm_provider}")
    
    async def generate_response(
        self,
        question: str,
        num_candidates: int = 100,
        limit: int = 5
    ) -> str:
        """
        Generate RAG response (from generate-response.js).
        
        Args:
            question: User question
            num_candidates: Number of candidates for vector search
            limit: Number of documents to retrieve
            
        Returns:
            Generated response
        """
        try:
            # Retrieve relevant documents
            documents = await self.ingestion.retrieve_documents(
                query=question,
                num_candidates=num_candidates,
                limit=limit,
                exact=False
            )
            
            # Build context from documents (MongoDB pattern)
            context = "\n\n".join([doc["text"] for doc in documents])
            
            # Create prompt (from MongoDB's generate-response.js)
            prompt = f"""A text is split into several chunks and you are provided a subset of these chunks as context to answer the question at the end.
Respond appropriately if the question cannot be feasibly answered without access to the full text.
Acknowledge limitations when the context provided is incomplete or does not contain relevant information to answer the question.
If you need to fill knowledge gaps using information outside of the context, clearly attribute it as such.

Context: {context}

Question: {question}"""
            
            # Generate response
            response = await self.llm.ainvoke(prompt)
            
            return response.content
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return f"Error generating response: {str(e)}"


# Example usage
if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv
    
    load_dotenv()
    
    async def main():
        # Initialize ingestion system
        ingestion = MongoDBDocumentIngestion(
            mongodb_uri=os.getenv("MONGODB_URI"),
            database_name="ai_agent_boilerplate",
            collection_name="documents"
        )
        
        # Create vector index
        ingestion.create_vector_index()
        
        # Ingest a sample PDF
        # result = await ingestion.ingest_pdf("sample.pdf")
        # print(f"Ingestion result: {result}")
        
        # Test retrieval
        results = await ingestion.retrieve_documents(
            "What are some movies that take place in the ocean?",
            num_candidates=100,
            limit=5
        )
        
        for doc in results:
            print(f"Score: {doc['score']:.4f} - Text: {doc['text'][:100]}...")
    
    asyncio.run(main()
