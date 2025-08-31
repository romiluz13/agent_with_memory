"""
Galileo AI Observability Integration
Replaces Prometheus/Grafana with Galileo for LLM monitoring
Based on Galileo's RAG MongoDB LangChain integration patterns
"""

import os
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from contextlib import contextmanager
import json

try:
    import galileo_observe as galileo
    from galileo_observe import GalileoObserve
    from galileo_observe.langchain import GalileoCallbackHandler
    GALILEO_AVAILABLE = True
except ImportError:
    GALILEO_AVAILABLE = False
    logging.warning("Galileo AI not installed. Install with: pip install galileo-observe")

logger = logging.getLogger(__name__)


class GalileoMonitor:
    """
    Galileo AI monitoring for LLM applications.
    Provides observability for RAG, agents, and LLM interactions.
    """
    
    def __init__(
        self,
        project_name: str = "ai-agent-boilerplate",
        api_key: Optional[str] = None,
        environment: str = "production",
        enable_prompt_monitoring: bool = True,
        enable_retrieval_monitoring: bool = True,
        enable_generation_monitoring: bool = True
    ):
        """
        Initialize Galileo monitoring.
        
        Args:
            project_name: Galileo project name
            api_key: Galileo API key (or set GALILEO_API_KEY env var)
            environment: Environment name (development, staging, production)
            enable_prompt_monitoring: Monitor prompts and templates
            enable_retrieval_monitoring: Monitor RAG retrieval quality
            enable_generation_monitoring: Monitor LLM generation quality
        """
        if not GALILEO_AVAILABLE:
            logger.error("Galileo AI not available. Monitoring disabled.")
            self.enabled = False
            return
        
        self.project_name = project_name
        self.environment = environment
        self.api_key = api_key or os.getenv("GALILEO_API_KEY")
        
        if not self.api_key:
            logger.warning("No Galileo API key found. Monitoring disabled.")
            self.enabled = False
            return
        
        # Initialize Galileo
        try:
            galileo.init(
                project_name=project_name,
                api_key=self.api_key,
                environment=environment
            )
            
            self.observer = GalileoObserve()
            self.enabled = True
            
            # Configuration flags
            self.enable_prompt_monitoring = enable_prompt_monitoring
            self.enable_retrieval_monitoring = enable_retrieval_monitoring
            self.enable_generation_monitoring = enable_generation_monitoring
            
            logger.info(f"Galileo monitoring initialized for project: {project_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Galileo: {e}")
            self.enabled = False
    
    def get_langchain_callback(self) -> Optional[GalileoCallbackHandler]:
        """
        Get Galileo callback handler for LangChain.
        
        Returns:
            Galileo callback handler or None if disabled
        """
        if not self.enabled:
            return None
        
        return GalileoCallbackHandler(
            project_name=self.project_name,
            environment=self.environment
        )
    
    @contextmanager
    def trace_generation(
        self,
        operation_name: str,
        model: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Context manager for tracing LLM generation.
        
        Args:
            operation_name: Name of the operation
            model: Model being used
            user_id: User identifier
            session_id: Session identifier
            metadata: Additional metadata
        """
        if not self.enabled or not self.enable_generation_monitoring:
            yield
            return
        
        trace_id = self.observer.start_trace(
            name=operation_name,
            metadata={
                "model": model,
                "user_id": user_id,
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat(),
                **(metadata or {})
            }
        )
        
        try:
            yield trace_id
        finally:
            self.observer.end_trace(trace_id)
    
    def log_retrieval(
        self,
        query: str,
        retrieved_documents: List[Dict[str, Any]],
        relevance_scores: Optional[List[float]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        """
        Log RAG retrieval for monitoring.
        
        Args:
            query: Search query
            retrieved_documents: Retrieved documents
            relevance_scores: Relevance scores for documents
            user_id: User identifier
            session_id: Session identifier
        """
        if not self.enabled or not self.enable_retrieval_monitoring:
            return
        
        try:
            self.observer.log_retrieval(
                query=query,
                documents=[doc.get("text", "") for doc in retrieved_documents],
                scores=relevance_scores or [doc.get("score", 0.0) for doc in retrieved_documents],
                metadata={
                    "user_id": user_id,
                    "session_id": session_id,
                    "document_count": len(retrieved_documents),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            logger.debug(f"Logged retrieval for query: {query[:50]}...")
            
        except Exception as e:
            logger.error(f"Failed to log retrieval: {e}")
    
    def log_generation(
        self,
        prompt: str,
        response: str,
        model: str,
        latency_ms: Optional[float] = None,
        token_count: Optional[int] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log LLM generation for monitoring.
        
        Args:
            prompt: Input prompt
            response: Generated response
            model: Model used
            latency_ms: Generation latency in milliseconds
            token_count: Number of tokens generated
            user_id: User identifier
            session_id: Session identifier
            metadata: Additional metadata
        """
        if not self.enabled or not self.enable_generation_monitoring:
            return
        
        try:
            self.observer.log_generation(
                prompt=prompt,
                response=response,
                model=model,
                metadata={
                    "latency_ms": latency_ms,
                    "token_count": token_count,
                    "user_id": user_id,
                    "session_id": session_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    **(metadata or {})
                }
            )
            
            logger.debug(f"Logged generation for model: {model}")
            
        except Exception as e:
            logger.error(f"Failed to log generation: {e}")
    
    def log_prompt_template(
        self,
        template_name: str,
        template_content: str,
        variables: Dict[str, Any],
        rendered_prompt: str
    ):
        """
        Log prompt template usage.
        
        Args:
            template_name: Name of the template
            template_content: Template content
            variables: Template variables
            rendered_prompt: Rendered prompt
        """
        if not self.enabled or not self.enable_prompt_monitoring:
            return
        
        try:
            self.observer.log_prompt(
                template_name=template_name,
                template=template_content,
                variables=variables,
                rendered=rendered_prompt,
                metadata={
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            logger.debug(f"Logged prompt template: {template_name}")
            
        except Exception as e:
            logger.error(f"Failed to log prompt template: {e}")
    
    def log_error(
        self,
        error_type: str,
        error_message: str,
        operation: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log errors for monitoring.
        
        Args:
            error_type: Type of error
            error_message: Error message
            operation: Operation that failed
            user_id: User identifier
            session_id: Session identifier
            metadata: Additional metadata
        """
        if not self.enabled:
            return
        
        try:
            self.observer.log_error(
                error_type=error_type,
                message=error_message,
                operation=operation,
                metadata={
                    "user_id": user_id,
                    "session_id": session_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    **(metadata or {})
                }
            )
            
            logger.error(f"Logged error in {operation}: {error_type}")
            
        except Exception as e:
            logger.error(f"Failed to log error: {e}")
    
    def log_feedback(
        self,
        feedback_type: str,  # positive, negative, correction
        content: str,
        related_generation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        """
        Log user feedback for model improvement.
        
        Args:
            feedback_type: Type of feedback
            content: Feedback content
            related_generation_id: ID of related generation
            user_id: User identifier
            session_id: Session identifier
        """
        if not self.enabled:
            return
        
        try:
            self.observer.log_feedback(
                feedback_type=feedback_type,
                content=content,
                generation_id=related_generation_id,
                metadata={
                    "user_id": user_id,
                    "session_id": session_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            logger.debug(f"Logged {feedback_type} feedback")
            
        except Exception as e:
            logger.error(f"Failed to log feedback: {e}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary of monitored metrics.
        
        Returns:
            Metrics summary
        """
        if not self.enabled:
            return {"status": "disabled"}
        
        try:
            return {
                "status": "enabled",
                "project": self.project_name,
                "environment": self.environment,
                "monitoring": {
                    "prompts": self.enable_prompt_monitoring,
                    "retrieval": self.enable_retrieval_monitoring,
                    "generation": self.enable_generation_monitoring
                },
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get metrics summary: {e}")
            return {"status": "error", "error": str(e)}


# FastAPI integration
class GalileoMiddleware:
    """
    FastAPI middleware for Galileo monitoring.
    """
    
    def __init__(self, app, monitor: GalileoMonitor):
        """
        Initialize middleware.
        
        Args:
            app: FastAPI app
            monitor: Galileo monitor instance
        """
        self.app = app
        self.monitor = monitor
    
    async def __call__(self, request, call_next):
        """Process request with monitoring."""
        start_time = datetime.utcnow()
        
        # Track request
        request_id = f"req_{start_time.timestamp()}"
        
        try:
            # Process request
            response = await call_next(request)
            
            # Log successful request
            if self.monitor.enabled:
                latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                # Log API call metrics
                self.monitor.observer.log_api_call(
                    endpoint=str(request.url.path),
                    method=request.method,
                    status_code=response.status_code,
                    latency_ms=latency_ms,
                    metadata={
                        "request_id": request_id,
                        "timestamp": start_time.isoformat()
                    }
                )
            
            return response
            
        except Exception as e:
            # Log error
            if self.monitor.enabled:
                self.monitor.log_error(
                    error_type=type(e).__name__,
                    error_message=str(e),
                    operation=f"{request.method} {request.url.path}",
                    metadata={"request_id": request_id}
                )
            raise


# Singleton instance
_galileo_monitor: Optional[GalileoMonitor] = None


def get_galileo_monitor() -> GalileoMonitor:
    """Get or create Galileo monitor instance."""
    global _galileo_monitor
    
    if _galileo_monitor is None:
        _galileo_monitor = GalileoMonitor(
            project_name=os.getenv("GALILEO_PROJECT_NAME", "ai-agent-boilerplate"),
            api_key=os.getenv("GALILEO_API_KEY"),
            environment=os.getenv("ENVIRONMENT", "production")
        )
    
    return _galileo_monitor


# Example usage
if __name__ == "__main__":
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Initialize monitor
    monitor = get_galileo_monitor()
    
    # Example: Monitor a generation
    with monitor.trace_generation("test_generation", "gpt-4o") as trace_id:
        # Simulate generation
        prompt = "What is the capital of France?"
        response = "The capital of France is Paris."
        
        # Log the generation
        monitor.log_generation(
            prompt=prompt,
            response=response,
            model="gpt-4o",
            latency_ms=150.5,
            token_count=10
        )
    
    # Example: Monitor retrieval
    monitor.log_retrieval(
        query="What are AI agents?",
        retrieved_documents=[
            {"text": "AI agents are autonomous systems...", "score": 0.95},
            {"text": "Agents can perform tasks...", "score": 0.87}
        ]
    )
    
    # Get metrics summary
    summary = monitor.get_metrics_summary()
    print(f"Metrics: {json.dumps(summary, indent=2)}")
