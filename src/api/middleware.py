"""
API Middleware Components
Authentication, logging, and request processing
"""

import time
import logging
import jwt
from typing import Optional
from datetime import datetime

from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Log all requests and responses."""
    
    async def dispatch(self, request: Request, call_next):
        """Process request and log details."""
        start_time = time.time()
        
        # Log request
        logger.info(f"Request: {request.method} {request.url.path}")
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Log response
        logger.info(
            f"Response: {request.method} {request.url.path} "
            f"- Status: {response.status_code} - Duration: {duration:.3f}s"
        )
        
        # Add custom headers
        response.headers["X-Process-Time"] = str(duration)
        response.headers["X-Request-ID"] = f"{datetime.utcnow().timestamp()}"
        
        return response


class AuthMiddleware(BaseHTTPMiddleware):
    """Handle authentication for protected routes."""
    
    # Public endpoints that don't require auth
    PUBLIC_PATHS = [
        "/",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/health",
        "/health/live",
        "/health/ready",
        "/metrics"
    ]
    
    async def dispatch(self, request: Request, call_next):
        """Check authentication for protected routes."""
        # Skip auth for public paths
        if request.url.path in self.PUBLIC_PATHS:
            return await call_next(request)
        
        # Skip auth for OPTIONS requests (CORS preflight)
        if request.method == "OPTIONS":
            return await call_next(request)
        
        # Check for API key in header
        api_key = request.headers.get("X-API-Key")
        if api_key:
            # Validate API key (simplified - in production, check against database)
            if self._validate_api_key(api_key):
                request.state.auth_type = "api_key"
                request.state.authenticated = True
                return await call_next(request)
        
        # Check for JWT token
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            if self._validate_jwt(token):
                request.state.auth_type = "jwt"
                request.state.authenticated = True
                return await call_next(request)
        
        # Allow unauthenticated access for now (remove in production)
        # In production, uncomment the following lines:
        # raise HTTPException(
        #     status_code=status.HTTP_401_UNAUTHORIZED,
        #     detail="Authentication required"
        # )
        
        request.state.auth_type = "none"
        request.state.authenticated = False
        return await call_next(request)
    
    def _validate_api_key(self, api_key: str) -> bool:
        """
        Validate API key.
        In production, check against database.
        """
        # Simplified validation
        import os
        valid_keys = os.getenv("VALID_API_KEYS", "").split(",")
        return api_key in valid_keys if valid_keys else True
    
    def _validate_jwt(self, token: str) -> bool:
        """
        Validate JWT token.
        """
        try:
            import os
            secret = os.getenv("JWT_SECRET_KEY", "your-secret-key")
            payload = jwt.decode(token, secret, algorithms=["HS256"])
            
            # Check expiration
            if "exp" in payload:
                if datetime.fromtimestamp(payload["exp"]) < datetime.utcnow():
                    return False
            
            return True
        except jwt.InvalidTokenError:
            return False


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Custom rate limiting middleware.
    Works with slowapi for more sophisticated limiting.
    """
    
    def __init__(self, app, requests_per_minute: int = 100):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.request_counts = {}
    
    async def dispatch(self, request: Request, call_next):
        """Apply rate limiting."""
        # Get client identifier
        client_id = self._get_client_id(request)
        
        # Check rate limit
        current_time = time.time()
        minute_window = int(current_time // 60)
        
        key = f"{client_id}:{minute_window}"
        
        if key not in self.request_counts:
            self.request_counts[key] = 0
        
        self.request_counts[key] += 1
        
        # Clean old entries
        self._cleanup_old_entries(minute_window)
        
        if self.request_counts[key] > self.requests_per_minute:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded"
            )
        
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(
            self.requests_per_minute - self.request_counts[key]
        )
        response.headers["X-RateLimit-Reset"] = str((minute_window + 1) * 60)
        
        return response
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting."""
        # Try to get authenticated user ID
        if hasattr(request.state, "user_id"):
            return f"user:{request.state.user_id}"
        
        # Fall back to IP address
        return f"ip:{request.client.host}"
    
    def _cleanup_old_entries(self, current_window: int):
        """Remove old rate limit entries."""
        old_windows = [
            key for key in self.request_counts
            if int(key.split(":")[1]) < current_window - 1
        ]
        for key in old_windows:
            del self.request_counts[key]


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Global error handling middleware."""
    
    async def dispatch(self, request: Request, call_next):
        """Catch and handle errors."""
        try:
            response = await call_next(request)
            return response
        except HTTPException as e:
            # Re-raise HTTP exceptions
            raise e
        except Exception as e:
            # Log unexpected errors
            logger.error(f"Unexpected error: {e}", exc_info=True)
            
            # Return generic error response
            return Response(
                content={
                    "error": "Internal server error",
                    "message": str(e) if logger.level == logging.DEBUG else "An error occurred"
                },
                status_code=500,
                media_type="application/json"
            )
