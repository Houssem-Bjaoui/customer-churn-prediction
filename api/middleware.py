"""
API Key authentication middleware.
Simple but sufficient for a portfolio project.
In production — use OAuth2 or JWT.
"""

import os
from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer(auto_error=False)

# Load from environment variable
# Never hardcode API keys
API_KEY = os.getenv("API_KEY", "dev_key_change_in_production")


async def verify_api_key(request: Request) -> None:
    """
    Dependency injected into protected endpoints.
    Checks Bearer token in Authorization header.
    """
    # Allow health check without auth
    if request.url.path in ["/health", "/", "/docs",
                              "/openapi.json", "/redoc"]:
        return

    auth_header = request.headers.get("Authorization", "")

    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error"  : "Missing API key",
                "detail" : "Include 'Authorization: "
                           "Bearer YOUR_KEY' header",
                "code"   : 403
            }
        )

    token = auth_header.split("Bearer ")[-1].strip()

    if token != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error" : "Invalid API key",
                "detail": "The provided API key is not valid",
                "code"  : 403
            }
        )