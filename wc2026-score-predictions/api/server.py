"""FastAPI application entry point — Commercial security grade (Enterprise v2)

Security layers:
  - Bearer Token authentication
  - CORS restrictions
  - Rate limiting (60 per IP per minute)
  - Security response headers
  - Structured logging (structlog)
  - Unified exception handler
  - Prometheus /metrics
"""
from __future__ import annotations

import os
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import REGISTRY, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware

from db.config import APP_ENV
from db.etl import init_db
from models.exceptions import QuantBetError

# ---------------------------------------------------------------------------
# Structured logging (Phase 1b)
# ---------------------------------------------------------------------------
_log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------
class RateLimiter:
    """IP-based sliding window rate limiter (in-memory, single process)."""

    def __init__(self, max_requests: int = 60, window_seconds: int = 60) -> None:
        self.max_requests = max_requests
        self.window = window_seconds
        self._store: dict[str, list[float]] = defaultdict(list)

    def _clean(self, ip: str, now: float) -> None:
        cutoff = now - self.window
        self._store[ip] = [t for t in self._store[ip] if t > cutoff]

    def is_allowed(self, ip: str) -> bool:
        now = time.time()
        self._clean(ip, now)
        if len(self._store[ip]) >= self.max_requests:
            return False
        self._store[ip].append(now)
        return True

    def remaining(self, ip: str) -> int:
        self._clean(ip, time.time())
        return max(0, self.max_requests - len(self._store[ip]))


rate_limiter = RateLimiter(max_requests=60, window_seconds=60)

# ---------------------------------------------------------------------------
# API Token
# ---------------------------------------------------------------------------
API_TOKEN = os.getenv("API_TOKEN", "dev-token" if APP_ENV == "development" else "")

if not API_TOKEN:
    raise RuntimeError(
        "API_TOKEN environment variable must be set."
        "Development mode: export API_TOKEN=dev-token"
    )


def verify_token(request: Request) -> bool:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return False
    return auth[7:] == API_TOKEN


# ---------------------------------------------------------------------------
# Security middleware
# ---------------------------------------------------------------------------
class SecurityMiddleware(BaseHTTPMiddleware):
    """Authentication + rate limiting + security headers."""

    async def dispatch(self, request: Request, call_next):
        # Health check and metrics skip token verification
        if request.url.path in ("/health", "/api/v1/health", "/metrics"):
            return await call_next(request)

        if not verify_token(request):
            _log.warning("auth_failed", ip=request.client.host if request.client else "unknown")
            return JSONResponse(
                {"detail": "Unauthorized — need valid Bearer token"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )

        ip = request.client.host if request.client else "unknown"
        if not rate_limiter.is_allowed(ip):
            return JSONResponse(
                {"detail": "Too many requests, try again later"},
                status_code=429,
                headers={"Retry-After": "60", "X-RateLimit-Remaining": "0"},
            )

        response = await call_next(request)

        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-RateLimit-Remaining"] = str(rate_limiter.remaining(ip))

        return response


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    _log.info("api_startup", env=APP_ENV, model_version="v9")
    init_db()
    yield


app = FastAPI(
    title="QuantBet-EV API",
    description="Football prediction quantification system — Prediction + Data query",
    version="9.1.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:8000",
        "http://127.0.0.1",
        "http://127.0.0.1:8000",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

# Security middleware (auth + rate limiting + security headers)
app.add_middleware(SecurityMiddleware)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    trace_id = str(uuid.uuid4())[:8]
    start = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start) * 1000
    _log.info(
        "request",
        trace_id=trace_id,
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        duration_ms=round(duration_ms, 1),
    )
    response.headers["X-Trace-ID"] = trace_id
    return response


# ---------------------------------------------------------------------------
# Register v1 routes
# ---------------------------------------------------------------------------
from api.routes import router as v1_router

app.include_router(v1_router, prefix="/api/v1")


# ---------------------------------------------------------------------------
# Unified exception handler
# ---------------------------------------------------------------------------

@app.exception_handler(QuantBetError)
async def quantbet_error_handler(request: Request, exc: QuantBetError):
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)


@app.exception_handler(HTTPException)
async def http_error_handler(request: Request, exc: HTTPException):
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception):
    _log.error("unhandled_error", path=request.url.path, error=str(exc))
    return JSONResponse({"detail": "Internal server error"}, status_code=500)


# ---------------------------------------------------------------------------
# Prometheus /metrics
# ---------------------------------------------------------------------------
from prometheus_client import CONTENT_TYPE_LATEST

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


# ---------------------------------------------------------------------------
# Compatibility /health legacy path
# ---------------------------------------------------------------------------
@app.get("/health")
async def health_legacy():
    from api.routes import _get_model
    from api.schemas import HealthResponse
    from db.config import IS_SQLITE

    dc = _get_model()
    resp = HealthResponse(
        status="ok",
        model_loaded=True,
        gas_active=getattr(dc, "_gas", None) is not None,
        teams=len(dc.teams),
        database="sqlite" if IS_SQLITE else "postgresql",
    )
    return resp.model_dump()
