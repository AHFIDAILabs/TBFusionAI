"""
Main FastAPI application for TBFusionAI.

Entry point for the API server with:
- FastAPI app configuration
- CORS middleware
- Route registration
- Error handling
- Static file serving
- Graceful degradation when models not loaded
"""

from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.api.dependencies import get_app_config, get_predictor_optional
from src.api.routes import router as api_router
from src.api.schemas import ErrorResponse
from src.logger import get_logger

logger = get_logger(__name__)

# Get configuration
config = get_app_config()

# Create FastAPI app
app = FastAPI(
    title=config.api.app_title,
    description=config.api.app_description,
    version=config.api.app_version,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)


@app.middleware("http")
async def normalize_forwarded_scheme(request: Request, call_next):
    """Normalize request scheme behind reverse proxies (e.g., Render)."""
    forwarded_proto = request.headers.get("x-forwarded-proto", "")
    host = request.headers.get("host", "")

    if forwarded_proto:
        request.scope["scheme"] = forwarded_proto.split(",")[0].strip()
    elif host.endswith(".onrender.com"):
        request.scope["scheme"] = "https"

    return await call_next(request)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=config.api.cors_credentials,
    allow_methods=config.api.cors_methods,
    allow_headers=config.api.cors_headers,
)

# Setup templates and static files
project_root = Path(__file__).parent.parent.parent
templates_dir = project_root / "frontend" / "templates"
static_dir = project_root / "frontend" / "static"

templates = Jinja2Templates(directory=str(templates_dir))

# Mount static files
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
else:
    logger.warning(f"Static directory not found: {static_dir}")


# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors."""
    logger.warning(f"Validation error: {exc.errors()}")

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error="ValidationError",
            message="Invalid request data",
            detail=str(exc.errors()),
            timestamp=datetime.now().isoformat(),
        ).dict(),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    logger.error(f"HTTP error {exc.status_code}: {exc.detail}")

    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=f"HTTPException_{exc.status_code}",
            message=exc.detail,
            detail=None,
            timestamp=datetime.now().isoformat(),
        ).dict(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="InternalServerError",
            message="An unexpected error occurred",
            detail=str(exc),
            timestamp=datetime.now().isoformat(),
        ).dict(),
    )


# Frontend routes
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def home(request: Request):
    """Render home page."""
    try:
        return templates.TemplateResponse("home.html", {"request": request})
    except Exception as e:
        logger.error(f"Failed to render home page: {str(e)}")
        return HTMLResponse(
            content="""
            <html>
                <head><title>TBFusionAI</title></head>
                <body>
                    <h1>TBFusionAI</h1>
                    <p>AI-powered TB Detection</p>
                    <p><a href="/api/docs">API Documentation</a></p>
                </body>
            </html>
            """,
            status_code=200,
        )


@app.get("/prediction", response_class=HTMLResponse, include_in_schema=False)
async def prediction_page(request: Request):
    """Render prediction page."""
    try:
        return templates.TemplateResponse("prediction.html", {"request": request})
    except Exception as e:
        logger.error(f"Failed to render prediction page: {str(e)}")
        return HTMLResponse(
            content="""
            <html>
                <head><title>Prediction - TBFusionAI</title></head>
                <body>
                    <h1>TB Prediction</h1>
                    <p>Use the <a href="/api/docs">API</a> to make predictions</p>
                </body>
            </html>
            """,
            status_code=200,
        )


@app.get("/faq", response_class=HTMLResponse, include_in_schema=False)
async def faq_page(request: Request):
    """Render FAQ page."""
    try:
        return templates.TemplateResponse("faq.html", {"request": request})
    except Exception as e:
        logger.error(f"Failed to render FAQ page: {str(e)}")
        return HTMLResponse(
            content="""
            <html>
                <head><title>FAQ - TBFusionAI</title></head>
                <body>
                    <h1>FAQ</h1>
                    <p>Documentation coming soon</p>
                </body>
            </html>
            """,
            status_code=200,
        )


# Include API routes
app.include_router(api_router, prefix=config.api.api_prefix, tags=["Predictions"])


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("=" * 70)
    logger.info("STARTING TBFUSIONAI API")
    logger.info("=" * 70)
    logger.info(f"Version: {config.api.app_version}")
    logger.info(f"Docs: http://localhost:{config.api.port}/api/docs")
    logger.info(f"Home: http://localhost:{config.api.port}/")
    logger.info("=" * 70)

    # Try to load models (non-blocking)
    predictor = get_predictor_optional()

    if predictor is not None:
        logger.info("✓ Models loaded successfully - Ready for predictions")
    else:
        logger.warning("=" * 70)
        logger.warning("⚠ MODELS NOT LOADED - SETUP REQUIRED")
        logger.warning("=" * 70)
        logger.warning("The API is running but predictions are not available.")
        logger.warning("")
        logger.warning("To enable predictions, run the ML pipeline:")
        logger.warning("")
        logger.warning(
            "  Docker:  docker exec tbfusionai-api python main.py run-pipeline"
        )
        logger.warning("  Local:   python main.py run-pipeline")
        logger.warning("")
        logger.warning("This will:")
        logger.warning("  1. Download the CODA TB dataset (~15 min)")
        logger.warning("  2. Extract audio features (~60 min)")
        logger.warning("  3. Train ML models (~30 min)")
        logger.warning("  4. Create ensemble model (~5 min)")
        logger.warning("")
        logger.warning("Total time: ~2 hours")
        logger.warning("")
        logger.warning(
            "Check status: http://localhost:{}/api/v1/status".format(config.api.port)
        )
        logger.warning("=" * 70)


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down TBFusionAI API")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.reload,
        workers=config.api.workers,
    )
