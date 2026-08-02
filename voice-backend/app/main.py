from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api import animations, psl
from app.services.psl_inference import load_model
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Create FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=settings.DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
)


# Configure CORS - Allow frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)


# Health check endpoint
@app.get("/health")
async def health_check():
    """
    Simple health check endpoint to verify the backend is running.

    Returns:
        Status object indicating the service is healthy
    """
    return {
        "status": "ok",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
    }


# Include API routers
app.include_router(
    animations.router,
    prefix="/api",
    tags=["animations"]
)

app.include_router(
    psl.router,
    prefix="/api/psl",
    tags=["PSL Recognition"]
)


# Startup event: Load PSL model
@app.on_event("startup")
async def startup_event():
    """Load the PSL recognition model on application startup"""
    logger.info("Starting D-VOICE backend...")
    try:
        logger.info("Loading PSL recognition model...")
        load_model()
        logger.info("PSL model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load PSL model: {str(e)}")
        logger.warning("PSL recognition endpoints will not be available")


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "D-VOICE Animation Service API",
        "version": settings.VERSION,
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "animations": "/api/animations",
            "psl_recognition": "/api/psl/recognize",
            "psl_health": "/api/psl/health",
            "psl_model_info": "/api/psl/model-info"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True
    )
