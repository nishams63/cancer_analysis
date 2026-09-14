"""FastAPI application for AADA Integration Service."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .routes import router
from .errors import AADAError, aada_error_handler, http_exception_handler, generic_exception_handler

app = FastAPI(
    title="AADA Autonomous AI Data Analyst - Integration Service",
    version="1.0.0",
    description="Unified Integration and Observability API coordinating Knowledge, Workflow, Agent, and Evaluation Engineers.",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception Handlers
app.add_exception_handler(AADAError, aada_error_handler)
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

# Include API Router
app.include_router(router)


@app.get("/health", tags=["System"])
def health_check():
    return {
        "status": "healthy",
        "service": "AADA Integration Service",
        "version": "1.0.0",
        "connected_components": [
            "knowledge-engineer",
            "workflow-engineer",
            "agent-engineer",
            "evaluation-engineer",
        ],
    }
