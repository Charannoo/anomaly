"""Main FastAPI entrypoint for PNTC Inspect backend."""

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .database import init_db
from .routes.assistant import router as assistant_router
from .routes.demo import router as demo_router
from .routes.inspections import router as inspections_router
from .routes.model import router as model_router

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context for database setup and readiness."""
    init_db()
    yield


app = FastAPI(
    title="PNTC Inspect API",
    description="Backend API for Multimodal RGB–3D Industrial Anomaly Detection and Metrology",
    version="5.4.0",
    lifespan=lifespan,
)

# CORS configuration for modern frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include route namespaces
app.include_router(inspections_router)
app.include_router(assistant_router)
app.include_router(model_router)
app.include_router(demo_router)


@app.get("/api/system/status")
def get_system_status():
    """Return backend operational status, detector freeze confirmation, and server metadata."""
    return {
        "status": "OPERATIONAL",
        "system_name": "PNTC Inspect",
        "detector_status": "FROZEN_VERIFIED",
        "canonical_tag": "h5d-pntc-verified",
        "canonical_metrics": {
            "I-AUROC": 0.96541000,
            "P-AUROC": 0.99416000,
            "AUPRO@0.3": 0.96939000,
        },
        "version": "5.4.0",
        "timestamp": time.time(),
    }


@app.get("/api/health")
def healthcheck():
    return {"status": "ok", "service": "pntc-inspect"}
