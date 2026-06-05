"""
FastAPI application — Customer Churn Prediction API
Endpoints: /predict  /batch  /health  /metrics

Run locally:
    uvicorn api.main:app --reload --port 8000

With API key:
    export API_KEY=your_secret_key
    uvicorn api.main:app --reload --port 8000
"""

import os
import time
import uuid
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import (
    FastAPI, Depends, HTTPException,
    UploadFile, File, Request, status
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import io

from api.schemas import (
    CustomerRequest, PredictionResponse,
    BatchResponse, BatchSummary,
    HealthResponse, FeatureImpact,
    PredictionExplanation
)
from api.predictor import ChurnPredictor
from api.middleware import verify_api_key

from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="Authorization")
# ── Logging ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(name)s — %(levelname)s — %(message)s'
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────
MODELS_PATH = Path(
    os.getenv("MODELS_PATH", "models/")
)

# ── Global predictor instance ─────────────────────────────
# Loaded ONCE at startup — shared across all requests
predictor = ChurnPredictor(models_path=MODELS_PATH)
START_TIME = time.time()


# ── Lifespan — startup and shutdown ──────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load model artifacts on startup.
    Clean up on shutdown.
    """
    logger.info("Starting up — loading model artifacts...")
    predictor.load()
    logger.info("✓ API ready to serve predictions")
    yield
    logger.info("Shutting down...")


# ── FastAPI app ───────────────────────────────────────────
app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "Predicts customer churn probability using XGBoost "
        "with SHAP explainability. "
        "Built as a portfolio project demonstrating "
        "production ML engineering."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# ── CORS ──────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request logging middleware ────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = (time.time() - start) * 1000
    logger.info(
        f"{request.method} {request.url.path} "
        f"→ {response.status_code} "
        f"({duration:.1f}ms)"
    )
    return response


# ════════════════════════════════════════════════════════
#  ENDPOINTS
# ════════════════════════════════════════════════════════

# ── GET / ─────────────────────────────────────────────────
@app.get("/", tags=["Root"])
async def root():
    """API information and available endpoints."""
    return {
        "name"       : "Customer Churn Prediction API",
        "version"    : "1.0.0",
        "status"     : "running",
        "docs"       : "/docs",
        "endpoints"  : {
            "health" : "GET  /health",
            "predict": "POST /predict",
            "batch"  : "POST /batch",
            "metrics": "GET  /metrics"
        }
    }


# ── GET /health ───────────────────────────────────────────
@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Monitoring"]
)
async def health_check():
    """
    Health check endpoint.
    No authentication required.
    Use this for Docker health checks and uptime monitoring.
    """
    return HealthResponse(
        status          = "healthy" if predictor.is_loaded
                          else "degraded",
        model_version   = predictor.model_version
                          if predictor.is_loaded else "unknown",
        model_loaded    = predictor.is_loaded,
        explainer_loaded= predictor.explainer is not None,
        auc_roc         = predictor.metadata['auc_roc']
                          if predictor.is_loaded else 0.0,
        threshold       = predictor.threshold
                          if predictor.is_loaded else 0.5,
        uptime_seconds  = round(time.time() - START_TIME, 1)
    )


# ── POST /predict ─────────────────────────────────────────
@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    dependencies=[Depends(verify_api_key)]
)
async def predict_single(
    customer: CustomerRequest
):
    """
    Predict churn probability for a single customer.

    Returns:
    - **churn_probability**: float between 0 and 1
    - **churn_label**: 0 (stays) or 1 (churns)
    - **risk_tier**: HIGH / MEDIUM / LOW
    - **explanation**: top 5 SHAP feature impacts

    Example curl:
    curl -X POST http://localhost:8000/predict \\
  -H "Authorization: Bearer your_api_key" \\
  -H "Content-Type: application/json" \\
  -d '{"tenure_months": 5, "contract": "Month-to-month", ...}'
  """
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please try again."
        )

    try:
        result = predictor.predict_single(
            customer.model_dump()
        )

        # Build typed response
        explanation = PredictionExplanation(
            base_value   = result['explanation']['base_value'],
            shap_sum     = result['explanation']['shap_sum'],
            top_features = [
                FeatureImpact(**f)
                for f in result['explanation']['top_features']
            ]
        )

        return PredictionResponse(
            churn_probability = result['churn_probability'],
            churn_label       = result['churn_label'],
            risk_tier         = result['risk_tier'],
            threshold_used    = result['threshold_used'],
            model_version     = result['model_version'],
            explanation       = explanation
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal prediction error"
        )


# ── POST /batch ───────────────────────────────────────────
@app.post(
    "/batch",
    response_model=BatchResponse,
    tags=["Prediction"],
    dependencies=[Depends(verify_api_key)]
)
async def predict_batch(
    file: UploadFile = File(
        ...,
        description="CSV file with customer records. "
                    "Max 10,000 rows."
    )
):
    """
    Batch prediction from CSV file upload.

    CSV must contain the same columns as the /predict endpoint.
    Returns predictions for all rows plus a summary.
    Max file size: 10MB. Max rows: 10,000.
    """
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded."
        )

    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Only CSV files accepted."
        )

    try:
        start_time = time.time()

        # Read CSV
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))

        # Validate row count
        if len(df) > 10_000:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"File has {len(df):,} rows. "
                       f"Maximum allowed is 10,000."
            )

        if len(df) == 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="CSV file is empty."
            )

        # Convert to list of dicts
        records = df.to_dict(orient='records')

        # Run batch predictions
        predictions = predictor.predict_batch(records)

        # Build summary
        successful = [
            p for p in predictions
            if p.get('status') == 'success'
        ]
        probs = [
            p['churn_probability']
            for p in successful
        ]

        summary = BatchSummary(
            total_records    = len(predictions),
            high_risk_count  = sum(
                1 for p in successful
                if p['risk_tier'] == 'HIGH'
            ),
            medium_risk_count= sum(
                1 for p in successful
                if p['risk_tier'] == 'MEDIUM'
            ),
            low_risk_count   = sum(
                1 for p in successful
                if p['risk_tier'] == 'LOW'
            ),
            avg_churn_prob   = round(
                float(np.mean(probs)), 4
            ) if probs else 0.0,
            processing_time_s= round(
                time.time() - start_time, 2
            )
        )

        batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        logger.info(
            f"Batch {batch_id}: {len(predictions)} records, "
            f"{summary.high_risk_count} high risk, "
            f"{summary.processing_time_s}s"
        )

        return BatchResponse(
            batch_id   = batch_id,
            summary    = summary,
            predictions= predictions
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch processing error: {str(e)}"
        )


# ── GET /metrics ──────────────────────────────────────────
@app.get(
    "/metrics",
    tags=["Monitoring"],
    dependencies=[Depends(verify_api_key)]
)
async def model_metrics():
    """
    Return model performance metrics from training.
    Useful for monitoring model quality over time.
    """
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded."
        )

    return {
        "model_version"    : predictor.metadata['model_name'],
        "trained_at"       : predictor.metadata['trained_at'],
        "performance"      : {
            "auc_roc"      : predictor.metadata['auc_roc'],
            "pr_auc"       : predictor.metadata['pr_auc'],
            "f1_score"     : predictor.metadata['f1_score'],
            "precision"    : predictor.metadata['precision'],
            "recall"       : predictor.metadata['recall'],
            "threshold"    : predictor.metadata[
                                'optimal_threshold'
                             ],
        },
        "training_data"    : {
            "train_size"   : predictor.metadata['train_size'],
            "test_size"    : predictor.metadata['test_size'],
            "churn_rate"   : predictor.metadata[
                                'churn_rate_train'
                             ],
            "n_features"   : predictor.metadata['n_features'],
        },
        "uptime_seconds"   : round(
            time.time() - START_TIME, 1
        )
    }