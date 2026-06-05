"""
Pydantic schemas for request and response validation.
Every field validated before touching the model.
"""

from pydantic import BaseModel, Field, field_validator
from enum import Enum


# ── Enums for categorical fields ─────────────────────────
class ContractType(str, Enum):
    month_to_month = "Month-to-month"
    one_year       = "One year"
    two_year       = "Two year"

class InternetService(str, Enum):
    dsl        = "DSL"
    fiber      = "Fiber optic"
    no_service = "No"

class PaymentMethod(str, Enum):
    electronic_check = "Electronic check"
    mailed_check     = "Mailed check"
    bank_transfer    = "Bank transfer"
    credit_card      = "Credit card"

class Gender(str, Enum):
    male   = "Male"
    female = "Female"

class YesNo(str, Enum):
    yes = "Yes"
    no  = "No"

class RiskTier(str, Enum):
    high   = "HIGH"
    medium = "MEDIUM"
    low    = "LOW"


# ── Request schema ────────────────────────────────────────
class CustomerRequest(BaseModel):
    """
    Single customer prediction request.
    All fields match the cleaned dataset schema.
    """
    # Demographics
    age              : int   = Field(..., ge=18, le=100,
                                     description="Customer age")
    gender           : Gender
    senior_citizen   : int   = Field(..., ge=0, le=1)
    has_partner      : int   = Field(..., ge=0, le=1)
    has_dependents   : int   = Field(..., ge=0, le=1)

    # Contract
    tenure_months    : int   = Field(..., ge=0, le=72,
                                     description="Months as customer")
    contract         : ContractType
    paperless_billing: int   = Field(..., ge=0, le=1)
    payment_method   : PaymentMethod

    # Services
    internet_service : InternetService
    online_security  : YesNo
    tech_support     : YesNo
    streaming_tv     : YesNo

    # Billing
    monthly_charges  : float = Field(..., gt=0, le=200,
                                     description="Monthly bill in USD")
    total_charges    : float = Field(..., ge=0,
                                     description="Total billed in USD")

    @field_validator('total_charges')
    @classmethod
    def total_charges_reasonable(cls, v, info):
        """
        total_charges should be roughly
        tenure × monthly_charges.
        Allow ±50% for promotions and discounts.
        """
        data = info.data
        if 'tenure_months' in data and 'monthly_charges' in data:
            expected = (
                data['tenure_months'] *
                data['monthly_charges']
            )
            if expected > 0:
                ratio = v / expected
                if not (0.1 <= ratio <= 3.0):
                    raise ValueError(
                        f"total_charges {v} seems inconsistent "
                        f"with tenure × monthly_charges "
                        f"({expected:.2f}). "
                        f"Expected ratio between 0.1 and 3.0."
                    )
        return round(v, 2)

    model_config = {
        "json_schema_extra": {
            "example": {
                "age"              : 32,
                "gender"           : "Female",
                "senior_citizen"   : 0,
                "has_partner"      : 1,
                "has_dependents"   : 0,
                "tenure_months"    : 5,
                "contract"         : "Month-to-month",
                "paperless_billing": 1,
                "payment_method"   : "Electronic check",
                "internet_service" : "Fiber optic",
                "online_security"  : "No",
                "tech_support"     : "No",
                "streaming_tv"     : "Yes",
                "monthly_charges"  : 89.50,
                "total_charges"    : 447.50
            }
        }
    }


# ── Feature impact schema ─────────────────────────────────
class FeatureImpact(BaseModel):
    feature   : str
    shap_value: float
    direction : str


# ── Explanation schema ────────────────────────────────────
class PredictionExplanation(BaseModel):
    base_value   : float
    shap_sum     : float
    top_features : list[FeatureImpact]


# ── Single prediction response ────────────────────────────
class PredictionResponse(BaseModel):
    churn_probability : float = Field(..., ge=0.0, le=1.0)
    churn_label       : int   = Field(..., ge=0, le=1)
    risk_tier         : RiskTier
    threshold_used    : float
    model_version     : str
    explanation       : PredictionExplanation


# ── Batch response ────────────────────────────────────────
class BatchSummary(BaseModel):
    total_records    : int
    high_risk_count  : int
    medium_risk_count: int
    low_risk_count   : int
    avg_churn_prob   : float
    processing_time_s: float


class BatchResponse(BaseModel):
    batch_id        : str
    summary         : BatchSummary
    predictions     : list[dict]


# ── Health response ───────────────────────────────────────
class HealthResponse(BaseModel):
    status         : str
    model_version  : str
    model_loaded   : bool
    explainer_loaded: bool
    auc_roc        : float
    threshold      : float
    uptime_seconds : float


# ── Error response ────────────────────────────────────────
class ErrorResponse(BaseModel):
    error  : str
    detail : str
    code   : int