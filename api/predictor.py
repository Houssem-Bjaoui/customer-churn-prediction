"""
Core prediction logic.
Loaded ONCE at API startup — never per request.
This prevents the 2-5 second latency of loading
the model on every call.
"""

import numpy as np
import pandas as pd
import joblib
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ChurnPredictor:
    """
    Encapsulates all model artifacts and prediction logic.
    Single instance shared across all API requests.
    """

    def __init__(self, models_path: Path):
        self.models_path  = models_path
        self.model        = None
        self.preprocessor = None
        self.explainer    = None
        self.metadata     = None
        self.feature_names     = None
        self.proc_feature_names = None
        self._loaded      = False

    def load(self) -> None:
        """
        Load all artifacts from disk.
        Called once during FastAPI startup event.
        """
        try:
            logger.info("Loading model artifacts...")

            self.preprocessor = joblib.load(
                self.models_path / 'preprocessor_v1.pkl'
            )
            logger.info("✓ Preprocessor loaded")

            self.model = joblib.load(
                self.models_path / 'xgboost_v1.pkl'
            )
            logger.info("✓ XGBoost model loaded")

            self.explainer = joblib.load(
                self.models_path / 'shap_explainer_v1.pkl'
            )
            logger.info("✓ SHAP explainer loaded")

            with open(
                self.models_path / 'model_metadata.json'
            ) as f:
                self.metadata = json.load(f)
            logger.info("✓ Metadata loaded")

            with open(
                self.models_path / 'proc_feature_names.json'
            ) as f:
                self.proc_feature_names = json.load(f)

            self._loaded = True
            logger.info("All artifacts loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load artifacts: {e}")
            raise

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def threshold(self) -> float:
        return self.metadata['optimal_threshold']

    @property
    def model_version(self) -> str:
        return self.metadata['model_name']

    def _build_feature_df(
        self, customer_data: dict
    ) -> pd.DataFrame:
        """
        Convert raw customer dict to feature DataFrame.
        Applies the same feature engineering as the
        training notebooks — must stay in sync.
        """
        df = pd.DataFrame([customer_data])

        # ── Engineered features ───────────────────────
        # Feature 1: is_new_customer
        df['is_new_customer'] = (
            df['tenure_months'] <= 6
        ).astype(int)

        # Feature 2: contract_risk_score
        contract_map = {
            'Month-to-month': 2,
            'One year'      : 1,
            'Two year'      : 0
        }
        df['contract_risk_score'] = (
            df['contract'].map(contract_map).astype(int)
        )

        # Feature 3: num_services
        df['num_services'] = (
            (df['online_security'] == 'Yes').astype(int) +
            (df['tech_support'] == 'Yes').astype(int) +
            (df['streaming_tv'] == 'Yes').astype(int)
        )

        # Feature 4: has_fiber_no_support
        df['has_fiber_no_support'] = (
            (df['internet_service'] == 'Fiber optic') &
            (df['tech_support'] == 'No')
        ).astype(int)

        # Feature 5: has_fiber_with_support
        df['has_fiber_with_support'] = (
            (df['internet_service'] == 'Fiber optic') &
            (df['tech_support'] == 'Yes')
        ).astype(int)

        # Feature 6: charges_per_month
        df['charges_per_month'] = (
            df['total_charges'] /
            df['tenure_months'].replace(0, 1)
        ).round(2)

        # Feature 7: service_value_ratio
        df['service_value_ratio'] = (
            df['monthly_charges'] /
            (df['num_services'] + 1)
        ).round(2)

        # Feature 8: senior_alone
        df['senior_alone'] = (
            (df['senior_citizen'] == 1) &
            (df['has_partner'] == 0) &
            (df['has_dependents'] == 0)
        ).astype(int)

        # Feature 9: payment_risk_score
        payment_map = {
            'Electronic check': 3,
            'Mailed check'    : 2,
            'Bank transfer'   : 1,
            'Credit card'     : 0
        }
        df['payment_risk_score'] = (
            df['payment_method']
            .map(payment_map)
            .astype(int)
        )

        # Feature 10: tenure_contract_risk
        df['tenure_contract_risk'] = (
            df['is_new_customer'] *
            df['contract_risk_score']
        )

        # One-hot encode categoricals
        cat_cols = [
            'gender', 'contract', 'internet_service',
            'online_security', 'tech_support',
            'streaming_tv', 'payment_method'
        ]
        df = pd.get_dummies(
            df, columns=cat_cols,
            drop_first=False, dtype=int
        )

        return df

    def _get_risk_tier(self, probability: float) -> str:
        if probability >= 0.65:
            return "HIGH"
        elif probability >= 0.35:
            return "MEDIUM"
        return "LOW"

    def _explain(
        self,
        processed_data: np.ndarray,
        top_n: int = 5
    ) -> dict:
        """
        Generate SHAP explanation for one preprocessed row.
        """
        shap_vals = self.explainer.shap_values(processed_data)

        if len(np.array(shap_vals).shape) > 1:
            shap_vals = shap_vals[0]

        impacts = [
            {
                'feature'   : self.proc_feature_names[i],
                'shap_value': round(float(shap_vals[i]), 4),
                'direction' : (
                    'increases_churn_risk'
                    if shap_vals[i] > 0
                    else 'decreases_churn_risk'
                ),
                '_abs'      : abs(float(shap_vals[i]))
            }
            for i in range(
                min(len(shap_vals),
                    len(self.proc_feature_names))
            )
        ]

        impacts.sort(key=lambda x: x['_abs'], reverse=True)
        for item in impacts:
            del item['_abs']

        return {
            'base_value'  : round(
                float(self.explainer.expected_value), 4
            ),
            'shap_sum'    : round(
                float(sum(
                    self.explainer.shap_values(
                        processed_data
                    ).flatten()
                )), 4
            ),
            'top_features': impacts[:top_n]
        }

    def predict_single(
        self,
        customer_data: dict,
        top_n_shap: int = 5
    ) -> dict:
        """
        Full prediction pipeline for one customer.
        Builds features → preprocesses → predicts → explains.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        # Build feature DataFrame
        df_features = self._build_feature_df(customer_data)

        # Align columns with training data
        # Add missing columns as 0
        # This handles unseen one-hot categories gracefully
        train_cols = self.proc_feature_names
        for col in train_cols:
            if col not in df_features.columns:
                df_features[col] = 0

        # Keep only known columns in correct order
        available = [
            c for c in train_cols
            if c in df_features.columns
        ]
        df_aligned = df_features[available]

        # Preprocess
        X_processed = self.preprocessor.transform(df_aligned)

        # Predict
        probability = float(
            self.model.predict_proba(X_processed)[0, 1]
        )
        label = int(probability >= self.threshold)

        # Explain
        explanation = self._explain(X_processed, top_n_shap)

        return {
            'churn_probability': round(probability, 4),
            'churn_label'      : label,
            'risk_tier'        : self._get_risk_tier(probability),
            'threshold_used'   : self.threshold,
            'model_version'    : self.model_version,
            'explanation'      : explanation
        }

    def predict_batch(
        self,
        records: list[dict]
    ) -> list[dict]:
        """
        Batch prediction for multiple customers.
        More efficient than calling predict_single in a loop.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        results = []
        for i, record in enumerate(records):
            try:
                result = self.predict_single(record)
                result['record_index'] = i
                result['status'] = 'success'
                results.append(result)
            except Exception as e:
                results.append({
                    'record_index': i,
                    'status'      : 'error',
                    'error'       : str(e)
                })

        return results