"""
=========================================================
Mandala Biomarker Encoder

Architecture
------------
47 Biomarkers
      │
 ┌────┼────┐
  │    │    │
  ▼    ▼    ▼
 RF   SVM  XGBoost
  │    │    │
  └────┼────┘
       ▼
 Soft Voting
      │
 Final Prediction

Author : ECG Intelligence System
=========================================================
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from .base_encoder import BaseEncoder

class CompatibleXGBClassifier(XGBClassifier):
    @property
    def _estimator_type(self):
        return "classifier"

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        return tags


class MandalaEncoder(BaseEncoder):
    """
    Mandala soft-voting ensemble with StandardScaler for SVM and configurable hyperparameters.
    """
    def __init__(
        self,
        random_state: int = 42,
        rf_params: dict = None,
        svm_params: dict = None,
        xgb_params: dict = None,
    ):
        super().__init__("Mandala Encoder")
        self.random_state = random_state

        # Resolve defaults
        if rf_params is None:
            rf_params = {"n_estimators": 300, "random_state": random_state, "n_jobs": -1}
        else:
            rf_params.setdefault("random_state", random_state)
            rf_params.setdefault("n_jobs", -1)

        if svm_params is None:
            svm_params = {"C": 1.0, "gamma": "scale", "probability": True, "kernel": "rbf", "random_state": random_state}
        else:
            svm_params.setdefault("probability", True)
            svm_params.setdefault("random_state", random_state)

        if xgb_params is None:
            xgb_params = {
                "n_estimators": 300,
                "max_depth": 6,
                "learning_rate": 0.05,
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "tree_method": "hist",
                "random_state": random_state,
                "n_jobs": -1
            }
        else:
            xgb_params.setdefault("random_state", random_state)
            xgb_params.setdefault("n_jobs", -1)

        # Scale features specifically for SVM to avoid scale sensitivity
        svm_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(**svm_params))
        ])

        base_voting = VotingClassifier(
            estimators=[
                (
                    "rf",
                    RandomForestClassifier(**rf_params),
                ),
                (
                    "svm",
                    svm_pipeline,
                ),
                (
                    "xgb",
                    CompatibleXGBClassifier(**xgb_params),
                ),
            ],
            voting="soft",
            n_jobs=-1,
        )

        self.ensemble = MultiOutputClassifier(base_voting)

    # ---------------------------------------------------
    # Training
    # ---------------------------------------------------

    def train(self, X_train, y_train):
        print("=" * 60)
        print("Training Mandala Soft Voting Ensemble")
        print("=" * 60)

        self.ensemble.fit(
            X_train,
            y_train,
        )

        print("Mandala Encoder Training Complete")

    # ---------------------------------------------------
    # Prediction
    # ---------------------------------------------------

    def predict(self, X):
        return self.ensemble.predict(X)

    # ---------------------------------------------------
    # Probability Prediction
    # ---------------------------------------------------

    def predict_proba(self, X):
        probabilities = self.ensemble.predict_proba(X)
        final_probabilities = []
        for probability in probabilities:
            final_probabilities.append(
                probability[:, 1]
            )

        return np.column_stack(
            final_probabilities
        )

    # ---------------------------------------------------
    # Models
    # ---------------------------------------------------

    def get_models(self):
        return {
            "mandala_ensemble": self.ensemble
        }

    # ---------------------------------------------------
    # Assign Loaded Models
    # ---------------------------------------------------

    def _assign_models(self, models):
        self.ensemble = models["mandala_ensemble"]