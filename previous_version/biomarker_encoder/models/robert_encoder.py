"""
=========================================================
Robert Biomarker Encoder

Architecture
------------
47 Biomarkers
      │
      ├── Random Forest
      │
      └── XGBoost
            │
      Probability Fusion (Meta Features)
            │
      Meta XGBoost
            │
      Final Prediction

Author : ECG Intelligence System
=========================================================
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import KFold
from sklearn.base import clone
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


class RobertEncoder(BaseEncoder):
    """
    Robert Encoder featuring proper Out-of-Fold (OOF) stacking to prevent data leakage.
    """
    def __init__(
        self,
        random_state: int = 42,
        rf_params: dict = None,
        xgb_params: dict = None,
        meta_params: dict = None
    ):
        super().__init__("Robert Encoder")
        self.random_state = random_state

        # Resolve defaults
        if rf_params is None:
            rf_params = {"n_estimators": 300, "random_state": random_state, "n_jobs": -1}
        else:
            rf_params.setdefault("random_state", random_state)
            rf_params.setdefault("n_jobs", -1)

        if xgb_params is None:
            xgb_params = {
                "n_estimators": 300,
                "max_depth": 6,
                "learning_rate": 0.05,
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "random_state": random_state,
                "tree_method": "hist",
                "n_jobs": -1
            }
        else:
            xgb_params.setdefault("random_state", random_state)
            xgb_params.setdefault("n_jobs", -1)

        if meta_params is None:
            meta_params = {
                "n_estimators": 200,
                "max_depth": 4,
                "learning_rate": 0.03,
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "random_state": random_state,
                "tree_method": "hist",
                "n_jobs": -1
            }
        else:
            meta_params.setdefault("random_state", random_state)
            meta_params.setdefault("n_jobs", -1)

        self.random_forest = MultiOutputClassifier(RandomForestClassifier(**rf_params))
        self.xgboost = MultiOutputClassifier(CompatibleXGBClassifier(**xgb_params))
        self.meta_model = MultiOutputClassifier(CompatibleXGBClassifier(**meta_params))

    # ---------------------------------------------------
    # Internal Utility
    # ---------------------------------------------------

    def _collect_probabilities(self, model, X):
        probabilities = []
        outputs = model.predict_proba(X)
        for prob in outputs:
            probabilities.append(prob[:, 1])
        return np.column_stack(probabilities)

    # ---------------------------------------------------
    # Training (Out-of-Fold Stacking)
    # ---------------------------------------------------

    def train(self, X_train, y_train):
        print("=" * 60)
        print("Training Robert Encoder using Out-of-Fold Stacking")
        print("=" * 60)

        # Convert to numpy arrays for reliable indexing
        X_arr = np.asarray(X_train)
        y_arr = np.asarray(y_train)
        
        num_samples = X_arr.shape[0]
        num_classes = y_arr.shape[1]

        # Folds definition
        n_splits = 5
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        # Arrays to hold out-of-fold meta features
        oof_rf = np.zeros((num_samples, num_classes))
        oof_xgb = np.zeros((num_samples, num_classes))

        print(f"Generating OOF predictions with {n_splits}-fold CV...")
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_arr), 1):
            X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
            y_tr, y_val = y_arr[train_idx], y_arr[val_idx]

            # Clone models to prevent fitting leakage
            rf_clone = clone(self.random_forest)
            xgb_clone = clone(self.xgboost)

            rf_clone.fit(X_tr, y_tr)
            xgb_clone.fit(X_tr, y_tr)

            oof_rf[val_idx] = self._collect_probabilities(rf_clone, X_val)
            oof_xgb[val_idx] = self._collect_probabilities(xgb_clone, X_val)
            print(f"  Fold {fold} finished.")

        # Combine OOF features
        meta_features = np.concatenate([oof_rf, oof_xgb], axis=1)

        print("Training Meta XGBoost Model on OOF predictions...")
        self.meta_model.fit(meta_features, y_arr)

        print("Fitting final base estimators on all training data...")
        self.random_forest.fit(X_arr, y_arr)
        self.xgboost.fit(X_arr, y_arr)

        print("Robert Encoder Stacking Training Complete")

    # ---------------------------------------------------
    # Prediction
    # ---------------------------------------------------

    def predict(self, X):
        rf_prob = self._collect_probabilities(self.random_forest, X)
        xgb_prob = self._collect_probabilities(self.xgboost, X)
        meta_features = np.concatenate([rf_prob, xgb_prob], axis=1)
        return self.meta_model.predict(meta_features)

    # ---------------------------------------------------
    # Probability Prediction
    # ---------------------------------------------------

    def predict_proba(self, X):
        rf_prob = self._collect_probabilities(self.random_forest, X)
        xgb_prob = self._collect_probabilities(self.xgboost, X)
        meta_features = np.concatenate([rf_prob, xgb_prob], axis=1)
        probabilities = self.meta_model.predict_proba(meta_features)
        
        final_prob = []
        for prob in probabilities:
            final_prob.append(prob[:, 1])
        return np.column_stack(final_prob)

    # ---------------------------------------------------
    # Model Dictionary
    # ---------------------------------------------------

    def get_models(self):
        return {
            "random_forest": self.random_forest,
            "xgboost": self.xgboost,
            "meta_model": self.meta_model
        }

    # ---------------------------------------------------
    # Load Utility
    # ---------------------------------------------------

    def _assign_models(self, models):
        self.random_forest = models["random_forest"]
        self.xgboost = models["xgboost"]
        self.meta_model = models["meta_model"]