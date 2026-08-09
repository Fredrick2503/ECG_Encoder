"""
=========================================================
Biomarker Encoder Trainer

Handles:
1. Feature Engineering (cleaning + selection)
2. Class Imbalance (SMOTE, ADASYN)
3. Hyperparameter Tuning (Optuna, GridSearch, RandomizedSearch)
4. Training and OOF Stacking
5. Logging, Checkpointing, and automatic saving.

Author : ECG Intelligence System
=========================================================
"""

import random
from pathlib import Path
import json
import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from ml.preprocessing.feature_engineer import ECGFeatureEngineer

# Configure logging
logger = logging.getLogger("BiomarkerTrainer")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class BiomarkerTrainer:
    def __init__(
        self,
        encoder,
        save_directory=None,
        feature_engineer: ECGFeatureEngineer = None,
        oversample_method: str = None,  # "smote", "adasyn"
        random_state: int = 42
    ):
        self.encoder = encoder
        self.save_directory = Path(save_directory) if save_directory else None
        self.feature_engineer = feature_engineer
        self.oversample_method = oversample_method
        self.random_state = random_state

        # Enforce reproducibility
        random.seed(random_state)
        np.random.seed(random_state)

    def _oversample(self, X, y):
        """Oversample training split using Label Powerset and SMOTE/ADASYN."""
        if self.oversample_method is None:
            return X, y

        from collections import Counter
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)

        # Map label combinations to single integer IDs
        y_tuples = [tuple(row) for row in y_arr]
        unique_tuples = list(set(y_tuples))
        tuple_to_id = {t: i for i, t in enumerate(unique_tuples)}
        y_single = np.array([tuple_to_id[t] for t in y_tuples])

        try:
            from imblearn.over_sampling import SMOTE, ADASYN
            
            # Oversampling requires >= 2 samples per class.
            # Ensure at least 6 samples per class combination by duplicate sampling.
            counts = Counter(y_single)
            X_list = [X_arr]
            y_list = [y_arr]
            for class_id, count in counts.items():
                if count < 6:
                    needed = 6 - count
                    idx = np.where(y_single == class_id)[0]
                    dup_indices = np.random.choice(idx, size=needed, replace=True)
                    X_list.append(X_arr[dup_indices])
                    y_list.append(y_arr[dup_indices])

            X_extended = np.concatenate(X_list, axis=0)
            y_extended = np.concatenate(y_list, axis=0)

            # Re-encode tuples
            y_tuples = [tuple(row) for row in y_extended]
            unique_tuples = list(set(y_tuples))
            tuple_to_id = {t: i for i, t in enumerate(unique_tuples)}
            y_single_ext = np.array([tuple_to_id[t] for t in y_tuples])

            min_samples = min(Counter(y_single_ext).values())
            k_neighbors = min(5, min_samples - 1)

            if self.oversample_method.lower() == "smote":
                sampler = SMOTE(k_neighbors=k_neighbors, random_state=self.random_state)
            elif self.oversample_method.lower() == "adasyn":
                sampler = ADASYN(n_neighbors=k_neighbors, random_state=self.random_state)
            else:
                return X, y

            X_res, y_single_res = sampler.fit_resample(X_extended, y_single_ext)

            # Convert single targets back to multi-label
            id_to_tuple = {i: t for t, i in tuple_to_id.items()}
            y_res = np.array([id_to_tuple[val] for val in y_single_res])
            return X_res, y_res

        except ImportError:
            logger.warning("imbalanced-learn (imblearn) is not installed. Falling back to Random Oversampling.")
            
            counts = Counter(y_single)
            max_count = max(counts.values())
            
            X_res_list = []
            y_res_list = []
            
            for class_val in unique_tuples:
                cid = tuple_to_id[class_val]
                idx = np.where(y_single == cid)[0]
                resampled_idx = np.random.choice(idx, size=max_count, replace=True)
                X_res_list.append(X_arr[resampled_idx])
                y_res_list.append(y_arr[resampled_idx])
                
            return np.concatenate(X_res_list, axis=0), np.concatenate(y_res_list, axis=0)

    def tune_hyperparameters(self, X_train, y_train, method="optuna", n_trials=15):
        """Find optimal hyperparameters using GridSearchCV, RandomizedSearchCV, or Optuna."""
        logger.info(f"Tuning hyperparameters using {method}...")
        
        best_rf_params = {}
        best_xgb_params = {}

        try:
            if method.lower() != "optuna":
                raise ImportError("Force fallback to scikit-learn CV search.")
                
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            def objective(trial):
                rf_params = {
                    "n_estimators": trial.suggest_int("rf_n_estimators", 100, 300, step=50),
                    "max_depth": trial.suggest_int("rf_max_depth", 4, 10),
                    "min_samples_leaf": trial.suggest_int("rf_min_samples_leaf", 1, 4),
                    "random_state": self.random_state,
                    "n_jobs": -1
                }
                xgb_params = {
                    "n_estimators": trial.suggest_int("xgb_n_estimators", 100, 300, step=50),
                    "max_depth": trial.suggest_int("xgb_max_depth", 4, 8),
                    "learning_rate": trial.suggest_float("xgb_learning_rate", 0.01, 0.1),
                    "random_state": self.random_state,
                    "n_jobs": -1
                }

                from sklearn.model_selection import KFold
                kf = KFold(n_splits=3, shuffle=True, random_state=self.random_state)
                scores = []
                X_arr = np.asarray(X_train)
                y_arr = np.asarray(y_train)

                for train_idx, val_idx in kf.split(X_arr):
                    X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
                    y_tr, y_val = y_arr[train_idx], y_arr[val_idx]

                    if self.oversample_method:
                        X_tr, y_tr = self._oversample(X_tr, y_tr)

                    rf = MultiOutputClassifier(RandomForestClassifier(**rf_params))
                    rf.fit(X_tr, y_tr)
                    preds = rf.predict(X_val)
                    scores.append(f1_score(y_val, preds, average="macro"))

                return np.mean(scores)

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials)
            logger.info(f"Best tuning score: {study.best_value:.4f}")
            logger.info(f"Best parameters: {study.best_params}")

            for k, v in study.best_params.items():
                if k.startswith("rf_"):
                    best_rf_params[k.replace("rf_", "")] = v
                elif k.startswith("xgb_"):
                    best_xgb_params[k.replace("xgb_", "")] = v

        except Exception as e:
            logger.warning(f"Optuna tuning failed or not installed: {str(e)}. Falling back to RandomizedSearchCV.")
            from sklearn.model_selection import RandomizedSearchCV

            rf_grid = {
                "estimator__n_estimators": [100, 200, 300],
                "estimator__max_depth": [4, 6, 8, 10],
                "estimator__min_samples_leaf": [1, 2, 4]
            }
            rf = MultiOutputClassifier(RandomForestClassifier(random_state=self.random_state, n_jobs=-1))
            X_tr, y_tr = X_train, y_train
            if self.oversample_method:
                X_tr, y_tr = self._oversample(X_train, y_train)

            search = RandomizedSearchCV(rf, rf_grid, n_iter=min(5, n_trials), cv=3, random_state=self.random_state, scoring="f1_macro", n_jobs=-1)
            try:
                search.fit(X_tr, y_tr)
                for k, v in search.best_params_.items():
                    best_rf_params[k.replace("estimator__", "")] = v
            except Exception as ex:
                logger.warning(f"RandomForest fallback tuning failed: {str(ex)}")

            from ecg_pipeline.biomarker_encoder.models.mandala_encoder import CompatibleXGBClassifier
            xgb_grid = {
                "estimator__n_estimators": [100, 200, 300],
                "estimator__max_depth": [4, 6, 8],
                "estimator__learning_rate": [0.01, 0.05, 0.1]
            }
            xgb = MultiOutputClassifier(CompatibleXGBClassifier(random_state=self.random_state, n_jobs=-1))
            search_xgb = RandomizedSearchCV(xgb, xgb_grid, n_iter=min(5, n_trials), cv=3, random_state=self.random_state, scoring="f1_macro", n_jobs=-1)
            try:
                search_xgb.fit(X_tr, y_tr)
                for k, v in search_xgb.best_params_.items():
                    best_xgb_params[k.replace("estimator__", "")] = v
            except Exception as ex:
                logger.warning(f"XGBoost fallback tuning failed: {str(ex)}")

        # Apply parameters back to the encoder estimators
        if hasattr(self.encoder, "ensemble") and hasattr(self.encoder.ensemble, "estimator"):
            try:
                params_to_set = {}
                for k, v in best_rf_params.items():
                    params_to_set[f"rf__{k}"] = v
                for k, v in best_xgb_params.items():
                    params_to_set[f"xgb__{k}"] = v
                self.encoder.ensemble.estimator.set_params(**params_to_set)
                logger.info("Successfully applied tuned parameters to Mandala Encoder.")
            except Exception as ex:
                logger.warning(f"Could not apply parameters to Mandala: {str(ex)}")
        elif hasattr(self.encoder, "random_forest") and hasattr(self.encoder, "xgboost"):
            try:
                if best_rf_params:
                    self.encoder.random_forest.estimator.set_params(**best_rf_params)
                if best_xgb_params:
                    self.encoder.xgboost.estimator.set_params(**best_xgb_params)
                logger.info("Successfully applied tuned parameters to Robert Encoder.")
            except Exception as ex:
                logger.warning(f"Could not apply parameters to Robert: {str(ex)}")

        tuned_results = {}
        for k, v in best_rf_params.items():
            tuned_results[f"rf_{k}"] = v
        for k, v in best_xgb_params.items():
            tuned_results[f"xgb_{k}"] = v
        return tuned_results

    def train(self, X_train, y_train):
        logger.info(f"Training {self.encoder.name}...")

        # 1. Feature Engineering
        if self.feature_engineer:
            logger.info("Fitting Feature Engineer...")
            X_train = self.feature_engineer.fit_transform(X_train, y_train)

        # 2. Handle Class Imbalance (Oversampling training split only)
        if self.oversample_method:
            logger.info(f"Applying oversampling ({self.oversample_method}) on training split...")
            X_train, y_train = self._oversample(X_train, y_train)

        # 3. Fit Encoder
        self.encoder.train(X_train, y_train)
        logger.info("Encoder Training Complete.")
        return self.encoder

    def validate(self, X_validation):
        logger.info("Generating Validation Predictions...")
        if self.feature_engineer:
            X_validation = self.feature_engineer.transform(X_validation)

        predictions = self.encoder.predict(X_validation)
        probabilities = self.encoder.predict_proba(X_validation)
        return predictions, probabilities

    def test(self, X_test):
        logger.info("Generating Test Predictions...")
        if self.feature_engineer:
            X_test = self.feature_engineer.transform(X_test)

        predictions = self.encoder.predict(X_test)
        probabilities = self.encoder.predict_proba(X_test)
        return predictions, probabilities

    def save(self):
        if self.save_directory is None:
            raise ValueError("save_directory was not specified.")

        self.encoder.save(self.save_directory)
        
        # Save feature engineer if used
        if self.feature_engineer:
            import joblib
            joblib.dump(self.feature_engineer, self.save_directory / "feature_engineer.pkl")
            
        # Log trainer config/metadata checkpoint
        checkpoint = {
            "oversample_method": self.oversample_method,
            "random_state": self.random_state,
            "feature_engineer_applied": self.feature_engineer is not None
        }
        with open(self.save_directory / "trainer_config.json", "w") as f:
            json.dump(checkpoint, f, indent=4)

        logger.info(f"Checkpoint saved to: {self.save_directory}")

    def fit(self, X_train, y_train, X_validation=None, X_test=None):
        self.train(X_train, y_train)

        results = {}
        if X_validation is not None:
            val_pred, val_prob = self.validate(X_validation)
            results["validation_predictions"] = val_pred
            results["validation_probabilities"] = val_prob

        if X_test is not None:
            test_pred, test_prob = self.test(X_test)
            results["test_predictions"] = test_pred
            results["test_probabilities"] = test_prob

        if self.save_directory is not None:
            self.save()

        return results