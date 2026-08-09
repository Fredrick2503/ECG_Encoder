"""
=========================================================
Base Encoder

Abstract base class for all Biomarker Encoders.

Every encoder must implement a common interface so that
it can be used interchangeably through the BiomarkerEncoder
interface.

Author : ECG Intelligence System
=========================================================
"""

from abc import ABC, abstractmethod
from pathlib import Path
import joblib


class BaseEncoder(ABC):
    """
    Abstract Base Class for all Biomarker Encoders.
    """

    def __init__(self, name: str):
        self.name = name

    # -----------------------------------------------------
    # Required Methods
    # -----------------------------------------------------

    @abstractmethod
    def train(self, X_train, y_train):
        """
        Train the encoder.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Predict labels.
        """
        pass

    @abstractmethod
    def predict_proba(self, X):
        """
        Predict class probabilities.
        """
        pass

    @abstractmethod
    def get_models(self):
        """
        Returns a dictionary containing all trained models.

        Example
        -------
        {
            "random_forest": rf_model,
            "xgboost": xgb_model,
            "meta": meta_model
        }
        """
        pass

    # -----------------------------------------------------
    # Common Utilities
    # -----------------------------------------------------

    def save(self, directory):
        """
        Save all models returned by get_models().
        """

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        models = self.get_models()

        for name, model in models.items():
            joblib.dump(model, directory / f"{name}.pkl")

    def load(self, directory):
        """
        Load all models returned by get_models().

        Derived classes must create the model dictionary
        before calling this function.
        """

        directory = Path(directory)

        models = self.get_models()

        for name in models.keys():
            models[name] = joblib.load(directory / f"{name}.pkl")

        self._assign_models(models)

    @abstractmethod
    def _assign_models(self, models):
        """
        Assign loaded models back to encoder members.

        Example
        -------
        self.random_forest = models["random_forest"]
        self.xgboost = models["xgboost"]
        self.meta_model = models["meta"]
        """
        pass