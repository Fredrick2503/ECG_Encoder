"""
=========================================================
Biomarker Encoder Interface

Provides a single entry point for all Biomarker Encoders.

Supported Models
----------------
1. Robert Encoder
2. Mandala Encoder

Author : ECG Intelligence System
=========================================================
"""

from pathlib import Path

from biomarker_encoder.models.robert_encoder import RobertEncoder
from biomarker_encoder.models.mandala_encoder import MandalaEncoder


class BiomarkerEncoder:
    """
    Unified interface for Biomarker Encoders.

    Example
    -------
    >>> encoder = BiomarkerEncoder("robert")
    >>> encoder.train(X_train, y_train)
    >>> predictions = encoder.predict(X_test)
    >>> probabilities = encoder.predict_proba(X_test)
    >>> encoder.save("artifacts/robert")
    """

    SUPPORTED_MODELS = {
        "robert": RobertEncoder,
        "mandala": MandalaEncoder,
    }

    def __init__(self, model: str = "robert", **kwargs):

        model = model.lower()

        if model not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model '{model}'. "
                f"Available models: {list(self.SUPPORTED_MODELS.keys())}"
            )

        self.model_name = model
        self.encoder = self.SUPPORTED_MODELS[model](**kwargs)

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    def train(self, X_train, y_train):

        self.encoder.train(
            X_train,
            y_train
        )

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    def predict(self, X):

        return self.encoder.predict(X)

    # --------------------------------------------------
    # Probability Prediction
    # --------------------------------------------------

    def predict_proba(self, X):

        return self.encoder.predict_proba(X)

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    def save(self, directory):

        directory = Path(directory)

        self.encoder.save(directory)

    # --------------------------------------------------
    # Load
    # --------------------------------------------------

    def load(self, directory):

        directory = Path(directory)

        self.encoder.load(directory)

    # --------------------------------------------------
    # Model Access
    # --------------------------------------------------

    def get_models(self):

        return self.encoder.get_models()

    # --------------------------------------------------
    # Information
    # --------------------------------------------------

    @property
    def name(self):

        return self.encoder.name

    @property
    def model(self):

        return self.model_name

    def __repr__(self):

        return (
            f"BiomarkerEncoder("
            f"model='{self.model_name}', "
            f"name='{self.encoder.name}')"
        )