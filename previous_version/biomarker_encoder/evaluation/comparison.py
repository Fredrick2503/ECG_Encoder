"""
=========================================================
Biomarker Encoder Comparison

Compares multiple Biomarker Encoders using a common
evaluation pipeline.

Author : ECG Intelligence System
=========================================================
"""

import pandas as pd

from biomarker_encoder.evaluation.metrics import BiomarkerMetrics


class BiomarkerComparison:
    """
    Compare multiple Biomarker Encoders.

    Example
    -------
    comparison = BiomarkerComparison()

    comparison.add_result(
        "Robert",
        y_test,
        robert_predictions,
        robert_probabilities
    )

    comparison.add_result(
        "Mandala",
        y_test,
        mandala_predictions,
        mandala_probabilities
    )

    results = comparison.summary()
    """

    def __init__(self, threshold: float = 0.5):

        self.metric_engine = BiomarkerMetrics(
            threshold=threshold
        )

        self.results = []

    # --------------------------------------------------
    # Add Encoder Result
    # --------------------------------------------------

    def add_result(
        self,
        model_name,
        y_true,
        predictions,
        probabilities=None,
    ):

        metrics = self.metric_engine.evaluate(
            y_true=y_true,
            predictions=predictions,
            probabilities=probabilities,
        )

        metrics.insert(
            0,
            "Model",
            model_name,
        )

        self.results.append(metrics)

    # --------------------------------------------------
    # Summary Table
    # --------------------------------------------------

    def summary(self):

        if len(self.results) == 0:
            raise ValueError(
                "No comparison results available."
            )

        table = pd.concat(
            self.results,
            ignore_index=True,
        )

        return table

    # --------------------------------------------------
    # Print Summary
    # --------------------------------------------------

    def print_summary(self):

        table = self.summary()

        print("\n")
        print("=" * 100)
        print("BIOMARKER ENCODER COMPARISON")
        print("=" * 100)

        print(table.to_string(index=False))

        print("=" * 100)

    # --------------------------------------------------
    # Best Model
    # --------------------------------------------------

    def best_model(
        self,
        metric="F1 Macro",
        maximize=True,
    ):

        table = self.summary()

        if metric not in table.columns:
            raise ValueError(
                f"{metric} not found."
            )

        if maximize:

            idx = table[metric].idxmax()

        else:

            idx = table[metric].idxmin()

        return table.loc[idx]

    # --------------------------------------------------
    # Export
    # --------------------------------------------------

    def export_csv(
        self,
        filepath,
    ):

        table = self.summary()

        table.to_csv(
            filepath,
            index=False,
        )

        print(
            f"Comparison exported to {filepath}"
        )

        return table