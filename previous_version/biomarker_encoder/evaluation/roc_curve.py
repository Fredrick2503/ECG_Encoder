"""
============================================================
ECG Biomarker Encoder
ROC Curve Evaluation
============================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_curve,
    auc,
)


class ECGROCCurve:

    def __init__(self):

        self.roc_results = {}

    def generate(
        self,
        y_true,
        y_probability,
        class_names,
    ):

        if isinstance(y_true, pd.DataFrame):
            y_true = y_true.to_numpy()

        if isinstance(y_probability, pd.DataFrame):
            y_probability = y_probability.to_numpy()

        y_true = np.asarray(y_true)
        y_probability = np.asarray(y_probability)

        self.roc_results = {}

        for index, label in enumerate(class_names):

            fpr, tpr, thresholds = roc_curve(
                y_true[:, index],
                y_probability[:, index],
            )

            roc_auc = auc(
                fpr,
                tpr,
            )

            self.roc_results[label] = {
                "fpr": fpr,
                "tpr": tpr,
                "thresholds": thresholds,
                "auc": roc_auc,
            }

        return self.roc_results

    def summary(self):

        rows = []

        for label, values in self.roc_results.items():

            rows.append(
                {
                    "Class": label,
                    "ROC AUC": values["auc"],
                }
            )

        return pd.DataFrame(rows)

    def display(self):

        table = self.summary()

        print("=" * 60)
        print("ROC AUC Summary")
        print("=" * 60)

        print(table)

        return table

    def export_csv(
        self,
        output_directory="outputs/roc_curve",
    ):

        output_directory = Path(output_directory)

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        summary = self.summary()

        summary.to_csv(
            output_directory / "roc_auc_summary.csv",
            index=False,
        )

        for label, values in self.roc_results.items():

            dataframe = pd.DataFrame(
                {
                    "False Positive Rate": values["fpr"],
                    "True Positive Rate": values["tpr"],
                    "Threshold": values["thresholds"],
                }
            )

            dataframe.to_csv(
                output_directory / f"{label}_roc.csv",
                index=False,
            )

        print(
            f"ROC CSV exported to:\n{output_directory.resolve()}"
        )

    def export_plots(
        self,
        output_directory="outputs/roc_curve",
        dpi=600,
    ):

        output_directory = Path(output_directory)

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        for label, values in self.roc_results.items():

            plt.figure(figsize=(6,6))

            plt.plot(
                values["fpr"],
                values["tpr"],
                linewidth=2,
                label=f"AUC = {values['auc']:.4f}",
            )

            plt.plot(
                [0,1],
                [0,1],
                linestyle="--",
                linewidth=1,
            )

            plt.xlim([0,1])
            plt.ylim([0,1.05])

            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")

            plt.title(f"ROC Curve - {label}")

            plt.legend(loc="lower right")

            plt.grid(True)

            plt.tight_layout()

            plt.savefig(
                output_directory / f"{label}_roc.png",
                dpi=dpi,
                bbox_inches="tight",
            )

            plt.close()

        print(
            f"ROC plots exported to:\n{output_directory.resolve()}"
        )