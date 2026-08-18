"""
============================================================
ECG Biomarker Encoder
Multi-label Confusion Matrix Generator
============================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import confusion_matrix

import numpy as np


class ECGConfusionMatrix:

    def __init__(self):

        self.results = {}

    def generate(
        self,
        y_true,
        y_pred,
        class_names,
    ):

        self.results = {}

        for index, label in enumerate(class_names):

            cm = confusion_matrix(
                y_true.iloc[:, index],
                y_pred.iloc[:, index],
            )

            self.results[label] = cm

        return self.results

    def display(self):

        if len(self.results) == 0:
            raise ValueError("Generate confusion matrices first.")

        for label, matrix in self.results.items():

            print("=" * 60)
            print(label)
            print("=" * 60)

            print(
                pd.DataFrame(
                    matrix,
                    index=["Actual Negative", "Actual Positive"],
                    columns=["Predicted Negative", "Predicted Positive"],
                )
            )

    def export_csv(
        self,
        output_directory="outputs/confusion_matrix",
    ):

        output_directory = Path(output_directory)

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        for label, matrix in self.results.items():

            dataframe = pd.DataFrame(
                matrix,
                index=["Actual Negative", "Actual Positive"],
                columns=["Predicted Negative", "Predicted Positive"],
            )

            dataframe.to_csv(
                output_directory / f"{label}.csv"
            )

        print(
            f"CSV files exported to:\n{output_directory.resolve()}"
        )

    def export_plots(
        self,
        output_directory="outputs/confusion_matrix",
        dpi=300,
    ):

        output_directory = Path(output_directory)

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        for label, matrix in self.results.items():

            plt.figure(figsize=(5,5))

            plt.imshow(
                matrix,
                interpolation="nearest",
            )

            plt.title(f"Confusion Matrix : {label}")

            plt.colorbar()

            ticks = np.arange(2)

            plt.xticks(
                ticks,
                [
                    "Negative",
                    "Positive",
                ],
            )

            plt.yticks(
                ticks,
                [
                    "Negative",
                    "Positive",
                ],
            )

            threshold = matrix.max() / 2

            for i in range(2):
                for j in range(2):

                    plt.text(
                        j,
                        i,
                        matrix[i, j],
                        ha="center",
                        color="white"
                        if matrix[i, j] > threshold
                        else "black",
                        fontsize=12,
                    )

            plt.ylabel("True Label")

            plt.xlabel("Predicted Label")

            plt.tight_layout()

            plt.savefig(
                output_directory / f"{label}.png",
                dpi=dpi,
            )

            plt.close()

        print(
            f"Plots exported to:\n{output_directory.resolve()}"
        )