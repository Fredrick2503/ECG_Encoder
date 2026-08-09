"""
=========================================================
ECG Stacking Ensemble Visualizer

Generates and saves research-grade clinical visualizations:
1. Confusion Matrices
2. ROC Curves
3. Precision-Recall Curves
4. Feature Importance
5. SHAP Summary Plots
6. Calibration Curves

Author : ECG Intelligence System
=========================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import logging
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
from sklearn.calibration import calibration_curve

logger = logging.getLogger("ECGVisualizer")


class ECGVisualizer:
    def __init__(self, save_dir: str, class_names: list = None):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.class_names = class_names

    def _get_class_name(self, index: int) -> str:
        if self.class_names and index < len(self.class_names):
            return self.class_names[index]
        return f"Class_{index}"

    def plot_confusion_matrices(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Generates and saves a confusion matrix heatmap for each target class using pure matplotlib."""
        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)
        num_classes = y_true_arr.shape[1]

        fig, axes = plt.subplots(1, num_classes, figsize=(4 * num_classes, 4))
        if num_classes == 1:
            axes = [axes]

        for c in range(num_classes):
            cm = confusion_matrix(y_true_arr[:, c], y_pred_arr[:, c])
            ax = axes[c]
            im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
            
            # Add text annotations
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(
                        j, i, format(cm[i, j], "d"),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black"
                    )
            
            ax.set_title(f"CM: {self._get_class_name(c)}")
            ax.set_ylabel("True label")
            ax.set_xlabel("Predicted label")
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Negative", "Positive"])
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["Negative", "Positive"])

        plt.tight_layout()
        save_path = self.save_dir / "confusion_matrices.png"
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
        logger.info(f"Saved confusion matrices to {save_path}")

    def plot_roc_curves(self, y_true: np.ndarray, probabilities: np.ndarray):
        """Generates and saves a combined ROC plot showing curves and AUC for all classes."""
        y_true_arr = np.asarray(y_true)
        probs_arr = np.asarray(probabilities)
        num_classes = y_true_arr.shape[1]

        plt.figure(figsize=(8, 6))
        
        for c in range(num_classes):
            fpr, tpr, _ = roc_curve(y_true_arr[:, c], probs_arr[:, c])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{self._get_class_name(c)} (AUC = {roc_auc:.3f})")

        # Plot micro-average
        try:
            fpr_micro, tpr_micro, _ = roc_curve(y_true_arr.ravel(), probs_arr.ravel())
            auc_micro = auc(fpr_micro, tpr_micro)
            plt.plot(fpr_micro, tpr_micro, label=f"Micro-average (AUC = {auc_micro:.3f})", linestyle="--", color="black")
        except Exception:
            pass

        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.7)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curves")
        plt.legend(loc="lower right")
        plt.grid(True, linestyle="--", alpha=0.5)

        save_path = self.save_dir / "roc_curves.png"
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
        logger.info(f"Saved ROC curves to {save_path}")

    def plot_precision_recall_curves(self, y_true: np.ndarray, probabilities: np.ndarray):
        """Generates and saves a combined Precision-Recall plot for all classes."""
        y_true_arr = np.asarray(y_true)
        probs_arr = np.asarray(probabilities)
        num_classes = y_true_arr.shape[1]

        plt.figure(figsize=(8, 6))
        
        for c in range(num_classes):
            precision, recall, _ = precision_recall_curve(y_true_arr[:, c], probs_arr[:, c])
            pr_auc = auc(recall, precision)
            plt.plot(recall, precision, label=f"{self._get_class_name(c)} (PR-AUC = {pr_auc:.3f})")

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall (PR) Curves")
        plt.legend(loc="lower left")
        plt.grid(True, linestyle="--", alpha=0.5)

        save_path = self.save_dir / "precision_recall_curves.png"
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
        logger.info(f"Saved Precision-Recall curves to {save_path}")

    def plot_feature_importance(self, model, feature_names: list, top_n: int = 15):
        """Plots the feature importances of Random Forest or XGBoost models using pure matplotlib."""
        try:
            if hasattr(model, "estimators_"):
                importances = np.mean([
                    getattr(est, "feature_importances_", np.zeros(len(feature_names)))
                    for est in model.estimators_
                ], axis=0)
            else:
                importances = getattr(model, "feature_importances_", None)

            if importances is None:
                logger.warning("Estimator does not support feature_importances_.")
                return

            indices = np.argsort(importances)[::-1][:top_n]
            top_features = [feature_names[i] for i in indices]
            top_importances = importances[indices]

            plt.figure(figsize=(10, 6))
            plt.barh(top_features[::-1], top_importances[::-1], color="skyblue")
            plt.title(f"Top {top_n} Feature Importances")
            plt.xlabel("Importance Score")
            plt.ylabel("Features")

            save_path = self.save_dir / "feature_importance.png"
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            plt.close()
            logger.info(f"Saved feature importance plot to {save_path}")
        except Exception as e:
            logger.error(f"Failed to plot feature importance: {str(e)}")

    def plot_calibration_curves(self, y_true: np.ndarray, probabilities: np.ndarray):
        """Generates and saves probability calibration curves for all classes."""
        y_true_arr = np.asarray(y_true)
        probs_arr = np.asarray(probabilities)
        num_classes = y_true_arr.shape[1]

        plt.figure(figsize=(8, 6))
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")

        for c in range(num_classes):
            try:
                prob_true, prob_pred = calibration_curve(y_true_arr[:, c], probs_arr[:, c], n_bins=5)
                plt.plot(prob_pred, prob_true, marker="o", label=self._get_class_name(c))
            except Exception as e:
                logger.warning(f"Calibration curve failed for class {c}: {str(e)}")

        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.title("Calibration Curves (Reliability Diagrams)")
        plt.legend(loc="upper left")
        plt.grid(True, linestyle="--", alpha=0.5)

        save_path = self.save_dir / "calibration_curves.png"
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
        logger.info(f"Saved calibration curves to {save_path}")

    def plot_shap_summary(self, model, X_train: pd.DataFrame):
        """Generates and saves a SHAP summary plot if shap is installed."""
        try:
            import shap
            if hasattr(model, "estimators_"):
                est = model.estimators_[0]
            else:
                est = model
                
            explainer = shap.TreeExplainer(est)
            X_sample = X_train.sample(min(100, len(X_train)), random_state=42) if len(X_train) > 100 else X_train
            shap_values = explainer.shap_values(X_sample)

            plt.figure(figsize=(10, 6))
            if isinstance(shap_values, list) and len(shap_values) > 1:
                shap.summary_plot(shap_values[1], X_sample, show=False)
            else:
                shap.summary_plot(shap_values, X_sample, show=False)

            save_path = self.save_dir / "shap_summary_plot.png"
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            plt.close()
            logger.info(f"Saved SHAP summary plot to {save_path}")
        except ImportError:
            logger.warning("SHAP package not installed. Skipping SHAP summary plot.")
        except Exception as e:
            logger.error(f"Failed to generate SHAP plot: {str(e)}")
