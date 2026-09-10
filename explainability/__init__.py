"""
Explainability (XAI) Package
============================
Integrates Integrated Gradients, Grad-CAM, and Waveform Landmark Translation
for 12-Lead ECG Representation Learning Models.
"""

from explainability.manager import XAIManager
from explainability.temporal_xai import TemporalExplainer
from explainability.morphology_xai import MorphologyExplainer
from explainability.translator import GradCAMTranslator
from explainability.visualizer import (
    plot_1d_attribution,
    plot_2d_attribution,
    plot_translated_attribution,
    normalize_attribution,
)

__all__ = [
    "XAIManager",
    "TemporalExplainer",
    "MorphologyExplainer",
    "GradCAMTranslator",
    "plot_1d_attribution",
    "plot_2d_attribution",
    "plot_translated_attribution",
    "normalize_attribution",
]
