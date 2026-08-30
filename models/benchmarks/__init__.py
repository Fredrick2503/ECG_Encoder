"""
Benchmark model architectures for MIT-BIH, MIT-BIH AF, and PTB-XL/CinC.
"""

from models.benchmarks.cnn_lstm import CNNLSTM
from models.benchmarks.ecg_former import ECGFormer
from models.benchmarks.cnn_transformer import CNNTransformer
from models.benchmarks.hybrid_bert_cnn import HybridBERTCNN
from models.benchmarks.foundational_ecgnet import FoundationalECGNet
from models.benchmarks.rr_af_detector import RRAFDetector

__all__ = [
    "CNNLSTM",
    "ECGFormer",
    "CNNTransformer",
    "HybridBERTCNN",
    "FoundationalECGNet",
    "RRAFDetector",
]
