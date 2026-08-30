import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import unittest
import torch
from models.benchmarks import (
    CNNLSTM,
    ECGFormer,
    CNNTransformer,
    HybridBERTCNN,
    FoundationalECGNet,
    RRAFDetector
)


def test_cnn_lstm_forward():
    model = CNNLSTM(in_channels=1, num_classes=5)
    x = torch.randn(4, 1, 280)
    out = model(x)
    assert out.shape == (4, 5)


def test_ecg_former_forward():
    model = ECGFormer(in_channels=1, num_classes=5, patch_size=14, d_model=64, nhead=4, num_layers=2)
    x = torch.randn(4, 1, 280)
    out = model(x)
    assert out.shape == (4, 5)


def test_cnn_transformer_forward():
    model = CNNTransformer(in_channels=1, num_classes=5, stem_channels=(16, 32, 64), d_model=64, nhead=4, num_layers=2)
    x = torch.randn(4, 1, 280)
    out = model(x)
    assert out.shape == (4, 5)


def test_hybrid_bert_cnn_forward():
    model = HybridBERTCNN(in_channels=1, num_classes=5, cnn_hidden=32, d_model=64, nhead=4, num_layers=2)
    x = torch.randn(4, 1, 280)
    out = model(x)
    assert out.shape == (4, 5)


def test_foundational_ecgnet_forward():
    model = FoundationalECGNet(in_channels=12, num_classes=5, base_channels=32, d_model=64, nhead=4, num_trans_layers=2)
    x = torch.randn(4, 12, 1000)
    out = model(x)
    assert out.shape == (4, 5)


def test_rr_af_detector_forward():
    model = RRAFDetector(in_dim=2, hidden_dim=32, num_layers=2, num_classes=2)
    x = torch.randn(4, 50, 2)
    out = model(x)
    assert out.shape == (4, 2)


if __name__ == "__main__":
    test_cnn_lstm_forward()
    test_ecg_former_forward()
    test_cnn_transformer_forward()
    test_hybrid_bert_cnn_forward()
    test_foundational_ecgnet_forward()
    test_rr_af_detector_forward()
    print("All 6 benchmark model forward tests passed successfully!")
