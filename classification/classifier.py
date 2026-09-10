import torch
import torch.nn as nn
from torch.utils.data import Dataset

class ZFusedDataset(Dataset):
    """
    Dataset to load extracted fused representations and labels.
    """
    def __init__(self, z_data, labels, record_ids=None, patient_ids=None):
        self.z_data = torch.tensor(z_data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.record_ids = record_ids
        self.patient_ids = patient_ids

    def __len__(self):
        return len(self.z_data)

    def __getitem__(self, idx):
        sample = {
            "z": self.z_data[idx],
            "label": self.labels[idx]
        }
        if self.record_ids is not None:
            sample["record_id"] = self.record_ids[idx]
        if self.patient_ids is not None:
            sample["patient_id"] = self.patient_ids[idx]
        return sample

class LinearProbeClassifier(nn.Module):
    """
    Linear Probe Classifier (C0 baseline)
    """
    def __init__(self, input_dim=1056, num_classes=5):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

class MLPClassifier(nn.Module):
    """
    Non-linear Multi-Layer Perceptron Classifier (C1 model)
    """
    def __init__(self, input_dim=1056, hidden_dim=256, num_classes=5, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.net(x)
