import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import numpy as np
import pandas as pd
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from data_management.ecg_record import ECGRecord
from data_management.label_encoder import PTBXLLabelEncoder
from data_management.splitter import PTBXLFoldSplitter
from data_management.metadata import PTBXLMetadataParser
from data_management.loader import PTBXLLoader
try:
    from data_management.sample_builder import ECGDataset
    from data_management.dataset_factory import DatasetFactory
except ImportError:
    ECGDataset = None
    DatasetFactory = None

class TestECGRecord(unittest.TestCase):
    def test_creation(self):
        signal = np.zeros((12, 500), dtype=np.float32)
        leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        labels = np.array([1, 0, 0, 0, 0], dtype=np.float32)
        
        record = ECGRecord(
            record_id="123",
            signal=signal,
            sampling_rate=500,
            leads=leads,
            labels=labels,
            age=45.0,
            sex="M"
        )
        
        self.assertEqual(record.record_id, "123")
        self.assertEqual(record.num_leads, 12)
        self.assertEqual(record.signal_length, 500)
        self.assertEqual(record.age, 45.0)
        self.assertEqual(record.sex, "M")
        np.testing.assert_array_equal(record.get_lead_signal("I"), signal[0])


class TestPTBXLLabelEncoder(unittest.TestCase):
    def setUp(self):
        self.classes = ["NORM", "MI", "STTC", "CD", "HYP"]
        self.encoder = PTBXLLabelEncoder(self.classes)

    def test_encode(self):
        encoded = self.encoder.encode(["NORM", "MI"])
        np.testing.assert_array_equal(encoded, np.array([1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32))

    def test_decode(self):
        # Index 0 is NORM, Index 3 is CD
        decoded = self.encoder.decode(np.array([0.9, 0.1, 0.0, 0.8, 0.0], dtype=np.float32))
        self.assertEqual(decoded, ["NORM", "CD"])


class TestPTBXLFoldSplitter(unittest.TestCase):
    def test_split(self):
        df = pd.DataFrame({
            "strat_fold": [1, 2, 9, 10, 1, 9, 10, 10]
        })
        splitter = PTBXLFoldSplitter(train_folds=[1, 2], val_folds=[9], test_folds=[10])
        train_df, val_df, test_df = splitter.split(df)
        
        self.assertEqual(len(train_df), 3)
        self.assertEqual(len(val_df), 2)
        self.assertEqual(len(test_df), 3)


class TestPTBXLLoaderAndParser(unittest.TestCase):
    @patch("pathlib.Path.exists")
    @patch("pandas.read_csv")
    def test_parser_load_scp(self, mock_read_csv, mock_exists):
        mock_exists.return_value = True
        mock_scp_df = pd.DataFrame({
            "diagnostic_class": ["MI", "NORM"],
            "subclass": ["AMI", "NORM_SUB"]
        }, index=["AMI_CODE", "NORM_CODE"])
        mock_read_csv.return_value = mock_scp_df
        
        parser = PTBXLMetadataParser(Path("dummy_db.csv"), Path("dummy_scp.csv"))
        parser.load_scp_statements()
        
        self.assertEqual(parser._code_to_class["AMI_CODE"], "MI")
        self.assertEqual(parser._code_to_subclass["AMI_CODE"], "AMI")

    @patch("wfdb.rdsamp")
    @patch("pathlib.Path.exists")
    @patch("pandas.read_csv")
    def test_loader_load_record(self, mock_read_csv, mock_exists, mock_rdsamp):
        mock_exists.return_value = True
        # Setup mock metadata
        mock_db_df = pd.DataFrame({
            "patient_id": [10001],
            "age": [50.0],
            "sex": ["M"],
            "filename_hr": ["records500/00000/00001_hr"],
            "filename_lr": ["records100/00000/00001_lr"],
            "scp_codes": ["{'AMI_CODE': 100.0}"],
            "strat_fold": [1]
        }, index=[1])
        
        mock_scp_df = pd.DataFrame({
            "diagnostic_class": ["MI"],
            "subclass": ["AMI"]
        }, index=["AMI_CODE"])
        
        # side_effect to return mock database first, then mock scp
        mock_read_csv.side_effect = [mock_scp_df, mock_db_df]
        
        # Setup mock wfdb signal
        mock_signal = np.ones((5000, 12), dtype=np.float32)
        mock_meta = {"sig_name": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]}
        mock_rdsamp.return_value = (mock_signal, mock_meta)
        
        encoder = PTBXLLabelEncoder(["NORM", "MI", "STTC", "CD", "HYP"])
        loader = PTBXLLoader(
            root_dir=Path("dummy_dir"),
            database_csv=Path("dummy_db.csv"),
            scp_csv=Path("dummy_scp.csv"),
            resolution="hr",
            label_encoder=encoder
        )
        
        record = loader.load_record(1)
        
        self.assertEqual(record.record_id, "1")
        self.assertEqual(record.sampling_rate, 500)
        self.assertEqual(record.age, 50.0)
        self.assertEqual(record.sex, "M")
        self.assertEqual(record.num_leads, 12)
        # Expected shape is transposed from wfdb's (5000, 12) to (12, 5000)
        self.assertEqual(record.signal.shape, (12, 5000))
        # Label for MI (index 1) should be 1.0
        np.testing.assert_array_equal(record.labels, np.array([0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32))


@unittest.skipIf(not HAS_TORCH, "PyTorch is not installed")
class TestECGDataset(unittest.TestCase):
    def test_dataset(self):
        mock_loader = MagicMock()
        mock_record = ECGRecord(
            record_id="1",
            signal=np.ones((12, 500), dtype=np.float32),
            sampling_rate=500,
            leads=["I"] * 12,
            labels=np.array([1, 0], dtype=np.float32)
        )
        mock_loader.load_record.return_value = mock_record
        
        dataset = ECGDataset(record_ids=[1], loader=mock_loader)
        self.assertEqual(len(dataset), 1)
        
        signal, label = dataset[0]
        self.assertTrue(isinstance(signal, torch.Tensor))
        self.assertTrue(isinstance(label, torch.Tensor))
        self.assertEqual(signal.shape, (12, 500))
        self.assertEqual(label.shape, (2,))


if __name__ == "__main__":
    unittest.main()
