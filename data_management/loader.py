from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, List, Optional, Union
import numpy as np
import pandas as pd
import wfdb

from data_management.ecg_record import ECGRecord
from data_management.metadata import PTBXLMetadataParser
from data_management.label_encoder import PTBXLLabelEncoder
from config.constants import STANDARD_12_LEADS

class BaseLoader(ABC):
    """Abstract Base Class for ECG dataset loaders."""
    
    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)
        self.metadata_df = None

    @abstractmethod
    def load_metadata(self) -> pd.DataFrame:
        """Loads dataset metadata dataframe."""
        pass

    @abstractmethod
    def load_record(self, record_id: Union[int, str]) -> ECGRecord:
        """Loads an ECGRecord by identifier."""
        pass


class PTBXLLoader(BaseLoader):
    """Loader for the PTB-XL dataset."""
    
    def __init__(
        self,
        root_dir: Path,
        database_csv: Path,
        scp_csv: Path,
        resolution: str = "hr",
        label_encoder: Optional[PTBXLLabelEncoder] = None
    ):
        super().__init__(root_dir)
        self.parser = PTBXLMetadataParser(database_csv, scp_csv)
        self.resolution = resolution  # "hr" for 500Hz, "lr" for 100Hz
        self.label_encoder = label_encoder

    def load_metadata(self) -> pd.DataFrame:
        """Loads and prepares the metadata database and SCP statement mappings."""
        self.parser.load_scp_statements()
        self.metadata_df = self.parser.load_database()
        return self.metadata_df

    def load_record(self, ecg_id: int) -> ECGRecord:
        """Loads a single ECGRecord from the database."""
        if self.metadata_df is None:
            self.load_metadata()
            
        if ecg_id not in self.metadata_df.index:
            raise KeyError(f"ECG ID {ecg_id} not found in metadata.")
            
        row = self.metadata_df.loc[ecg_id]
        
        # Resolve file path based on resolution
        if self.resolution == "hr":
            file_path = self.root_dir / row["filename_hr"]
            sampling_rate = 500
        else:
            file_path = self.root_dir / row["filename_lr"]
            sampling_rate = 100
            
        # Read signal from physical file using wfdb
        signal, meta = wfdb.rdsamp(str(file_path))
        
        # Raw signal returned by wfdb is (signal_length, num_leads).
        # We standardise it to (num_leads, signal_length) for models.
        signal = signal.T.astype(np.float32)
        
        # Get lead names from metadata or fallback to standard 12 leads
        leads = meta.get("sig_name", STANDARD_12_LEADS)
        
        # Resolve labels
        scp_codes = row.get("scp_codes", {})
        diagnostic_classes = self.parser.get_diagnostic_classes(scp_codes)
        
        encoded_labels = np.zeros(0, dtype=np.float32)
        if self.label_encoder is not None:
            encoded_labels = self.label_encoder.encode(diagnostic_classes)
            
        # Extract demographic/patient details
        patient_id = int(row["patient_id"]) if pd.notna(row["patient_id"]) else None
        age = float(row["age"]) if pd.notna(row["age"]) else None
        
        # Normalize sex string
        sex_raw = row["sex"]
        sex = None
        if pd.notna(sex_raw):
            if str(sex_raw).strip().upper() in ["M", "1", "1.0", "MALE"]:
                sex = "M"
            elif str(sex_raw).strip().upper() in ["F", "0", "0.0", "FEMALE"]:
                sex = "F"
                
        # Build additional metadata dictionary
        record_meta = {
            "diagnostic_classes": diagnostic_classes,
            "subclasses": self.parser.get_subclasses(scp_codes),
            "recording_date": row.get("recording_date"),
            "strat_fold": row.get("strat_fold"),
            "height": row.get("height"),
            "weight": row.get("weight"),
            "nurse": row.get("nurse"),
            "device": row.get("device")
        }
        
        return ECGRecord(
            record_id=str(ecg_id),
            signal=signal,
            sampling_rate=sampling_rate,
            leads=leads,
            labels=encoded_labels,
            metadata=record_meta,
            patient_id=patient_id,
            age=age,
            sex=sex
        )
