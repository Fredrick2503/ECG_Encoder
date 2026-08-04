import ast
import pandas as pd
from pathlib import Path
from typing import Dict, List, Set

class PTBXLMetadataParser:
    """Class to parse and clean PTB-XL database metadata and SCP statement mappings."""
    
    def __init__(self, database_csv: Path, scp_csv: Path):
        self.database_csv = Path(database_csv)
        self.scp_csv = Path(scp_csv)
        self.database_df = None
        self.scp_df = None
        self._code_to_class = {}
        self._code_to_subclass = {}

    def load_scp_statements(self) -> pd.DataFrame:
        """Loads SCP statements and constructs mappings from SCP code to diagnostic class/subclass."""
        if not self.scp_csv.exists():
            raise FileNotFoundError(f"SCP statement mappings CSV not found at {self.scp_csv}")
        
        self.scp_df = pd.read_csv(self.scp_csv, index_col=0)
        
        # Populate translation dictionaries
        for code, row in self.scp_df.iterrows():
            diag_class = row.get("diagnostic_class")
            subclass = row.get("subclass")
            
            if pd.notna(diag_class):
                self._code_to_class[code] = diag_class
            if pd.notna(subclass):
                self._code_to_subclass[code] = subclass
                
        return self.scp_df

    def load_database(self) -> pd.DataFrame:
        """Loads and cleans the main PTB-XL database CSV."""
        if not self.database_csv.exists():
            raise FileNotFoundError(f"PTB-XL database CSV not found at {self.database_csv}")
        
        # Read database index by ecg_id
        df = pd.read_csv(self.database_csv, index_col="ecg_id")
        
        # Parse scp_codes string representation into dictionaries
        df["scp_codes"] = df["scp_codes"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else {})
        
        self.database_df = df
        return df

    def get_diagnostic_classes(self, scp_codes: Dict[str, float]) -> List[str]:
        """
        Maps dictionary of SCP codes to a list of matching aggregate diagnostic classes.
        Example: {'NDI': 100.0, 'STR': 0.0} -> ['MI']
        """
        classes = set()
        for code, likelihood in scp_codes.items():
            # If code is assigned (likelihood > 0 or 100)
            if likelihood > 0:
                mapped_class = self._code_to_class.get(code)
                if mapped_class:
                    classes.add(mapped_class)
        return list(classes)

    def get_subclasses(self, scp_codes: Dict[str, float]) -> List[str]:
        """Maps dictionary of SCP codes to subclasses."""
        subclasses = set()
        for code, likelihood in scp_codes.items():
            if likelihood > 0:
                mapped_sub = self._code_to_subclass.get(code)
                if mapped_sub:
                    subclasses.add(mapped_sub)
        return list(subclasses)
