from abc import ABC, abstractmethod
from typing import Tuple
import pandas as pd

class BaseSplitter(ABC):
    """Abstract Base Class for dataset splitting strategies."""
    
    @abstractmethod
    def split(self, metadata_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Splits the metadata DataFrame into train, val, and test partitions."""
        pass


class PTBXLFoldSplitter(BaseSplitter):
    """
    Standard fold-based splitting for PTB-XL.
    Folds 1-8: Train
    Fold 9: Validation
    Fold 10: Test
    """
    
    def __init__(self, train_folds=(1, 2, 3, 4, 5, 6, 7, 8), val_folds=(9,), test_folds=(10,)):
        self.train_folds = list(train_folds)
        self.val_folds = list(val_folds)
        self.test_folds = list(test_folds)

    def split(self, metadata_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits PTB-XL metadata DataFrame according to the 'strat_fold' column.
        
        Args:
            metadata_df (pd.DataFrame): PTB-XL database metadata.
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train_df, val_df, test_df)
        """
        if "strat_fold" not in metadata_df.columns:
            raise KeyError("Metadata DataFrame must contain a 'strat_fold' column.")
            
        train_df = metadata_df[metadata_df["strat_fold"].isin(self.train_folds)]
        val_df = metadata_df[metadata_df["strat_fold"].isin(self.val_folds)]
        test_df = metadata_df[metadata_df["strat_fold"].isin(self.test_folds)]
        
        return train_df, val_df, test_df
