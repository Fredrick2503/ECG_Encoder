from pathlib import Path
from typing import Tuple, Optional, Any, Callable
from torch.utils.data import DataLoader

from config.config import PTBXL_CONFIG, DATASET_NAME
from data_management.downloader import PTBXLDownloader
from data_management.loader import PTBXLLoader
from data_management.label_encoder import PTBXLLabelEncoder
from data_management.splitter import PTBXLFoldSplitter
from data_management.sample_builder import ECGDataset

class DatasetFactory:
    """Factory class to create and configure ECG datasets and PyTorch DataLoaders."""
    
    @staticmethod
    def create_datasets(
        dataset_type: str = "ptbxl",
        download: bool = True,
        resolution: str = "hr",
        preprocessor: Optional[Any] = None,
        transform: Optional[Callable] = None,
        data_dir: Optional[Path] = None
    ) -> Tuple[ECGDataset, ECGDataset, ECGDataset, PTBXLLoader]:
        """
        Creates train, validation, and test ECGDatasets.
        
        Returns:
            Tuple[ECGDataset, ECGDataset, ECGDataset, PTBXLLoader]: 
            (train_dataset, val_dataset, test_dataset, loader)
        """
        if dataset_type.lower() != "ptbxl":
            raise ValueError(f"Unsupported dataset type: {dataset_type}. Currently only 'ptbxl' is supported.")
            
        # Resolve config and directories
        cfg = PTBXL_CONFIG.copy()
        if data_dir is not None:
            data_dir = Path(data_dir)
            cfg["raw_dir"] = data_dir
            cfg["database_csv"] = data_dir / "ptbxl_database.csv"
            cfg["scp_csv"] = data_dir / "scp_statements.csv"
            
        # Download dataset if requested
        if download:
            downloader = PTBXLDownloader(
                download_dir=cfg["raw_dir"],
                dataset_name=DATASET_NAME,
                lightweight=cfg.get("lightweight", True)
            )
            downloader.download()
            
        # Instantiate Label Encoder
        label_encoder = PTBXLLabelEncoder(classes=cfg["classes"])
        
        # Instantiate Loader
        loader = PTBXLLoader(
            root_dir=cfg["raw_dir"],
            database_csv=cfg["database_csv"],
            scp_csv=cfg["scp_csv"],
            resolution=resolution,
            label_encoder=label_encoder
        )
        metadata_df = loader.load_metadata()
        
        # Filter metadata_df to only include records that exist physically on disk
        import os
        import pandas as pd
        
        target_subfolder = "records500" if resolution == "hr" else "records100"
        subfolder_path = cfg["raw_dir"] / target_subfolder
        
        existing_filenames = set()
        if subfolder_path.exists():
            for root, _, files in os.walk(subfolder_path):
                for f in files:
                    if f.endswith(".hea"):
                        p = Path(root) / f
                        rel = p.relative_to(cfg["raw_dir"])
                        # Normalize paths to forward slashes to match metadata CSV format
                        existing_filenames.add(str(rel.with_suffix("")).replace("\\", "/"))
                        
        col_name = "filename_hr" if resolution == "hr" else "filename_lr"
        metadata_df = metadata_df[metadata_df[col_name].isin(existing_filenames)]
        num_existing = len(metadata_df)
            
        # If very few records are found (e.g. lightweight mode), replicate across splits
        if num_existing < 10:
            train_df = metadata_df.copy()
            train_df["strat_fold"] = 1
            val_df = metadata_df.copy()
            val_df["strat_fold"] = 9
            test_df = metadata_df.copy()
            test_df["strat_fold"] = 10
            metadata_df = pd.concat([train_df, val_df, test_df])
            
        # Split using standard PTB-XL splitter
        splitter = PTBXLFoldSplitter()
        train_df, val_df, test_df = splitter.split(metadata_df)
        
        # Build datasets
        train_dataset = ECGDataset(
            record_ids=train_df.index.tolist(),
            loader=loader,
            preprocessor=preprocessor,
            transform=transform
        )
        val_dataset = ECGDataset(
            record_ids=val_df.index.tolist(),
            loader=loader,
            preprocessor=preprocessor,
            transform=transform
        )
        test_dataset = ECGDataset(
            record_ids=test_df.index.tolist(),
            loader=loader,
            preprocessor=preprocessor,
            transform=transform
        )
        
        return train_dataset, val_dataset, test_dataset, loader

    @staticmethod
    def create_dataloaders(
        dataset_type: str = "ptbxl",
        download: bool = True,
        resolution: str = "hr",
        preprocessor: Optional[Any] = None,
        transform: Optional[Callable] = None,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        data_dir: Optional[Path] = None
    ) -> Tuple[DataLoader, DataLoader, DataLoader, PTBXLLoader]:
        """
        Creates train, validation, and test PyTorch DataLoaders.
        
        Returns:
            Tuple[DataLoader, DataLoader, DataLoader, PTBXLLoader]:
            (train_loader, val_loader, test_loader, loader)
        """
        train_ds, val_ds, test_ds, loader = DatasetFactory.create_datasets(
            dataset_type=dataset_type,
            download=download,
            resolution=resolution,
            preprocessor=preprocessor,
            transform=transform,
            data_dir=data_dir
        )
        
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        return train_loader, val_loader, test_loader, loader
