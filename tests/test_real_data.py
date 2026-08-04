import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

def run_validation():
    if HAS_TORCH:
        from data_management.dataset_factory import DatasetFactory
        
        print("Initializing PyTorch loaders (Downloading dataset if it doesn't exist)...")
        train_loader, val_loader, test_loader, loader = DatasetFactory.create_dataloaders(
            dataset_type="ptbxl",
            download=True,
            batch_size=4
        )

        # Fetch a single batch
        signals, labels = next(iter(train_loader))

        print("\n--- Test Success (with PyTorch) ---")
        print("Signal tensor batch shape:", signals.shape)  # Expected: [4, 12, 5000]
        print("Label tensor batch shape:", labels.shape)    # Expected: [4, 5] (Multi-hot)

    else:
        print("PyTorch is not installed. Running lightweight validation (download and metadata/record parsing)...")
        from config.config import PTBXL_CONFIG, DATASET_NAME
        from data_management.downloader import PTBXLDownloader
        from data_management.loader import PTBXLLoader
        from data_management.label_encoder import PTBXLLabelEncoder
        from data_management.splitter import PTBXLFoldSplitter

        cfg = PTBXL_CONFIG.copy()
        
        # Download dataset if not exists
        print("Checking/Downloading dataset...")
        downloader = PTBXLDownloader(download_dir=cfg["raw_dir"], dataset_name=DATASET_NAME)
        downloader.download()
        
        # Instantiate Loader
        print("Loading metadata...")
        label_encoder = PTBXLLabelEncoder(classes=cfg["classes"])
        loader = PTBXLLoader(
            root_dir=cfg["raw_dir"],
            database_csv=cfg["database_csv"],
            scp_csv=cfg["scp_csv"],
            resolution="hr",
            label_encoder=label_encoder
        )
        metadata_df = loader.load_metadata()
        
        # Split
        splitter = PTBXLFoldSplitter()
        train_df, val_df, test_df = splitter.split(metadata_df)
        print(f"Train records: {len(train_df)}, Val records: {len(val_df)}, Test records: {len(test_df)}")
        
        # Load first record
        first_id = train_df.index[0]
        print(f"Loading record ID: {first_id}...")
        record = loader.load_record(first_id)
        
        print("\n--- Test Success (without PyTorch) ---")
        print("ECG Record ID:", record.record_id)
        print("Signal shape (num_leads, length):", record.signal.shape)  # Expected: (12, 5000)
        print("Labels array (multi-hot):", record.labels)
        print("Age:", record.age)
        print("Sex:", record.sex)

if __name__ == "__main__":
    run_validation()
