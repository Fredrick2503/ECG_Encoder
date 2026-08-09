import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from data_management.dataset_factory import DatasetFactory
from representation_generation.biomarker_extractor import ECGFeatureExtractor

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("FullFeatureExtraction")

def extract_features_for_record(record_id, loader, extractor):
    try:
        record = loader.load_record(record_id)
        # We extract features using lead II (index 1 in standard 12-lead ECG: I, II, III...)
        lead_ii_signal = record.signal[1, :]
        
        # Extract features
        features, _ = extractor._extract_single_lead(lead_ii_signal)
        features["ecg_id"] = record_id
        return features
    except Exception as e:
        # Return dict with NaN features to keep index alignment
        return {"ecg_id": record_id, "error": str(e)}

def main():
    logger.info("Initializing Data Loader...")
    
    # Use DatasetFactory to load metadata and setup loader
    train_ds, val_ds, test_ds, loader = DatasetFactory.create_datasets(
        dataset_type="ptbxl",
        download=False,
        resolution="hr"
    )
    
    metadata_df = loader.load_metadata()
    
    # Filter to records that physically exist
    cfg = loader.root_dir
    target_subfolder = "records500"
    subfolder_path = Path(loader.root_dir) / target_subfolder
    
    existing_filenames = set()
    if subfolder_path.exists():
        for root, _, files in os.walk(subfolder_path):
            for f in files:
                if f.endswith(".hea"):
                    p = Path(root) / f
                    rel = p.relative_to(loader.root_dir)
                    existing_filenames.add(str(rel.with_suffix("")).replace("\\", "/"))
                    
    metadata_df = metadata_df[metadata_df["filename_hr"].isin(existing_filenames)]
    record_ids = metadata_df.index.tolist()
    
    logger.info(f"Total existing records to extract features from: {len(record_ids)}")
    
    # Initialize extractor
    extractor = ECGFeatureExtractor(fs=500, leads="II")
    
    logger.info("Starting Parallel Feature Extraction (this may take a few minutes)...")
    
    # Using joblib for parallel extraction
    results = Parallel(n_jobs=-1, prefer="processes")(
        delayed(extract_features_for_record)(rid, loader, extractor)
        for rid in tqdm(record_ids, desc="Extracting features")
    )
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Log any errors
    if "error" in results_df.columns:
        errors = results_df[results_df["error"].notna()]
        if len(errors) > 0:
            logger.warning(f"Failed to extract features for {len(errors)} records.")
            # Print a few examples of errors
            logger.warning(f"Sample errors: {errors['error'].head().tolist()}")
        results_df = results_df.drop(columns=["error"], errors="ignore")
        
    # Set ecg_id as index
    results_df.set_index("ecg_id", inplace=True)
    
    # Merge back into metadata_df
    logger.info("Merging extracted features with metadata...")
    final_df = metadata_df.join(results_df, how="inner")
    
    # Save output
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "full_biomarker_features.csv"
    final_df.to_csv(output_path)
    logger.info(f"Successfully saved extracted features to {output_path}. Shape: {final_df.shape}")

if __name__ == "__main__":
    main()
