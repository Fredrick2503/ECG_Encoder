from pathlib import Path

# Project structure paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directory structure
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Specific dataset directories
PTBXL_RAW_DIR = RAW_DATA_DIR / "ptbxl"
MITBIH_RAW_DIR = RAW_DATA_DIR / "mitbih"

# Ensure directories exist
for path in [DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, PTBXL_RAW_DIR, MITBIH_RAW_DIR]:
    path.mkdir(parents=True, exist_ok=True)
