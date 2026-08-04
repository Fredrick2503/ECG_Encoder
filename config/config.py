from config.constants import PTBXL_CLASSES
from config.paths import PTBXL_RAW_DIR

# Kaggle dataset path
DATASET_NAME = "khyeh0719/ptb-xl-dataset"

# Default project wide settings
SAMPLING_RATE = 500
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

# Dataset-specific default parameters
PTBXL_CONFIG = {
    "raw_dir": PTBXL_RAW_DIR,
    "database_csv": PTBXL_RAW_DIR / "ptbxl_database.csv",
    "scp_csv": PTBXL_RAW_DIR / "scp_statements.csv",
    "classes": PTBXL_CLASSES,
    "sampling_rate": SAMPLING_RATE,
    "lightweight": False
}
