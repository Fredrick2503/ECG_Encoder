import json
from pathlib import Path

def generate_notebook():
    notebook_path = Path("notebooks/01_data_management_and_preprocessing.ipynb")
    
    cells = []
    
    # Cell 1: Title
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# ECG Foundation Representation System\n",
            "## Phase 1 Research Workspace: Data Management & Signal Preprocessing\n",
            "\n",
            "This notebook serves as an interactive playground and visualization dashboard for the completed modules:\n",
            "- **Data Management**: PTB-XL metadata analysis, dataset splitting, and WFDB record loading.\n",
            "- **Preprocessing**: Butterworth bandpass, Notch, FIR filtering, Wavelet denoising, Z-score/Min-Max/Robust normalization, Pan-Tompkins QRS beat detection/segmentation, DBSCAN outlier removal, and SMOTE-ENN balancing."
        ]
    })
    
    # Cell 2: Imports & Path setup
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import sys\n",
            "from pathlib import Path\n",
            "\n",
            "# Ensure project root is in path\n",
            "project_root = Path.cwd().parent\n",
            "if str(project_root) not in sys.path:\n",
            "    sys.path.append(str(project_root))\n",
            "\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "print(\"Environment initialized and path resolved.\")"
        ]
    })

    # Cell 3: Data Management Markdown
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Data Management: Metadata Parsing & Dataset Loader\n",
            "Here we download the dataset using the lightweight mode (which fetches the metadata CSV files and the first record of both 100Hz and 500Hz formats directly from PhysioNet, ~10MB total) and examine the database statistics."
        ]
    })

    # Cell 4: Download dataset
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from data_management.downloader import PTBXLDownloader\n",
            "from config.config import PTBXL_CONFIG, DATASET_NAME\n",
            "\n",
            "cfg = PTBXL_CONFIG.copy()\n",
            "\n",
            "# Initialize downloader with lightweight mode\n",
            "downloader = PTBXLDownloader(cfg[\"raw_dir\"], dataset_name=DATASET_NAME, lightweight=True)\n",
            "downloader.download()"
        ]
    })

    # Cell 5: Load database metadata
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from data_management.loader import PTBXLLoader\n",
            "from data_management.label_encoder import PTBXLLabelEncoder\n",
            "from data_management.splitter import PTBXLFoldSplitter\n",
            "\n",
            "label_encoder = PTBXLLabelEncoder(classes=cfg[\"classes\"])\n",
            "loader = PTBXLLoader(\n",
            "    root_dir=cfg[\"raw_dir\"],\n",
            "    database_csv=cfg[\"database_csv\"],\n",
            "    scp_csv=cfg[\"scp_csv\"],\n",
            "    resolution=\"hr\",\n",
            "    label_encoder=label_encoder\n",
            ")\n",
            "metadata_df = loader.load_metadata()\n",
            "\n",
            "# Print dataset dimensions\n",
            "print(f\"Total records loaded: {len(metadata_df)}\")\n",
            "print(f\"Leads configured: {cfg['classes']}\")"
        ]
    })

    # Cell 6: Visualize metadata statistics markdown
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Demographics & Fold Splitting\n",
            "Let's visualize the distribution of age, sex, and the train/val/test fold splits."
        ]
    })

    # Cell 7: Plot demographics
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n",
            "\n",
            "# Age histogram\n",
            "metadata_df[\"age\"].dropna().hist(bins=30, ax=axes[0], color=\"#1f77b4\", edgecolor=\"black\")\n",
            "axes[0].set_title(\"Patient Age Distribution\")\n",
            "axes[0].set_xlabel(\"Age (Years)\")\n",
            "axes[0].set_ylabel(\"Count\")\n",
            "\n",
            "# Sex pie chart\n",
            "sex_counts = metadata_df[\"sex\"].value_counts()\n",
            "axes[1].pie(sex_counts, labels=sex_counts.index, autopct='%1.1f%%', colors=[\"#33b5e5\", \"#ff4444\"], startangle=90)\n",
            "axes[1].set_title(\"Patient Gender Distribution\")\n",
            "\n",
            "# Split distribution\n",
            "splitter = PTBXLFoldSplitter()\n",
            "train_df, val_df, test_df = splitter.split(metadata_df)\n",
            "split_counts = [len(train_df), len(val_df), len(test_df)]\n",
            "axes[2].bar([\"Train (Folds 1-8)\", \"Val (Fold 9)\", \"Test (Fold 10)\"], split_counts, color=[\"#2bbbad\", \"#ffbb33\", \"#aa66cc\"])\n",
            "axes[2].set_title(\"Dataset Split Distribution\")\n",
            "axes[2].set_ylabel(\"Number of Records\")\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    })

    # Cell 8: Signal Visualization Markdown
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Loading a Raw ECG Record\n",
            "We load the first record (ID 1) and plot its 12-lead waveforms."
        ]
    })

    # Cell 9: Plot 12 leads
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "record = loader.load_record(1)\n",
            "time = np.arange(record.signal.shape[1]) / record.sampling_rate\n",
            "\n",
            "plt.figure(figsize=(15, 12))\n",
            "for i in range(min(12, record.signal.shape[0])):\n",
            "    plt.subplot(6, 2, i + 1)\n",
            "    plt.plot(time[:1500], record.signal[i, :1500], color=\"crimson\", linewidth=1.0)\n",
            "    plt.title(f\"Lead: {record.leads[i]}\")\n",
            "    plt.xlabel(\"Time (Seconds)\")\n",
            "    plt.ylabel(\"mV\")\n",
            "    plt.grid(True, linestyle=\"--\", alpha=0.5)\n",
            "\n",
            "plt.suptitle(f\"Raw 12-Lead ECG - Record ID {record.record_id} (Patient Age: {record.age}, Sex: {record.sex})\", fontsize=16)\n",
            "plt.tight_layout(rect=[0, 0, 1, 0.96])\n",
            "plt.show()"
        ]
    })

    # Cell 10: Filtering Markdown
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Signal Preprocessing: Filtering and Wavelet Denoising\n",
            "Here we benchmark the zero-phase Butterworth Bandpass filter, the Notch filter (to remove 60 Hz powerline interference), and discrete wavelet transform soft-thresholding denoising (using `db4`)."
        ]
    })

    # Cell 11: Plot filtering comparison
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from preprocessing.filters import ButterworthFilter, NotchFilter, WaveletDenoise\n",
            "\n",
            "raw_signal = record.signal\n",
            "fs = record.sampling_rate\n",
            "\n",
            "# 1. Butterworth Bandpass (0.5 - 45 Hz)\n",
            "butter_filt = ButterworthFilter(lowcut=0.5, highcut=45.0, order=4)\n",
            "butter_signal = butter_filt.process(raw_signal, fs)\n",
            "\n",
            "# 2. Notch Filter (60 Hz)\n",
            "notch_filt = NotchFilter(notch_freq=60.0)\n",
            "notch_signal = notch_filt.process(raw_signal, fs)\n",
            "\n",
            "# 3. Wavelet Denoise (db4, level 4)\n",
            "wavelet_filt = WaveletDenoise(wavelet=\"db4\", level=4)\n",
            "wavelet_signal = wavelet_filt.process(raw_signal, fs)\n",
            "\n",
            "# Plot Comparison (Lead II, index 1)\n",
            "lead_idx = 1\n",
            "plt.figure(figsize=(15, 10))\n",
            "t_crop = time[:1000]\n",
            "\n",
            "plt.subplot(4, 1, 1)\n",
            "plt.plot(t_crop, raw_signal[lead_idx, :1000], color=\"gray\")\n",
            "plt.title(\"Raw ECG Signal\")\n",
            "plt.ylabel(\"mV\")\n",
            "\n",
            "plt.subplot(4, 1, 2)\n",
            "plt.plot(t_crop, butter_signal[lead_idx, :1000], color=\"blue\")\n",
            "plt.title(\"Butterworth Bandpass Filtered (0.5 - 45 Hz)\")\n",
            "plt.ylabel(\"mV\")\n",
            "\n",
            "plt.subplot(4, 1, 3)\n",
            "plt.plot(t_crop, notch_signal[lead_idx, :1000], color=\"orange\")\n",
            "plt.title(\"Notch Filtered (60 Hz Removal)\")\n",
            "plt.ylabel(\"mV\")\n",
            "\n",
            "plt.subplot(4, 1, 4)\n",
            "plt.plot(t_crop, wavelet_signal[lead_idx, :1000], color=\"green\")\n",
            "plt.title(\"Wavelet Denoised (db4 Soft Thresholding)\")\n",
            "plt.ylabel(\"mV\")\n",
            "plt.xlabel(\"Time (Seconds)\")\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    })

    # Cell 12: Normalization Markdown
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Signal Normalization\n",
            "We compare standard Z-score standardisation (mean=0, std=1), Min-Max scaling ([0, 1]), and Robust Scaling (median/IQR based)."
        ]
    })

    # Cell 13: Normalization Code
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from preprocessing.normalization import ZScoreNormalizer, MinMaxNormalizer, RobustNormalizer\n",
            "\n",
            "lead_sig = butter_signal[[lead_idx]]\n",
            "\n",
            "zscore_out = ZScoreNormalizer().process(lead_sig, fs)\n",
            "minmax_out = MinMaxNormalizer(feature_range=(0.0, 1.0)).process(lead_sig, fs)\n",
            "robust_out = RobustNormalizer().process(lead_sig, fs)\n",
            "\n",
            "plt.figure(figsize=(15, 6))\n",
            "plt.plot(t_crop, zscore_out[0, :1000], label=f\"Z-Score (Mean={np.mean(zscore_out):.2f}, Std={np.std(zscore_out):.2f})\", color=\"blue\")\n",
            "plt.plot(t_crop, minmax_out[0, :1000], label=f\"Min-Max (Range={np.min(minmax_out):.1f} to {np.max(minmax_out):.1f})\", color=\"green\", alpha=0.8)\n",
            "plt.plot(t_crop, robust_out[0, :1000], label=f\"Robust Scale (Median={np.median(robust_out):.2f})\", color=\"purple\", linestyle=\"--\")\n",
            "\n",
            "plt.title(\"ECG Normalization Output Comparison\")\n",
            "plt.xlabel(\"Time (Seconds)\")\n",
            "plt.ylabel(\"Normalized Scale\")\n",
            "plt.legend()\n",
            "plt.grid(True)\n",
            "plt.show()"
        ]
    })

    # Cell 14: Segmentation Markdown
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Beat Segmentation & Pan-Tompkins QRS Detection\n",
            "The Pan-Tompkins algorithm isolates QRS complexes, finds R-peak indices, and cuts heartbeats centered around the peaks."
        ]
    })

    # Cell 15: QRS Detection Plot
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from preprocessing.segmentation import PanTompkinsSegmenter\n",
            "\n",
            "segmenter = PanTompkinsSegmenter(pre_r_samples=150, post_r_samples=250)\n",
            "\n",
            "# Run the peak detection explicitly to plot R-peak indicators\n",
            "lead_II = butter_signal[1]\n",
            "peaks = segmenter._detect_r_peaks(lead_II, fs)\n",
            "\n",
            "plt.figure(figsize=(15, 8))\n",
            "\n",
            "# 1. Plot R-peak detection indicators\n",
            "plt.subplot(2, 1, 1)\n",
            "plt.plot(time[:2500], lead_II[:2500], color=\"blue\", label=\"Clean ECG Lead II\")\n",
            "plt.scatter(peaks[peaks < 2500] / fs, lead_II[peaks[peaks < 2500]], color=\"red\", marker=\"v\", s=100, label=\"Detected R-Peaks\")\n",
            "plt.title(\"Pan-Tompkins R-Peak Detection (First 5 seconds)\")\n",
            "plt.ylabel(\"mV\")\n",
            "plt.legend()\n",
            "plt.grid(True)\n",
            "\n",
            "# 2. Plot segmented individual beats\n",
            "beats = segmenter.process(butter_signal, fs)\n",
            "plt.subplot(2, 1, 2)\n",
            "beat_time = np.arange(beats.shape[2]) / fs\n",
            "for i in range(min(5, len(beats))):\n",
            "    plt.plot(beat_time, beats[i, 1], label=f\"Beat {i+1}\")\n",
            "\n",
            "plt.title(\"Segmented Single Heartbeats Centered around R-Peaks (Shape: (beats, leads, window))\")\n",
            "plt.xlabel(\"Time (Seconds) centered at R-peak\")\n",
            "plt.ylabel(\"mV\")\n",
            "plt.legend()\n",
            "plt.grid(True)\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    })

    # Cell 16: Outlier Detection Markdown
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Outlier Detection using DBSCAN\n",
            "We extract statistical signal descriptors from our record and apply density-based clustering to check for outlier signals in the batch."
        ]
    })

    # Cell 17: Outlier Code
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from preprocessing.outlier_detection import DBSCANOutlierDetector\n",
            "\n",
            "# Create a synthetic batch of records: 9 clean, 1 outlier (amplitude anomaly)\n",
            "batch_records = []\n",
            "for i in range(9):\n",
            "    batch_records.append(butter_signal[:, :1000])\n",
            "batch_records.append(butter_signal[:, :1000] * 20.0) # Scaled outlier record\n",
            "\n",
            "signals_batch = np.array(batch_records)\n",
            "detector = DBSCANOutlierDetector(eps=2.0, min_samples=3)\n",
            "inliers, outliers = detector.detect_outliers(signals_batch)\n",
            "\n",
            "print(f\"Inlier indices: {list(inliers)}\")\n",
            "print(f\"Detected Outlier indices (anomalies flagged): {list(outliers)}\")"
        ]
    })

    # Cell 18: Balancing Markdown
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Training Resampling & Balancing (SMOTE-ENN)\n",
            "SMOTE-ENN is applied to address class imbalance during training. We map class configurations via label power-sets for multi-label data."
        ]
    })

    # Cell 19: Resampling code
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from preprocessing.balancing import ECGDatasetBalancer\n",
            "\n",
            "# Generate dummy imbalanced class distribution (15 Class 0, 15 Class 1)\n",
            "toy_signals = np.random.normal(0, 1.0, (30, 2, 100))\n",
            "toy_labels = np.array([0] * 15 + [1] * 15)\n",
            "\n",
            "balancer = ECGDatasetBalancer(random_state=42)\n",
            "signals_res, labels_res = balancer.balance_dataset(toy_signals, toy_labels)\n",
            "\n",
            "print(f\"Original dataset size: {len(toy_signals)} records\")\n",
            "print(f\"Resampled balanced dataset size: {len(signals_res)} records\")"
        ]
    })

    # Cell 20: Preprocessing Profile Manager Markdown
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 7. Preprocessing Profile Manager Facade\n",
            "The `PreprocessingManager` constructs configured pipelines (`temporal`, `morphology`, `biomarker`) and runs them directly on `ECGRecord` objects."
        ]
    })

    # Cell 21: Manager run and compare
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from preprocessing.manager import PreprocessingManager\n",
            "\n",
            "manager = PreprocessingManager()\n",
            "\n",
            "# Run Temporal pipeline profile\n",
            "rec_temporal = manager.preprocess_record(record, \"temporal\")\n",
            "print(f\"Temporal signal shape: {rec_temporal.signal.shape}\")\n",
            "\n",
            "# Run Morphology pipeline profile\n",
            "rec_morphology = manager.preprocess_record(record, \"morphology\")\n",
            "print(f\"Morphology signal shape (individual heartbeats): {rec_morphology.signal.shape}\")\n",
            "\n",
            "# Run Biomarker pipeline profile\n",
            "rec_biomarker = manager.preprocess_record(record, \"biomarker\")\n",
            "print(f\"Biomarker signal shape: {rec_biomarker.signal.shape}\")"
        ]
    })

    notebook_data = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.11"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook_data, f, indent=1)
        
    print(f"Notebook successfully generated at: {notebook_path}")

if __name__ == "__main__":
    generate_notebook()
