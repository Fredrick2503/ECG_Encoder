# 🫀 ECG Foundation Representation System

> **Multimodal Deep Foundation Architecture & Clinical Decision Support System for 12-Lead Electrocardiograms**  
> *Engineered for Master's Thesis Research, Clinical Validation, and Independent Exploration.*

---

## 🌟 Quick Overview

The **ECG Foundation Representation System** is a production-ready, tri-modal representation learning and diagnostic classification pipeline for 12-lead electrocardiograms (ECGs). It bridges the gap between raw, continuous physiological waveform dynamics and classical clinical diagnostic criteria:

```
                                 ┌─────────────────────────┐
                                 │   12-Lead ECG Signal    │
                                 │ (12 channels x 1000 ts) │
                                 └───────────┬─────────────┘
                                             │
               ┌─────────────────────────────┼─────────────────────────────┐
               │                             │                             │
               ▼                             ▼                             ▼
    ┌──────────────────────┐      ┌──────────────────────┐      ┌──────────────────────┐
    │   Temporal Encoder   │      │  Morphology Encoder  │      │  Biomarker Service   │
    │    (1D ResNet-SE)    │      │ (2D Spectrogram CNN) │      │ (24 Feats + 24 Mask) │
    │    Output: 512-D     │      │    Output: 512-D     │      │    Output: 32-D      │
    └──────────┬───────────┘      └──────────┬───────────┘      └──────────┬───────────┘
               │                             │                             │
               └─────────────────────────────┼─────────────────────────────┘
                                             │
                                             ▼
                              ┌─────────────────────────────┐
                              │  Multimodal Concatenation   │
                              │     Z_fused (1056-D)        │
                              └──────────────┬──────────────┘
                                             │
                                             ▼
                              ┌─────────────────────────────┐
                              │ Dense Bottleneck (128-D)    │
                              │  + Multi-Label Classifier   │
                              └──────────────┬──────────────┘
                                             │
                                             ▼
                              ┌─────────────────────────────┐
                              │ Youden-Calibrated Threshold │
                              │ [NORM, MI, STTC, CD, HYP]   │
                              └─────────────────────────────┘
```

---

## ⚙️ Environment Setup & Installation

### Step 1: System Requirements
- **Operating System:** Windows 10/11, Linux, or macOS
- **Python:** Python 3.10 or 3.11 recommended
- **Hardware:** NVIDIA CUDA GPU supported and auto-detected; full CPU execution is also supported automatically.

### Step 2: Create & Activate Virtual Environment

#### 🪟 Windows (Command Prompt or PowerShell)
```cmd
:: 1. Create virtual environment
python -m venv .venv

:: 2. Activate in Windows Command Prompt (CMD):
.\.venv\Scripts\activate.bat

:: OR activate in Windows PowerShell:
.\.venv\Scripts\Activate.ps1
```

#### 🐧 Linux / 🍎 macOS (Terminal)
```bash
# 1. Create virtual environment
python3 -m venv .venv

# 2. Activate in Bash/Zsh:
source .venv/bin/activate
```

### Step 3: Install Dependencies

**Windows:**
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Linux / macOS:**
```bash
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Verify Model Weights & Artifacts

**Windows:**
```powershell
python scripts\run_engine.py info
```

**Linux / macOS:**
```bash
python3 scripts/run_engine.py info
```
*(Confirms that all 7 pretrained neural network weights and calibration files report `PRESENT`.)*

### Step 5 (Optional): Configure Gemini API Key for LLM Clinical Reports
For enhanced AI clinical interpretations, set the Gemini API key in your terminal or in a `.env` file in the project root:

- **Windows (Command Prompt):** `set GEMINI_API_KEY=your_gemini_api_key_here`
- **Windows (PowerShell):** `$env:GEMINI_API_KEY="your_gemini_api_key_here"`
- **Linux / macOS:** `export GEMINI_API_KEY="your_gemini_api_key_here"`
- **Or via `.env` file:** Create `.env` in the repository root containing `GEMINI_API_KEY=your_key`

*(If omitted, the system automatically uses the offline clinical heuristic generator with full interval analyses.)*

---

## 🚀 Running the Project

### ⚡ 1-Click All-in-One Bootstrap (Setup Env → Download Data → Run Web App)

To initialize and run the whole project in one automated go from scratch:

- **🪟 Windows (1-Click File Explorer):** Double-click [`setup_and_run.bat`](setup_and_run.bat)
- **🪟 Windows (Command Prompt / PowerShell):**
  ```cmd
  .\setup_and_run.bat
  :: Or via python:
  python scripts\setup_and_run.py
  ```
- **🐧 Linux / 🍎 macOS (Terminal):**
  ```bash
  bash setup_and_run.sh
  # Or via python:
  python3 scripts/setup_and_run.py
  ```

#### What the Bootstrap Pipeline Executes Automatically:
1. **[1/5] Environment Initialization:** Checks for Python 3.10+, creates `.venv` if not present, and seamlessly pivots execution into the virtual environment.
2. **[2/5] Dependency Installation:** Verifies and installs all required libraries from `requirements.txt` via `pip`.
3. **[3/5] Weights & Artifacts Registry Check:** Confirms that all 7 foundational model weights, classifier heads, and biomarker transformers are present and intact.
4. **[4/5] PTB-XL Data Verification & PhysioNet Download:** Checks the local cache in `web_app/data/cached_signals/`. If additional waveforms are needed, automatically streams and caches authentic 12-lead waveforms directly from PhysioNet (`ptb-xl/1.0.3`) via `wfdb`.
5. **[5/5] Engine Smoke Test:** Runs a 1-second forward pass through the 1056-D multimodal representation pipeline on your active device (GPU CUDA or CPU).
6. **[LAUNCH] Web Application:** Resolves local port availability, starts the web server, and automatically opens your default browser at `http://localhost:8000`.

#### Advanced Bootstrap CLI Flags:
```powershell
# Fast launch (skip dependency checks and data downloads if already completed)
python scripts\setup_and_run.py --skip-install --skip-data

# Download a larger batch of records (e.g. 50 records) before starting
python scripts\setup_and_run.py --limit-data 50

# Custom port or headless server mode (no browser window)
python scripts\setup_and_run.py --port 8080 --no-browser
```

---

### 📥 Standalone PTB-XL Data Downloader & Cache Manager (`download_data`)

If you want to manage, inspect, or pre-download authentic PTB-XL 12-lead ECG records from PhysioNet independently of the web server:

**Windows (CMD / PowerShell):**
```cmd
:: 1. Inspect local disk cache status and record count
python scripts\download_data.py --status

:: 2. Download and cache the top 30 records from PhysioNet
python scripts\download_data.py --limit 30

:: 3. Download specific PTB-XL records by ID
python scripts\download_data.py --records 1,2,21826,10755
```

**Linux / macOS:**
```bash
python3 scripts/download_data.py --status
python3 scripts/download_data.py --limit 30
python3 scripts/download_data.py --records 1,2,21826,10755
```
Downloaded records are transposed to standard diagnostic shape `(12, 1000)` and cached as `.npy` in `web_app/data/cached_signals/<ecg_id>.npy` for immediate offline access.

---

### 🎛️ Workflow Launchers Summary Table

| Workflow | Windows 1-Click (`.bat`) | Windows CLI (CMD / PowerShell) | Linux / macOS CLI |
| :--- | :--- | :--- | :--- |
| **Complete Bootstrap** | [`setup_and_run.bat`](setup_and_run.bat) | `.\setup_and_run.bat` | `bash setup_and_run.sh` |
| **Master Control Menu** | [`run_application.bat`](run_application.bat) | `python scripts\run_app.py` | `python3 scripts/run_app.py` |
| **Foundation Engine Pipeline** | [`run_engine.bat`](run_engine.bat) | `python scripts\run_engine.py demo` | `python3 scripts/run_engine.py demo` |
| **Individual Diagnostic Report** | [`generate_report.bat`](generate_report.bat) | `python scripts\generate_report.py --open` | `python3 scripts/generate_report.py --open` |
| **Clinical Web Application** | [`run_web_app.bat`](run_web_app.bat) | `python scripts\run_web.py` | `python3 scripts/run_web.py` |
| **Unit & Integration Tests** | [`run_tests.bat`](run_tests.bat) | `python scripts\run_engine.py test` | `python3 scripts/run_engine.py test` |
| **PhysioNet Data Cache** | — | `python scripts\download_data.py --limit 30` | `python3 scripts/download_data.py --limit 30` |

---

### 1️⃣ Master Application Launcher (`run_app`)

Launches an interactive command-line control panel allowing you to access all features without memorizing CLI arguments:

- **Windows 1-Click:** Double-click [`run_application.bat`](run_application.bat)
- **Windows CLI:** `python scripts\run_app.py`
- **Linux/macOS:** `python3 scripts/run_app.py`

```text
  ==============================================================================
    ECG FOUNDATION REPRESENTATION SYSTEM — MASTER CONTROL PANEL
    Tri-Modal 1056-D Foundation Architecture & Clinical Decision Support
  ==============================================================================
  Please select an operation:

    [1] Launch Web Application (Clinical Report Generator & Visualizer)
    [2] Run ECG Foundation Engine (Verification Demo & 12-Lead Inference)
    [3] Generate Individual ECG Clinical Diagnostic Report (HTML + Summary)
    [4] Inspect Model Architecture & Frozen Weights Registry
    [5] Run Engine Verification Unit Tests
    [6] List Available PTB-XL ECG Catalog Records (600 Records)
    [0] Exit
  ------------------------------------------------------------------------------
```

**Direct Subcommand Execution:**
```cmd
:: Windows Command Prompt / PowerShell:
python scripts\run_app.py web --port 8000
python scripts\run_app.py engine demo
python scripts\run_app.py report --sample REC-21826 --open
python scripts\run_app.py info
python scripts\run_app.py test

:: Linux / macOS:
python3 scripts/run_app.py web --port 8000
python3 scripts/run_app.py engine demo
python3 scripts/run_app.py report --sample REC-21826 --open
```

---

### 2️⃣ ECG Foundation Engine (`run_engine`)

Directly exercises the multimodal feature extractors, multimodal fusion engine, and calibrated diagnostic classifiers:

- **Windows 1-Click:** Double-click [`run_engine.bat`](run_engine.bat)
- **Windows CLI (CMD / PowerShell):**
  ```cmd
  :: Run end-to-end inference verification demo (batch of 12-lead signals)
  python scripts\run_engine.py demo

  :: Force CPU inference
  python scripts\run_engine.py demo --device cpu

  :: Print model registry, tensor shapes, and file sizes
  python scripts\run_engine.py info

  :: Run engine unit & integration test suite
  python scripts\run_engine.py test

  :: Predict diagnostic conditions for an external .npy or .csv ECG file
  python scripts\run_engine.py predict --input path\to\ecg.npy

  :: Extract and save 1056-D fused multimodal representation (z_fused)
  python scripts\run_engine.py encode --input path\to\ecg.npy --output z_fused.npy
  ```

---

### 3️⃣ Individual Clinical Diagnostic Report Generator (`generate_report`)

Generates a complete, publication-grade individual medical diagnostic report for any record in the PTB-XL database or any custom 12-lead signal file.

- **Windows 1-Click:** Double-click [`generate_report.bat`](generate_report.bat) (prompts for record ID or defaults to `REC-01`)
- **Windows CLI (CMD / PowerShell):**
  ```cmd
  :: Generate report for default sample and open immediately in default browser
  python scripts\generate_report.py --open

  :: Generate report for a specific PTB-XL record (e.g. REC-21826: Acute MI + Hypertrophy)
  python scripts\generate_report.py --sample REC-21826 --open

  :: Generate report for your own custom 12-lead ECG signal (.npy or .csv)
  python scripts\generate_report.py --input path\to\my_signal.npy --open

  :: List available preset catalog samples (600 records)
  python scripts\generate_report.py --list
  ```

#### What gets generated:
1. **Console Medical Summary:** Clean ASCII overview of patient demographics, Youden-calibrated probabilities, and 24 electrophysiological biomarkers with reference ranges.
2. **High-Contrast HTML Medical Report** (saved to `outputs/reports/<sample_id>_clinical_report.html`):
   - Standard **12-lead diagnostic ECG tracing** (4x3 layout: I, II, III; aVR, aVL, aVF; V1–V6 + full-width 10s Lead II rhythm strip & 1.0 mV calibration pulse).
   - Core Lab medical stationery layout and patient demographics.
   - Calibrated 5-class superclass prediction bars with Youden decision thresholds.
   - 24 electrophysiological biomarkers table with status tags (`NORMAL`, `HIGH`, `LOW`).
   - Electrophysiological & clinical narrative impression.
   - Full A4/Letter print optimization (`@media print`) — press `Ctrl+P` or click "Print Report" for PDF export.
3. **Structured JSON Export** (saved to `outputs/reports/<sample_id>_clinical_report.json`): Contains raw numeric representations, probabilities, thresholds, and biomarker values.

---

### 4️⃣ Clinical Diagnostic Report Generator Web App (`run_web`)

Interactive web application hosting the ECG foundation model, real-time waveform visualization, 3D latent space projections, and automated clinical report export:

- **Windows 1-Click:** Double-click [`run_web_app.bat`](run_web_app.bat) *(or [`run_report_app.bat`](run_report_app.bat))*
- **Windows CLI (CMD / PowerShell):**
  ```cmd
  :: Launch on default port (8000) and open browser automatically
  python scripts\run_web.py

  :: Launch on custom port in headless mode
  python scripts\run_web.py --port 8080 --no-browser
  ```
- **Linux / macOS:**
  ```bash
  python3 scripts/run_web.py --port 8000
  ```

The application will launch at **`http://localhost:8000`** with:
- **12-Lead Diagnostic Waveform Tracing**: 4x3 layout with continuous Lead II rhythm strip & 1.0 mV square calibration pulse.
- **PTB-XL Record Explorer**: Live search and pagination across authentic clinical records.
- **Calibrated Superclass Classification**: Real-time probabilities and Youden-optimal F1 decision cutoffs.
- **Electrophysiological Biomarkers**: 24 interval and amplitude features compared against clinical reference ranges.
- **Multimodal Latent Space Explorer**: 3D t-SNE and PCA projections across 1056 dimensions.
- **One-Click Clinical Report Print**: Formatted for medical stationery printing and PDF archival (`Ctrl+P`).

---

## 🎯 Target Diagnostic Superclasses

The model performs multi-label prediction across the 5 primary PTB-XL superclasses with **Youden-calibrated decision thresholds**:

| Class | Clinical Pathology | Calibrated Threshold ($\tau$) | Macro ROC-AUC | F1-Score |
| :---: | :--- | :---: | :---: | :---: |
| **NORM** | Normal Sinus Rhythm | `0.53` (53.0%) | 0.932 | 0.865 |
| **MI** | Myocardial Infarction | `0.26` (26.0%) | 0.915 | 0.798 |
| **STTC** | ST/T-Wave Changes | `0.25` (25.0%) | 0.894 | 0.762 |
| **CD** | Conduction Disturbance (e.g. BBB) | `0.30` (30.0%) | 0.928 | 0.824 |
| **HYP** | Ventricular Hypertrophy | `0.33` (33.0%) | 0.887 | 0.741 |
| **MACRO** | **Macro Average Performance** | — | **0.911** | **0.798** |

---

## 📦 Model Weights & Artifacts Registry

All weights and transformation artifacts are stored locally in [`models/`](models/) and [`biomarkers/`](biomarkers/):

| Component | Path | Dimension / Type | Description |
| :--- | :--- | :--- | :--- |
| **Temporal Encoder** | `models/C5_full_dataset.pt` | 512-D | 1D ResNet with Squeeze-and-Excitation blocks |
| **Morphology Encoder** | `models/morphology_encoder_v1.pt` | 512-D | 2D Spectrogram ResNet for time-frequency dynamics |
| **Biomarker Encoder** | `biomarkers/attention_mlp_best.pt` | 32-D | Attention-MLP autoencoder over 24 clinical features |
| **Unified Classifier** | `models/classification_mlp.pt` | 1056-D $\to$ 128-D $\to$ 5 | Bottleneck joint representation + multi-label head |
| **Calibrated Thresholds**| `models/classification_mlp_thresholds.npy`| 5 values | Optimal F1 cutoffs calibrated via Youden's Index |
| **Biomarker Scaler** | `biomarkers/scaler.pkl` | RobustScaler | Normalization parameters for clinical features |
| **Biomarker Imputer** | `biomarkers/imputer.pkl` | IterativeImputer | Median missingness imputation model |

---

## 📂 Repository Structure

```
ECG_Encoder/
├── setup_and_run.bat           # 1-Click All-in-One Bootstrap (Setup -> Download -> Web App)
├── setup_and_run.sh            # 1-Click Bootstrap Shell Script for Linux/macOS
├── run_application.bat         # 1-Click Master Application Launcher (Interactive Menu)
├── run_engine.bat              # 1-Click Foundation Engine Demo & Verification
├── generate_report.bat         # 1-Click Individual Clinical Report Generator
├── run_web_app.bat             # 1-Click Web Application Server & Browser Launch
├── run_tests.bat               # 1-Click Engine Unit & Integration Test Suite
├── launch_web_app.bat          # Convenience Web App Launcher
├── run_report_app.bat          # Legacy Web App Launcher
├── requirements.txt            # Python dependencies
├── main.py                     # Root CLI entry point
│
├── scripts/                    # Core executable scripts
│   ├── setup_and_run.py        # All-in-one automated bootstrap & web app launcher
│   ├── download_data.py        # PhysioNet PTB-XL waveform downloader & local cache
│   ├── run_app.py              # Master application controller & CLI subcommands
│   ├── run_engine.py           # Engine runner (demo, info, test, predict, encode)
│   ├── generate_report.py      # Standalone individual report generator (HTML/JSON)
│   ├── run_web.py              # Web application HTTP server
│   ├── quick_predict.py        # Instant single-command prediction
│   ├── setup_and_run.bat       # Convenience bootstrap batch runner
│   ├── run_app.bat             # Convenience master launcher batch runner
│   ├── run_engine.bat          # Convenience engine batch runner
│   ├── generate_report.bat     # Convenience report generator batch runner
│   ├── run_web.bat             # Convenience web app batch runner
│   ├── run_demo.bat            # Quick demo batch runner
│   └── run_all_tests.bat       # Quick test batch runner
│
├── ecg_engine/                 # High-level engine facade (SOLID architecture)
│   ├── engine.py               # ECGEncoderEngine orchestration pipeline
│   ├── interfaces.py           # Dataclasses & type definitions
│   ├── preprocessor.py         # Signal standardization (12, 1000) & normalization
│   ├── biomarker_service.py    # 24 clinical feature extraction & masking
│   ├── classifier.py           # Multi-label classifier with calibrated thresholds
│   ├── fusion.py               # Multimodal tensor concatenation (1056-D)
│   └── wrappers.py             # Pretrained model wrappers
│
├── web_app/                    # Web application backend and static stationery assets
│   ├── server.py               # HTTP request handlers & REST API endpoints
│   ├── sample_manager.py       # PTB-XL sample cache & authentic WFDB loader
│   ├── llm_service.py          # Gemini API clinical interpreter & offline heuristics
│   └── static/                 # Clinical stationery HTML5/CSS/JS frontend
│       ├── index.html          # High-contrast clinical stationery report generator
│       ├── style.css           # Monochrome hospital design system
│       └── app.js              # Diagnostic canvas renderer & state management
│
├── models/                     # Pretrained PyTorch weights & decision thresholds
├── biomarkers/                 # Biomarker encoder architectures, scalers & imputers
├── morphology_encoder/         # 2D Spectrogram extraction & ResNet encoder
├── temporal_encoder/           # 1D ResNet-SE temporal sequence encoder
├── tests/                      # Automated unit and integration test suite
└── outputs/                    # Output directory for generated clinical reports
    └── reports/                # Saved HTML and JSON diagnostic reports
```

---

## 🧪 Testing & Quality Assurance

To verify system health and ensure all neural networks, preprocessors, and classifier thresholds function as expected:

- **Windows 1-Click:** Double-click [`run_tests.bat`](run_tests.bat) *(or [`scripts/run_all_tests.bat`](scripts/run_all_tests.bat))*
- **Windows CLI (CMD / PowerShell):**
  ```cmd
  .\run_tests.bat
  :: Or via python:
  python scripts\run_engine.py test
  ```
- **Linux / macOS:**
  ```bash
  python3 scripts/run_engine.py test
  ```

All unit tests in `tests/test_ecg_engine.py` will execute and report `PASSED`.

---

## 💡 Troubleshooting & FAQ

- **CUDA vs CPU:** If an NVIDIA GPU is available with CUDA installed, PyTorch will automatically utilize `cuda`. To explicitly run on CPU, pass `--device cpu` to any engine or report script.
- **Port In Use (Web App):** If port `8000` is occupied by another process, `scripts/run_web.py` automatically detects this and binds to the next available sequential port (e.g. `8001`, `8002`).
- **Windows Console Fonts:** The scripts use UTF-8 console output (`chcp 65001`) with ASCII progress bars to ensure clean rendering in standard Windows Command Prompt and PowerShell.