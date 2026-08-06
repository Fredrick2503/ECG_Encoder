# Data Management Lessons Learned

This document logs critical research insights, environment constraints, and pipeline optimizations discovered during the development of the Data Management module.

---

## 1. Environment Optimization under Network/Data Constraints
- **Insight:** Downloading massive deep learning libraries (like PyTorch ~2GB) or full datasets (PTB-XL ~1.7GB) can fail or time out on limited connections (e.g., mobile data), leading to corrupt files (e.g., `BadZipFile` exceptions) and pipeline halts.
- **Solution:**
  - **Lightweight Verification Mode:** Implementing a lightweight download option that fetches only metadata CSVs and the first record of each resolution (high-res 500Hz, low-res 100Hz, total ~10MB) directly from the PhysioNet HTTP files index. This allowed verification of loaders, splitting logic, and raw record reading in seconds.
  - **Conditional Dependency Guarding:** Packaging modules (like `data_management/__init__.py`) with `try-except` blocks for PyTorch-dependent classes. This allows data engineers and researchers to run loading and preprocessing tasks without needing a full PyTorch environment.
  - **HTTP Range Resuming:** Modifying the downloader to use `requests` with `Range` headers to support resuming interrupted downloads instead of starting over.

---

## 2. Directory Renaming vs. Slow Copying on Single Drive
- **Insight:** When using Python's `shutil.move` on existing directories, it falls back to copying files recursively one-by-one. For large datasets like PTB-XL (containing over 43,000 files), this fallback takes several minutes.
- **Solution:** Clean target directories first, then perform an atomic `os.rename` operation of the source folders, reducing the operation time from minutes to milliseconds.

---

## 3. Fast Existence Checks on Slow Virtual Filesystems (OneDrive)
- **Insight:** Virtual filesystems like OneDrive hook into sequential file operations. Calling `Path.exists()` sequentially 21,837 times in a loop takes over 10-15 minutes, causing training start stalls.
- **Solution:** Execute a single `os.walk` of the directory to collect all existing filenames in a Python `set`, and then perform constant-time set membership checks in memory, reducing startup overhead to under 0.2 seconds.
