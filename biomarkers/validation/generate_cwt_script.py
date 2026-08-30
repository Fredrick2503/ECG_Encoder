import os
from pathlib import Path

def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    src_path = project_root / "biomarkers" / "extract_features.py"
    dst_path = project_root / "biomarkers" / "extract_features_cwt.py"
    
    with open(src_path, "r") as f:
        code = f.read()
        
    # Replace DWT with CWT for delineation
    code = code.replace('method="dwt"', 'method="cwt"')
    code = code.replace("using DWT", "using CWT")
    
    # Update default paths and settings
    code = code.replace('ROW_LIMIT = 4500', 'ROW_LIMIT = -1')
    code = code.replace('CSV_OUTPUT_PATH = OUTPUT_DIR / "ecg_biomarkers_4500.csv"', 'CSV_OUTPUT_PATH = OUTPUT_DIR / "ecg_biomarkers_full_cwt.csv"')
    code = code.replace('REPORT_OUTPUT_PATH = OUTPUT_DIR / "extraction_report.txt"', 'REPORT_OUTPUT_PATH = OUTPUT_DIR / "full_extraction_report_cwt.md"')
    code = code.replace('qc_logs.csv', 'qc_logs_cwt.csv')
    
    with open(dst_path, "w") as f:
        f.write(code)
        
    print(f"Created {dst_path}")

if __name__ == "__main__":
    main()
