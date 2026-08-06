# Data Management Thesis Methodology & Implementation Notes

These notes summarize the methodology and implementation sections of the Data Management research for inclusion in the thesis document.

---

## Chapter 3: Methodology

### 3.1 Dataset Selection & Characteristics
The foundation representations are trained on the **PTB-XL dataset** (v1.0.3), a large, publicly available electrocardiography database containing 21,837 clinical 12-lead ECG records from 18,885 patients.
- **Resolution:** High-resolution signals are sampled at 500 Hz (length 5000), and low-resolution signals at 100 Hz (length 1000).
- **Labels:** Annotations are mapped to 5 super-classes:
  - `NORM` (Normal ECG)
  - `MI` (Myocardial Infarction)
  - `STTC` (ST/T Change)
  - `CD` (Conduction Disturbance)
  - `HYP` (Hypertrophy)
- **Train/Val/Test Split:** Standardized splitting partition using stratified folds:
  - **Training:** Folds 1–8 (17,418 records)
  - **Validation:** Fold 9 (2,183 records)
  - **Testing:** Fold 10 (2,198 records)
