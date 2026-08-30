# ECG Foundation Representation System — Master Index of Reports, Documentation, and Logs

This document serves as a centralized registry of all research reports, benchmark results, engineering documentation, and runtime logs for the ECG Foundation Representation System.

---

## 1. Core Documentation

These documents cover overall methodology, system architecture, academic thesis notes, and findings.

*   [thesis_notes.md](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/docs/thesis_notes.md) — [[Open Folder](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/docs/)]
    *   **Description:** Academic documentation compiling methodology, model architectures (ECG-ResNet1D-SE and ECG-Transformer), loss functions (ASL, Sqrt-BCE), and decision threshold calibration. Contains summary tables for Benchmark Suites B and C.
*   [unified_representation_notes.md](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/docs/unified_representation_notes.md) — [[Open Folder](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/docs/)]
    *   **Description:** Validation and analysis notes for the joint multi-modal representation space produced by the temporal (ECGResNet1D-SE) and morphology (ResNet2D-Spectrogram) encoders. Includes clustering metrics (Silhouette, NMI, ARI) and geometric insights.
*   [README.md](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/README.md) — [[Open Folder](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/)]
    *   **Description:** The root-level project readme containing setup instructions, environment dependencies, and standard execution workflows.

### Architectural High-Level Design (HLD) & Low-Level Design (LLD)
*   **Data Management Module:**
    *   [HLD.md](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/docs/data_management/HLD.md) | [LLD.md](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/docs/data_management/LLD.md) — [[Open Folder](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/docs/data_management/)]
*   **Preprocessing Module:**
    *   [HLD.md](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/docs/preprocessing/HLD.md) | [LLD.md](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/docs/preprocessing/LLD.md) — [[Open Folder](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/docs/preprocessing/)]
*   **Temporal Encoder Module:**
    *   [HLD.md](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/docs/temporal_encoder/HLD.md) | [LLD.md](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/docs/temporal_encoder/LLD.md) — [[Open Folder](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/docs/temporal_encoder/)]

---

## 2. Benchmark & Validation Reports

These files reside in `outputs/reports/` and capture the quantitative findings of experiment sweeps and validation benchmarks.

*   **Benchmark Suites Results:**
    *   [benchmark_b_results.md](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/reports/benchmark_b_results.md) | [CSV](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/reports/benchmark_b_results.csv) — [[Open Folder](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/reports/)]
        *   *Baseline benchmarking comparing Fixed vs. Optimized per-class decision thresholds.*
    *   [benchmark_c_results.md](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/reports/benchmark_c_results.md) | [CSV](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/reports/benchmark_c_results.csv) — [[Open Folder](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/reports/)]
        *   *Imbalance mitigation experiments covering class weighting, oversampling, and class-balanced loss.*
    *   [benchmark_d_results.md](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/reports/benchmark_d_results.md) | [CSV](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/reports/benchmark_d_results.csv) — [[Open Folder](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/reports/)]
        *   *Benchmark results evaluating robustness against domain shifts and synthetic signal transformations.*
    *   [benchmark_e_results.md](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/reports/benchmark_e_results.md) | [CSV](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/reports/benchmark_e_results.csv) — [[Open Folder](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/reports/)]
        *   *Results for Benchmark Suite E, focusing on advanced representation scaling.*

*   **Validation & Domain Adaptation Reports:** — [[Open Folder](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/reports/)]
    *   [comprehensive_validation_report.md](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/reports/comprehensive_validation_report.md) — *Comprehensive model metrics audit across multiple evaluation splits.*
    *   [external_validation_report.md](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/reports/external_validation_report.md) — *Generalization assessment on external datasets to measure out-of-distribution performance.*
    *   [fine_grained_validation_report.md](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/reports/fine_grained_validation_report.md) — *Segmented performance metrics detailed by specific leads and demographic cohorts.*
    *   [robustness_validation_report.md](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/reports/robustness_validation_report.md) — *Validation of model performance under adversarial noise and signal distortions.*
    *   [severity_robustness_report.md](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/reports/severity_robustness_report.md) | [CSV](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/reports/severity_robustness_report.csv) — *Evaluates representation resilience as ECG abnormalities scale in clinical severity.*
    *   [domain_shift_forensics_report.md](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/reports/domain_shift_forensics_report.md) — *Forensic audit of covariate and target shift across validation sets.*

*   **Model Baselines & Training Reports:** — [[Open Folder](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/reports/)]
    *   [model_b_baseline_locked.md](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/reports/model_b_baseline_locked.md) — *Locked model configuration and performance metrics for the baseline Suite B model.*
    *   [resnet_training_report.md](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/reports/resnet_training_report.md) — *Detailed logs and training convergence reports for the ResNet architecture.*
    *   [classification_engine_report.md](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/reports/classification_engine_report.md) — *Engine-level log of the classification layers' decision boundary optimization.*
    *   [final_lock_evaluation.md](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/reports/final_lock_evaluation.md) — *Final benchmark metrics for the locked production candidate.*
    *   [experiments_comparison_report.md](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/reports/experiments_comparison_report.md) — *High-level comparative summary of validation metrics across major training phases.*
    *   [sweep_report.md](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/reports/sweep_report.md) — *Summary report mapping metrics across hyperparameter grids.*

*   **Fusion, Morphology, & Representation Verdicts:** — [[Open Folder](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/reports/)]
    *   [m15_final_verdict.md](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/reports/m15_final_verdict.md) | [HTML](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/reports/m15_final_verdict.html) — *Ultimate evaluation report validating joint representation alignment and the final fusion performance.*
    *   [f4_fusion_verdict.md](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/reports/f4_fusion_verdict.md) — *Verification verdict analyzing the early/late learned fusion mechanisms.*
    *   [morphology_verdict_report.md](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/reports/morphology_verdict_report.md) | [Cluster Validation CSV](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/reports/morphology_cluster_validation.csv) — *Specific performance validation of 2D spectrogram morphology encoders.*
    *   [scalogram_verdict_report.md](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/reports/scalogram_verdict_report.md) — *Performance review of scalogram-based representation models.*
    *   [representation_validation_results.md](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/reports/representation_validation_results.md) | [CSV](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/reports/representation_validation_results.csv) — *Metrics confirming distinction and generalization of representations in downstream linear probing.*
    *   [nonlinear_fusion_comparison.md](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/reports/nonlinear_fusion_comparison.md) — *Comparison study between linear aggregation and multi-layer perceptron (MLP) non-linear fusion.*

*   **Audit & Debugging Diagnostics:** — [[Open Folder](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/reports/)]
    *   [lead_dropout_audit.md](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/reports/lead_dropout_audit.md) | [Audit CSV](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/reports/lead_dropout_audit_results.csv) — *Audits Model B behavior when individual leads or lead groups (limb, augmented, chest) are dropped.*
    *   [hyp_collapse_investigation.md](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/reports/hyp_collapse_investigation.md) — *XAI diagnosis of Left Ventricular Hypertrophy (HYP) classification collapse under V5/chest-lead masking, validating Sokolow-Lyon diagnostic reliance.*
    *   [lead_dropout_training_report.md](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/reports/lead_dropout_training_report.md) | [Training CSV](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/reports/lead_dropout_training_results.csv) — *Evaluates Model B classification performance when trained with randomized lead masking over 5 seeds.*
    *   [noise_robust_training_report.md](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/reports/noise_robust_training_report.md) | [Training CSV](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/reports/noise_robust_training_results.csv) — *Evaluates Model B classification performance when trained with realistic signal noise augmentations.*
    *   [missing_biomarkers_report.md](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/reports/missing_biomarkers_report.md) — *Quantifies classification degradation under simulated biomarker missingness at test time.*
    *   [missingness_aware_biomarker_report.md](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/reports/missingness_aware_biomarker_report.md) | [Comparison CSV](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/reports/biomarker_encoder_comparison.csv) — *Presents evaluation results of the new missingness-aware biomarker encoder under varying missingness rates.*
    *   [modality_dropout_report.md](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/reports/modality_dropout_report.md) | [Results CSV](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/reports/modality_dropout_results.csv) — *Evaluates classification engine under all modalities aggregation slices permutations (T, M, B).*
    *   [multimodal_calibration_audit.md](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/reports/multimodal_calibration_audit.md) — *Examines confidence calibration curves (ECE, MCE) for fused multi-modal encoders.*
    *   [suite_d_phase_d0_diagnosis.md](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/reports/suite_d_phase_d0_diagnosis.md) — *Troubleshooting report mapping anomalous gradient updates in Suite D baseline training.*
    *   [xai_integration_report.md](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/reports/xai_integration_report.md) — *Validates integration of explainability methods (e.g., Integrated Gradients, SHAP) into representation layers.*
    *   [eda_class_distribution.md](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/reports/eda_class_distribution.md) — *Analyzes dataset class imbalance across the raw, train, val, and test splits.*
    *   [ensemble_per_class_metrics.md](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/reports/ensemble_per_class_metrics.md) — *Tracks per-class performance improvements achieved by model ensembling.*

---

## 3. Log Files

System logs tracking execution runs, training schedules, resource monitoring, and workflow automations.

*   [sync_experiment_notebook.log](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/logs/sync_experiment_notebook.log) — [[Open Folder](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/logs/)]
    *   **Description:** Logs updates and synchronization events mapping MLflow trials to Jupyter evaluation notebooks.
*   [monitor.log](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/logs/monitor.log) — [[Open Folder](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/logs/)]
    *   **Description:** Execution monitor logs recording continuous training status, system resources, and batch evaluations.
*   [resnet_monitor.log](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/logs/resnet_monitor.log) — [[Open Folder](file:///C:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/logs/)]
    *   **Description:** Process-specific monitoring logs for ResNet encoder training and evaluation stages.
