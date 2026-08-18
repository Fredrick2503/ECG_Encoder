# ECG Foundation Representation System

This repository implements representation learning and classification for ECG data using clinical biomarkers and self-supervised encoders.

## SHAP Explainability Tool

The project features a SHAP explainability pipeline that provides instance-level feature attributions in terms of the original 24 clinical biomarkers (rather than the 32 latent embedding dimensions) to answer: **"Why did the model classify this specific ECG as this class?"**

The pipeline uses `shap.KernelExplainer` to estimate feature contributions across the entire prediction pipeline: imputation, scaling, missingness indicator concatenation, Attention MLP encoding, and downstream Logistic Regression classification.

### Requirements

To run explainability, ensure you have the required packages installed:
```bash
pip install -r requirements.txt
```

### Running Instance-level Explanations

You can run explainability on either a specific test set instance or by extracting biomarkers directly from a raw 12-lead ECG signal.

#### 1. Explain Test Set Instances Automatically
Run the script without arguments to evaluate and explain 3 representative test set instances (NORM, MI, and HYP/STTC):
```bash
python explain_instance.py
```
This prints the predicted probabilities and SHAP feature contributions to the console and saves explanation bar plots to `outputs/shap_explanations/`.

#### 2. Explain a Specific Test Set Record ID
You can specify a record ID (e.g., 3) from the test split to run the SHAP explanation:
```bash
python explain_instance.py --record-id 3
```

#### 3. Explain a Custom Test Set Index
You can explain an instance by its index inside the test split (e.g. index 18):
```bash
python explain_instance.py --test-index 18
```

#### 4. Specify Save Directory
To save the generated SHAP visualization plots to a custom folder:
```bash
python explain_instance.py --save-dir my_explanations_folder
```

### Outputs and Visualizations

For each explained instance, the script:
1. Reports the **Ground Truth** and **Predicted Class probabilities**.
2. Identifies the **Top Positive** (pushing the model towards this diagnosis) and **Top Negative** (pushing the model away from this diagnosis) contributing biomarker features.
3. Saves a horizontal bar plot showing the SHAP value for each of the 24 biomarkers, annotated with its raw feature value.