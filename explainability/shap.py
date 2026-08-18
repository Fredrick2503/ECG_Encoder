import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import shap

class SHAPExplainerWrapper:
    """
    Wraps the ECG biomarker preprocessing, encoder, and downstream classifier pipeline
    to compute and visualize instance-level SHAP explainability.
    """
    def __init__(self, model, imputer, scaler, classifiers, features, labels, device):
        self.model = model
        self.imputer = imputer
        self.scaler = scaler
        self.classifiers = classifiers
        self.features = features
        self.labels = labels
        self.device = device
        self.explainer = None

    def predict_pipeline(self, X_24d):
        """
        Takes raw 24-dimensional biomarker features (possibly containing NaNs),
        applies imputer, scaler, missingness mask, encoder, and downstream classifier.
        Returns diagnostic probabilities of shape (N, len(labels)).
        """
        # Ensure input is 2D numpy array
        if len(X_24d.shape) == 1:
            X_24d = X_24d.reshape(1, -1)
            
        # 1. Compute missingness mask M (1.0 if not NaN, 0.0 if NaN)
        M = (~np.isnan(X_24d)).astype(np.float32)
        
        # 2. Impute NaNs using the pre-trained imputer
        X_imputed = self.imputer.transform(X_24d)
        
        # 3. Scale using the pre-trained scaler
        X_scaled = self.scaler.transform(X_imputed)
        
        # 4. Concatenate scaled features and mask (48 dimensions)
        X_combined = np.hstack([X_scaled, M])
        
        # 5. Get encoder embeddings (32 dimensions)
        X_tensor = torch.tensor(X_combined, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            if hasattr(self.model, "loss_function"):  # VAE
                _, _, mu, _, _ = self.model(X_tensor)
                latent = mu.cpu().numpy()
            else:
                _, latent, _ = self.model(X_tensor)
                latent = latent.cpu().numpy()
                
        # 6. Compute classification probabilities using downstream classifiers
        probs = np.zeros((X_24d.shape[0], len(self.labels)))
        for idx in range(len(self.labels)):
            probs[:, idx] = self.classifiers[idx].predict_proba(latent)[:, 1]
            
        return probs

    def fit_explainer(self, background_data, n_samples=50):
        """
        Initializes the SHAP KernelExplainer with a sample of the background data.
        """
        # Sample background data to speed up KernelExplainer execution
        if len(background_data) > n_samples:
            background_sample = shap.sample(background_data, n_samples, random_state=42)
        else:
            background_sample = background_data
            
        self.explainer = shap.KernelExplainer(self.predict_pipeline, background_sample)

    def explain_instance(self, x_instance):
        """
        Generates prediction probabilities and SHAP explanation for a single instance.
        """
        if self.explainer is None:
            raise ValueError("Explainer is not initialized. Call fit_explainer first.")
            
        if len(x_instance.shape) == 1:
            x_instance = x_instance.reshape(1, -1)
            
        # Get predictions
        probs = self.predict_pipeline(x_instance)[0]
        
        # Calculate SHAP values (returns list of arrays for multi-output)
        shap_values_list = self.explainer.shap_values(x_instance)
        
        return probs, shap_values_list

    def plot_explanation(self, shap_values, x_instance, class_name, prob, save_path=None):
        """
        Creates a custom horizontal bar plot showing feature contributions for a specific class.
        """
        # shap_values is a 1D array of shape (24,) for this specific class
        if len(x_instance.shape) == 2:
            x_instance = x_instance.flatten()
            
        # Pair feature names with shap values and original values
        data = []
        for name, sv, val in zip(self.features, shap_values, x_instance):
            data.append({
                "feature": name,
                "shap_value": sv,
                "val_str": f"{val:.2f}" if not np.isnan(val) else "NaN"
            })
            
        df = pd.DataFrame(data)
        
        # Sort by absolute SHAP value
        df["abs_shap"] = df["shap_value"].abs()
        df = df.sort_values(by="abs_shap", ascending=True)  # ascending=True for horizontal bar plot
        
        # Colors: red for positive attribution, blue for negative attribution
        colors = ["#ff0d57" if val > 0 else "#1e88e5" for val in df["shap_value"]]
        
        plt.figure(figsize=(10, 8), dpi=150)
        
        # Grid lines
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        
        bars = plt.barh(df["feature"], df["shap_value"], color=colors, height=0.6)
        
        # Add labels to the bars
        for bar, val_str in zip(bars, df["val_str"]):
            width = bar.get_width()
            align = 'left' if width < 0 else 'right'
            offset = -5 if width < 0 else 5
            plt.annotate(
                f"({val_str})",
                xy=(width, bar.get_y() + bar.get_height() / 2),
                xytext=(offset, 0),
                textcoords="offset points",
                ha=align, va='center',
                fontsize=8, color="#555555"
            )
            
        plt.title(f"SHAP Explanation for Class: {class_name} (Prob: {prob:.4f})", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("SHAP Value (Feature Contribution)", fontsize=11, labelpad=10)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            print(f"SHAP plot saved to {save_path}")
        else:
            plt.show()
