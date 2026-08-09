import numpy as np
import pandas as pd
import torch
import time
from typing import Dict
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, silhouette_score, 
    davies_bouldin_score, calinski_harabasz_score, accuracy_score, f1_score, roc_auc_score
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer

class BiomarkerEvaluator:
    def __init__(self, device: torch.device = None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_reconstruction_metrics(self, original: np.ndarray, reconstructed: np.ndarray) -> Dict[str, float]:
        mse = mean_squared_error(original, reconstructed)
        mae = mean_absolute_error(original, reconstructed)
        rmse = np.sqrt(mse)
        
        # Huber Loss calculation (delta = 1.0)
        error = np.abs(original - reconstructed)
        huber_mask = error <= 1.0
        huber = np.mean(np.where(huber_mask, 0.5 * (error ** 2), error - 0.5))
        
        return {
            "MSE": float(mse),
            "MAE": float(mae),
            "RMSE": float(rmse),
            "Huber_Loss": float(huber)
        }

    def get_latent_space_metrics(self, embeddings: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Compute Silhouette, Davies-Bouldin, and Calinski-Harabasz scores on latent embeddings."""
        # Convert multi-hot label array to single class IDs by choosing the first class present (or a dummy if empty)
        # Choose the argmax as dominant class
        if labels is None or len(labels) == 0:
            return {"Silhouette_Score": np.nan, "Davies_Bouldin_Index": np.nan, "Calinski_Harabasz_Index": np.nan}
            
        class_labels = np.argmax(labels, axis=1)
        
        # Check if there are at least 2 distinct classes to compute clustering metrics
        unique_classes = np.unique(class_labels)
        if len(unique_classes) < 2:
            return {"Silhouette_Score": np.nan, "Davies_Bouldin_Index": np.nan, "Calinski_Harabasz_Index": np.nan}
            
        try:
            sil = silhouette_score(embeddings, class_labels)
            db = davies_bouldin_score(embeddings, class_labels)
            ch = calinski_harabasz_score(embeddings, class_labels)
            return {
                "Silhouette_Score": float(sil),
                "Davies_Bouldin_Index": float(db),
                "Calinski_Harabasz_Index": float(ch)
            }
        except Exception:
            return {"Silhouette_Score": np.nan, "Davies_Bouldin_Index": np.nan, "Calinski_Harabasz_Index": np.nan}

    def get_embedding_stability(self, original: np.ndarray, reconstructed: np.ndarray, embeddings: np.ndarray) -> Dict[str, float]:
        """Compute Cosine Similarity, reconstruction correlation, and reconstruction consistency."""
        # Pairwise cosine similarity between inputs and reconstructions
        from sklearn.metrics.pairwise import cosine_similarity
        cos_sims = []
        for i in range(len(original)):
            sim = cosine_similarity(original[i].reshape(1, -1), reconstructed[i].reshape(1, -1))[0, 0]
            cos_sims.append(sim)
        mean_cos_sim = np.mean(cos_sims)
        
        # Feature correlation between original and reconstructed
        correlations = []
        for i in range(original.shape[1]):
            std_orig = np.std(original[:, i])
            std_recon = np.std(reconstructed[:, i])
            if std_orig > 1e-6 and std_recon > 1e-6:
                corr = np.corrcoef(original[:, i], reconstructed[:, i])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
        mean_correlation = np.mean(correlations) if correlations else np.nan
        
        # Embedding standard deviation / stability (reconstruction consistency)
        recon_diff = np.mean(np.std(original - reconstructed, axis=0))
        
        return {
            "Reconstruction_Cosine_Similarity": float(mean_cos_sim),
            "Feature_Correlation": float(mean_correlation),
            "Reconstruction_Consistency_Std": float(recon_diff)
        }

    def get_downstream_metrics(self, train_embeddings: np.ndarray, test_embeddings: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Train a lightweight classifier on the embeddings and evaluate downstream performance."""
        # Simple One-vs-Rest Logistic Regression classifier
        clf = OneVsRestClassifier(LogisticRegression(max_iter=1000, random_state=42))
        try:
            clf.fit(train_embeddings, y_train)
            preds = clf.predict(test_embeddings)
            probs = clf.predict_proba(test_embeddings)
            
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average="macro", zero_division=0)
            
            try:
                auc = roc_auc_score(y_test, probs, average="macro")
            except Exception:
                auc = np.nan
                
            return {
                "Downstream_Accuracy": float(acc),
                "Downstream_F1_Score": float(f1),
                "Downstream_ROC_AUC": float(auc)
            }
        except Exception as e:
            print(f"Downstream evaluation failed: {e}")
            return {
                "Downstream_Accuracy": np.nan,
                "Downstream_F1_Score": np.nan,
                "Downstream_ROC_AUC": np.nan
            }

    def evaluate_model(self, model: torch.nn.Module, train_loader, test_loader, y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, any]:
        model.eval()
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Get all embeddings and reconstructions
        test_inputs = []
        test_reconstructed = []
        test_embeddings = []
        test_logits = []
        
        # Time inference
        start_time = time.time()
        with torch.no_grad():
            for batch_x, _ in test_loader:
                batch_x = batch_x.to(self.device)
                N = batch_x.size(1) // 2
                orig_x = batch_x[:, :N]
                
                # Check VAE vs Standard
                if hasattr(model, "loss_function"):  # VAE
                    reconstructed, z, mu, logvar, class_logits = model(batch_x)
                    latent = mu
                else:
                    reconstructed, latent, class_logits = model(batch_x)
                    
                test_inputs.append(orig_x.cpu().numpy())
                test_reconstructed.append(reconstructed.cpu().numpy())
                test_embeddings.append(latent.cpu().numpy())
                test_logits.append(class_logits.cpu().numpy())
                
        inference_time = (time.time() - start_time) / len(test_loader.dataset)
        
        test_inputs = np.concatenate(test_inputs, axis=0)
        test_reconstructed = np.concatenate(test_reconstructed, axis=0)
        test_embeddings = np.concatenate(test_embeddings, axis=0)
        test_logits = np.concatenate(test_logits, axis=0)
        
        # Get train embeddings for downstream task
        train_embeddings = []
        with torch.no_grad():
            for batch_x, _ in train_loader:
                batch_x = batch_x.to(self.device)
                if hasattr(model, "loss_function"):  # VAE
                    reconstructed, z, mu, logvar, class_logits = model(batch_x)
                    latent = mu
                else:
                    reconstructed, latent, class_logits = model(batch_x)
                train_embeddings.append(latent.cpu().numpy())
        train_embeddings = np.concatenate(train_embeddings, axis=0)
        
        # Calculate direct classification metrics
        # Apply sigmoid to logits to get probabilities
        test_probs = 1.0 / (1.0 + np.exp(-test_logits))
        test_preds = (test_probs >= 0.5).astype(float)
        
        direct_acc = accuracy_score(y_test, test_preds)
        direct_f1 = f1_score(y_test, test_preds, average="macro", zero_division=0)
        try:
            direct_auc = roc_auc_score(y_test, test_probs, average="macro")
        except Exception:
            direct_auc = np.nan
            
        direct_metrics = {
            "Direct_Accuracy": float(direct_acc),
            "Direct_F1_Score": float(direct_f1),
            "Direct_ROC_AUC": float(direct_auc)
        }
        
        # Calculate all metrics
        recon_metrics = self.get_reconstruction_metrics(test_inputs, test_reconstructed)
        latent_metrics = self.get_latent_space_metrics(test_embeddings, y_test)
        stability_metrics = self.get_embedding_stability(test_inputs, test_reconstructed, test_embeddings)
        downstream_metrics = self.get_downstream_metrics(train_embeddings, test_embeddings, y_train, y_test)
        
        results = {
            "num_parameters": num_params,
            "inference_time_per_sample": inference_time,
            **recon_metrics,
            **latent_metrics,
            **stability_metrics,
            **downstream_metrics,
            **direct_metrics
        }
        
        return results, test_embeddings, test_reconstructed, test_inputs
