import os
import sys
import time
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ClusteringValidation")

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Settings
SEED = 42
K_CLUSTERS = 5
SAMPLE_SIZE = 5000  # For fast silhouette and t-SNE computations

LABELS = ["NORM", "MI", "STTC", "CD", "HYP"]

def main():
    biomarkers_dir = project_root / "biomarkers"
    raw_csv = biomarkers_dir / "ecg_biomarkers_full.csv"
    
    if not raw_csv.exists():
        logger.error(f"Missing raw features file at {raw_csv}!")
        return
        
    logger.info(f"Loading raw features and labels from {raw_csv}...")
    df_raw = pd.read_csv(raw_csv)
    
    # Map each record to dominant diagnostic class
    logger.info("Determining dominant diagnostic class for each record...")
    dominant_classes = []
    for _, row in df_raw[LABELS].iterrows():
        active = [c for c in LABELS if row[c] == 1]
        dominant_classes.append(active[0] if active else "OTHER")
    df_raw["dominant_class"] = dominant_classes
    
    models = ["attention_mlp", "beta_vae", "ft_transformer"]
    results = {}
    
    plots_dir = biomarkers_dir / "validation_plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    for name in models:
        emb_csv = biomarkers_dir / f"embeddings_{name}.csv"
        if not emb_csv.exists():
            logger.warning(f"Embeddings file for {name} not found at {emb_csv}!")
            continue
            
        logger.info(f"Processing embeddings for {name}...")
        df_emb = pd.read_csv(emb_csv)
        
        # Ensure alignment on record_id
        df_merged = pd.merge(df_emb, df_raw[["record_id", "dominant_class"] + LABELS], on="record_id")
        
        latent_cols = [c for c in df_emb.columns if c.startswith("latent_")]
        X = df_merged[latent_cols].values
        
        # 1. Unsupervised K-Means clustering (K=5)
        logger.info("Running K-Means (K=5)...")
        kmeans = KMeans(n_clusters=K_CLUSTERS, random_state=SEED, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        df_merged["cluster"] = cluster_labels
        
        # 2. Evaluate clustering metrics
        logger.info("Calculating clustering metrics...")
        # Silhouette Score on a random sample to save time
        np.random.seed(SEED)
        sample_indices = np.random.choice(len(X), min(SAMPLE_SIZE, len(X)), replace=False)
        X_sample = X[sample_indices]
        sample_cluster_labels = cluster_labels[sample_indices]
        
        sil = silhouette_score(X_sample, sample_cluster_labels)
        ari = adjusted_rand_score(df_merged["dominant_class"], cluster_labels)
        nmi = normalized_mutual_info_score(df_merged["dominant_class"], cluster_labels)
        
        # 3. Dimensionality Reduction for Visualization
        logger.info("Running PCA...")
        pca = PCA(n_components=2, random_state=SEED)
        X_pca = pca.fit_transform(X)
        
        logger.info("Running t-SNE on sample...")
        X_tsne_sample = X[sample_indices]
        tsne = TSNE(n_components=2, random_state=SEED, perplexity=30)
        X_tsne = tsne.fit_transform(X_tsne_sample)
        
        # 4. Generate & Save Visualization Plots
        logger.info("Generating visualization plots...")
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # PCA Plot
        scatter_pca = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap="tab10", s=2, alpha=0.6)
        axes[0].set_title(f"PCA 2D Projection ({name.replace('_', ' ').title()})")
        axes[0].set_xlabel("PC 1")
        axes[0].set_ylabel("PC 2")
        fig.colorbar(scatter_pca, ax=axes[0], label="K-Means Cluster")
        
        # t-SNE Plot
        scatter_tsne = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=sample_cluster_labels, cmap="tab10", s=4, alpha=0.7)
        axes[1].set_title(f"t-SNE 2D Projection (Sample, {name.replace('_', ' ').title()})")
        axes[1].set_xlabel("t-SNE 1")
        axes[1].set_ylabel("t-SNE 2")
        fig.colorbar(scatter_tsne, ax=axes[1], label="K-Means Cluster")
        
        plt.tight_layout()
        plot_path = plots_dir / f"clustering_{name}.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        logger.info(f"Saved visualization plots to {plot_path}")
        
        # 5. Cluster-vs-Class Prevalence Analysis
        prevalence = []
        for c_idx in range(K_CLUSTERS):
            cluster_subset = df_merged[df_merged["cluster"] == c_idx]
            prevalence_row = {"Cluster": c_idx, "Size": len(cluster_subset)}
            for lbl in LABELS:
                # Calculate percentage of members with this label
                prevalence_row[lbl] = f"{(cluster_subset[lbl].mean() * 100.0):.2f}%"
            # Most common dominant class
            dom_counts = cluster_subset["dominant_class"].value_counts()
            prevalence_row["Dominant Class"] = dom_counts.index[0] if not dom_counts.empty else "None"
            prevalence.append(prevalence_row)
            
        df_prevalence = pd.DataFrame(prevalence)
        
        results[name] = {
            "Silhouette": sil,
            "ARI": ari,
            "NMI": nmi,
            "prevalence": df_prevalence,
            "plot_rel_path": f"validation_plots/clustering_{name}.png"
        }

    # Generate Report
    report_path = biomarkers_dir / "clustering_validation_report.md"
    logger.info(f"Writing clustering validation report to {report_path}...")
    with open(report_path, "w") as f:
        f.write("# ECG Biomarker Embedding Clustering Validation Report\n\n")
        f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Executive Summary\n\n")
        f.write("We performed unsupervised clustering validation on the 32-dimensional latent representation spaces ")
        f.write("learned by **Attention MLP**, **Beta-VAE**, and **FT-Transformer** autoencoders. ")
        f.write("Using K-Means clustering (K=5) without providing diagnostic labels during training or clustering, ")
        f.write("we evaluate how well the latent representations naturally segregate based on the underlying physiological ECG classes.\n\n")
        
        f.write("## 1. Unsupervised Clustering Metrics Comparison\n\n")
        f.write("| Model Type | Silhouette Score (K=5) | Adjusted Rand Index (ARI) | Normalized Mutual Info (NMI) |\n")
        f.write("| --- | --- | --- | --- |\n")
        for name in models:
            if name in results:
                res = results[name]
                f.write(f"| {name.replace('_', ' ').title()} | {res['Silhouette']:.6f} | {res['ARI']:.6f} | {res['NMI']:.6f} |\n")
        f.write("\n")
        
        f.write("> [!NOTE]\n")
        f.write("> **Metric Interpretations**:\n")
        f.write("> - **Silhouette Score**: Measures cluster compactness and separation. Higher means clusters are better defined.\n")
        f.write("> - **Adjusted Rand Index (ARI)**: Measures agreement between clustering and actual dominant diagnostic classes (corrected for chance). 0 represents random alignment, 1 is perfect alignment.\n")
        f.write("> - **Normalized Mutual Information (NMI)**: Measures mutual information scaling between clusters and labels. Higher NMI indicates stronger alignment of latent clustering with medical diagnoses.\n\n")
        
        f.write("## 2. Cluster-to-Class Prevalence Breakdown\n\n")
        for name in models:
            if name in results:
                f.write(f"### {name.replace('_', ' ').title()} Clusters\n\n")
                df_prev = results[name]["prevalence"]
                f.write("| Cluster | Size | NORM | MI | STTC | CD | HYP | Primary Diagnostic Class |\n")
                f.write("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
                for _, row in df_prev.iterrows():
                    f.write(
                        f"| {row['Cluster']} | {row['Size']} | {row['NORM']} | {row['MI']} | "
                        f"{row['STTC']} | {row['CD']} | {row['HYP']} | {row['Dominant Class']} |\n"
                    )
                f.write("\n")
                
        f.write("## 3. Dimensionality Reduction Visualizations\n\n")
        f.write("Visualizations show 2D projections of the latent space color-coded by unsupervised K-Means clusters.\n\n")
        
        for name in models:
            if name in results:
                f.write(f"### {name.replace('_', ' ').title()} Latent Space\n\n")
                f.write(f"![{name} Clustering Projections]({results[name]['plot_rel_path']})\n\n")
                
        f.write("## 4. Diagnosis and Natural Separation Verdict\n\n")
        
        # Rank by ARI + NMI
        best_separation_model = "attention_mlp"
        best_score = -1.0
        for name in models:
            if name in results:
                score = results[name]["ARI"] + results[name]["NMI"]
                if score > best_score:
                    best_score = score
                    best_separation_model = name
                    
        f.write(f"Based on clustering validation, **{best_separation_model.replace('_', ' ').title()}** shows the best natural diagnostic separation in its unsupervised latent representations. ")
        f.write("Its latent coordinates group records in a way that correlates most cleanly with actual diagnostic labels, ")
        f.write("rendering it highly suitable for downstream linear probing and downstream clustering applications.\n")

    logger.info("Clustering validation report completed.")

if __name__ == "__main__":
    main()
