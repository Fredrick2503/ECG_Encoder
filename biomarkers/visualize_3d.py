import os
import sys
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Clustering3DVisualization")

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

SEED = 42
K_CLUSTERS = 5
SAMPLE_SIZE = 3000  # Size for 3D t-SNE to remain fast and clear

LABELS = ["NORM", "MI", "STTC", "CD", "HYP"]

def main():
    biomarkers_dir = project_root / "biomarkers"
    raw_csv = biomarkers_dir / "ecg_biomarkers_full.csv"
    
    if not raw_csv.exists():
        logger.error(f"Missing raw features file at {raw_csv}!")
        return
        
    logger.info(f"Loading raw features and labels from {raw_csv}...")
    df_raw = pd.read_csv(raw_csv)
    
    dominant_classes = []
    for _, row in df_raw[LABELS].iterrows():
        active = [c for c in LABELS if row[c] == 1]
        dominant_classes.append(active[0] if active else "OTHER")
    df_raw["dominant_class"] = dominant_classes
    
    models = ["attention_mlp", "beta_vae", "ft_transformer"]
    plots_dir = biomarkers_dir / "validation_plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Try importing plotly
    try:
        import plotly.express as px
        has_plotly = True
        logger.info("Plotly is available. Interactive HTML plots will be generated.")
    except ImportError:
        has_plotly = False
        logger.warning("Plotly is not available. Skipping interactive HTML plots.")
        
    for name in models:
        emb_csv = biomarkers_dir / f"embeddings_{name}.csv"
        if not emb_csv.exists():
            logger.warning(f"Embeddings file for {name} not found at {emb_csv}!")
            continue
            
        logger.info(f"Processing embeddings for {name}...")
        df_emb = pd.read_csv(emb_csv)
        
        df_merged = pd.merge(df_emb, df_raw[["record_id", "dominant_class"] + LABELS], on="record_id")
        latent_cols = [c for c in df_emb.columns if c.startswith("latent_")]
        X = df_merged[latent_cols].values
        
        # Run KMeans
        logger.info("Running K-Means (K=5)...")
        kmeans = KMeans(n_clusters=K_CLUSTERS, random_state=SEED, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        df_merged["cluster"] = cluster_labels
        df_merged["cluster_str"] = df_merged["cluster"].astype(str)
        
        # PCA 3D
        logger.info("Running PCA 3D...")
        pca = PCA(n_components=3, random_state=SEED)
        X_pca = pca.fit_transform(X)
        df_merged["PC1"] = X_pca[:, 0]
        df_merged["PC2"] = X_pca[:, 1]
        df_merged["PC3"] = X_pca[:, 2]
        
        # t-SNE 3D
        logger.info("Running t-SNE 3D on sample...")
        np.random.seed(SEED)
        sample_indices = np.random.choice(len(X), min(SAMPLE_SIZE, len(X)), replace=False)
        df_sample = df_merged.iloc[sample_indices].copy()
        X_sample = X[sample_indices]
        
        tsne = TSNE(n_components=3, random_state=SEED, perplexity=30)
        X_tsne = tsne.fit_transform(X_sample)
        df_sample["t-SNE 1"] = X_tsne[:, 0]
        df_sample["t-SNE 2"] = X_tsne[:, 1]
        df_sample["t-SNE 3"] = X_tsne[:, 2]
        
        # 1. Matplotlib static 3D plots
        # Matplotlib PCA 3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=cluster_labels, cmap="tab10", s=2, alpha=0.6)
        ax.set_title(f"PCA 3D Projection ({name.replace('_', ' ').title()})")
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.set_zlabel("PC 3")
        fig.colorbar(scatter, ax=ax, label="K-Means Cluster")
        plt.tight_layout()
        pca_png_path = plots_dir / f"clustering_3d_pca_{name}.png"
        plt.savefig(pca_png_path, dpi=150)
        plt.close()
        logger.info(f"Saved static 3D PCA plot to {pca_png_path}")
        
        # Matplotlib t-SNE 3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], c=df_sample["cluster"], cmap="tab10", s=4, alpha=0.7)
        ax.set_title(f"t-SNE 3D Projection (Sample, {name.replace('_', ' ').title()})")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_zlabel("t-SNE 3")
        fig.colorbar(scatter, ax=ax, label="K-Means Cluster")
        plt.tight_layout()
        tsne_png_path = plots_dir / f"clustering_3d_tsne_{name}.png"
        plt.savefig(tsne_png_path, dpi=150)
        plt.close()
        logger.info(f"Saved static 3D t-SNE plot to {tsne_png_path}")
        
        # 2. Plotly Interactive 3D plots
        if has_plotly:
            # PCA 3D Interactive
            logger.info("Generating Plotly PCA 3D plot...")
            fig_pca = px.scatter_3d(
                df_merged, x="PC1", y="PC2", z="PC3",
                color="cluster_str",
                hover_data=["record_id", "dominant_class"],
                title=f"PCA 3D Projection ({name.replace('_', ' ').title()})",
                labels={"cluster_str": "Cluster"},
                opacity=0.6
            )
            fig_pca.update_traces(marker=dict(size=2))
            pca_html_path = plots_dir / f"clustering_3d_pca_{name}.html"
            fig_pca.write_html(str(pca_html_path))
            logger.info(f"Saved interactive 3D PCA plot to {pca_html_path}")
            
            # t-SNE 3D Interactive
            logger.info("Generating Plotly t-SNE 3D plot...")
            fig_tsne = px.scatter_3d(
                df_sample, x="t-SNE 1", y="t-SNE 2", z="t-SNE 3",
                color="cluster_str",
                hover_data=["record_id", "dominant_class"],
                title=f"t-SNE 3D Projection (Sample, {name.replace('_', ' ').title()})",
                labels={"cluster_str": "Cluster"},
                opacity=0.7
            )
            fig_tsne.update_traces(marker=dict(size=3))
            tsne_html_path = plots_dir / f"clustering_3d_tsne_{name}.html"
            fig_tsne.write_html(str(tsne_html_path))
            logger.info(f"Saved interactive 3D t-SNE plot to {tsne_html_path}")

if __name__ == "__main__":
    main()
