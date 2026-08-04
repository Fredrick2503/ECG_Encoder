import json
from pathlib import Path

def generate_notebook_02():
    notebook_path = Path("notebooks/02_temporal_representation_learning.ipynb")
    
    cells = []
    
    # Cell 1: Title
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# ECG Foundation Representation System\n",
            "## Phase 2 Research Workspace: Temporal Representation Learning\n",
            "\n",
            "This notebook demonstrates self-supervised pretraining strategies and supervised fine-tuning of the **Temporal Encoder**:\n",
            "1. **Reconstruction Learning**: Compressing full ECG inputs and decoding them.\n",
            "2. **Masked Autoencoder (MAE)**: Reconstructing randomly masked segments.\n",
            "3. **Contrastive Learning (SimCLR)**: Maximizing cosine similarity between augmented views of the same record using a temperature-scaled InfoNCE loss.\n",
            "4. **Supervised Fine-Tuning & Evaluation**: Multi-label classifier training, computing diagnostics metrics (Hamming Loss, Macro F1, Macro AUC).\n",
            "5. **Explainability**: Generating gradient-based saliency maps over leads and time."
        ]
    })
    
    # Cell 2: Imports
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import sys\n",
            "from pathlib import Path\n",
            "\n",
            "# Ensure project root is in path\n",
            "project_root = Path.cwd().parent\n",
            "if str(project_root) not in sys.path:\n",
            "    sys.path.append(str(project_root))\n",
            "\n",
            "import torch\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "from torch.utils.data import TensorDataset, DataLoader\n",
            "\n",
            "print(f\"Environment initialized. PyTorch version: {torch.__version__}\")"
        ]
    })

    # Cell 3: Data setup Markdown
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Setup Toy Dataset\n",
            "We generate synthetic 12-lead ECG signals ($N=128$, channels=12, length=1000) and multi-hot target labels ($5$ diagnostic categories) to simulate training loaders."
        ]
    })

    # Cell 4: Create dataloaders
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "np.random.seed(42)\n",
            "torch.manual_seed(42)\n",
            "\n",
            "# 128 samples, 12 leads, 1000 sequence steps\n",
            "signals = np.random.randn(128, 12, 1000).astype(np.float32)\n",
            "labels = np.random.randint(0, 2, (128, 5)).astype(np.float32)\n",
            "\n",
            "# Create splits\n",
            "train_ds = TensorDataset(torch.tensor(signals[:96]), torch.tensor(labels[:96]))\n",
            "val_ds = TensorDataset(torch.tensor(signals[96:]), torch.tensor(labels[96:]))\n",
            "\n",
            "train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)\n",
            "val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)\n",
            "\n",
            "print(f\"Train samples: {len(train_ds)} | Validation samples: {len(val_ds)}\")"
        ]
    })

    # Cell 5: Initialize Model Markdown
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Initialize Model Components\n",
            "We instantiate the `ECGBiLSTM` encoder and the companion `ECGReconstructionDecoder` (which maps the $256$-dimensional latent vector back to the original $12 \times 1000$ waveform space)."
        ]
    })

    # Cell 6: Initialize model code
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from temporal_encoder.encoder import ECGBiLSTM, ECGReconstructionDecoder\n",
            "\n",
            "model = ECGBiLSTM(input_size=12, hidden_size=128, num_layers=2, num_classes=5)\n",
            "decoder = ECGReconstructionDecoder(latent_dim=256, num_leads=12, signal_length=1000)\n",
            "\n",
            "print(\"Encoder architecture:\")\n",
            "print(model)\n",
            "print(\"\\nDecoder architecture:\")\n",
            "print(decoder)"
        ]
    })

    # Cell 7: SSL Reconstruction Markdown
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Pretraining Strategy 1: Reconstruction Learning\n",
            "Encodes the full unmasked signal and trains the model to minimize the reconstruction Mean Squared Error (MSE)."
        ]
    })

    # Cell 8: Reconstruction code
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from temporal_encoder.strategies import ReconstructionLearningStrategy\n",
            "from temporal_encoder.trainer import TemporalTrainer\n",
            "\n",
            "trainer = TemporalTrainer(model, lr=1e-3)\n",
            "recon_strategy = ReconstructionLearningStrategy()\n",
            "\n",
            "print(\"Starting Reconstruction pretraining...\")\n",
            "recon_history = trainer.fit(\n",
            "    train_loader=train_loader,\n",
            "    epochs=5,\n",
            "    is_pretraining=True,\n",
            "    strategy=recon_strategy,\n",
            "    decoder=decoder\n",
            ")\n",
            "\n",
            "plt.figure(figsize=(8, 4))\n",
            "plt.plot(recon_history[\"train_loss\"], marker='o', color=\"blue\")\n",
            "plt.title(\"Reconstruction Pretraining Loss\")\n",
            "plt.xlabel(\"Epoch\")\n",
            "plt.ylabel(\"MSE Loss\")\n",
            "plt.grid(True)\n",
            "plt.show()"
        ]
    })

    # Cell 9: MAE Markdown
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Pretraining Strategy 2: Masked Autoencoder (MAE)\n",
            "Masks $30\\%$ of the time-steps, processes visible signals, and decodes the masked regions, calculating MSE loss strictly over the masked points."
        ]
    })

    # Cell 10: MAE code
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from temporal_encoder.strategies import MaskedAutoencoderStrategy\n",
            "\n",
            "mae_strategy = MaskedAutoencoderStrategy(mask_ratio=0.3)\n",
            "mae_history = trainer.fit(\n",
            "    train_loader=train_loader,\n",
            "    epochs=5,\n",
            "    is_pretraining=True,\n",
            "    strategy=mae_strategy,\n",
            "    decoder=decoder\n",
            ")\n",
            "\n",
            "plt.figure(figsize=(8, 4))\n",
            "plt.plot(mae_history[\"train_loss\"], marker='s', color=\"orange\")\n",
            "plt.title(\"Masked Autoencoder Pretraining Loss\")\n",
            "plt.xlabel(\"Epoch\")\n",
            "plt.ylabel(\"Masked MSE Loss\")\n",
            "plt.grid(True)\n",
            "plt.show()"
        ]
    })

    # Cell 11: Contrastive Markdown
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Pretraining Strategy 3: Contrastive Learning (SimCLR)\n",
            "Applies Gaussian noise and scaling to create two views of each ECG signal, then projects their representation embeddings to a contrastive projection sphere and minimizes InfoNCE loss."
        ]
    })

    # Cell 12: Contrastive code
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from temporal_encoder.strategies import ContrastiveLearningStrategy\n",
            "\n",
            "contrastive_strategy = ContrastiveLearningStrategy(temperature=0.1, projection_dim=64, latent_dim=256)\n",
            "contrastive_history = trainer.fit(\n",
            "    train_loader=train_loader,\n",
            "    epochs=5,\n",
            "    is_pretraining=True,\n",
            "    strategy=contrastive_strategy\n",
            ")\n",
            "\n",
            "plt.figure(figsize=(8, 4))\n",
            "plt.plot(contrastive_history[\"train_loss\"], marker='^', color=\"green\")\n",
            "plt.title(\"Contrastive Pretraining Loss\")\n",
            "plt.xlabel(\"Epoch\")\n",
            "plt.ylabel(\"InfoNCE Loss\")\n",
            "plt.grid(True)\n",
            "plt.show()"
        ]
    })

    # Cell 13: Supervised Fine-Tuning Markdown
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Supervised Downstream Fine-Tuning\n",
            "We fine-tune the pretrained model end-to-end on multi-label targets using Binary Cross Entropy (BCE) with logits."
        ]
    })

    # Cell 14: Supervised training code
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "supervised_history = trainer.fit(\n",
            "    train_loader=train_loader,\n",
            "    val_loader=val_loader,\n",
            "    epochs=5,\n",
            "    is_pretraining=False\n",
            ")\n",
            "\n",
            "plt.figure(figsize=(8, 4))\n",
            "plt.plot(supervised_history[\"train_loss\"], label=\"Train Loss\", marker='o')\n",
            "plt.plot(supervised_history[\"val_loss\"], label=\"Val Loss\", marker='x')\n",
            "plt.title(\"Supervised Downstream Classification Fine-Tuning\")\n",
            "plt.xlabel(\"Epoch\")\n",
            "plt.ylabel(\"BCE Loss\")\n",
            "plt.legend()\n",
            "plt.grid(True)\n",
            "plt.show()"
        ]
    })

    # Cell 15: Evaluation Markdown
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 7. Model Evaluation\n",
            "We run predictions on the validation loader and compute subset accuracy, Hamming Loss, Macro F1, and Macro ROC-AUC."
        ]
    })

    # Cell 16: Evaluation code
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from temporal_encoder.predictor import TemporalPredictor\n",
            "from temporal_encoder.evaluator import TemporalEvaluator\n",
            "\n",
            "predictor = TemporalPredictor(model)\n",
            "val_probs = predictor.predict_proba(val_loader)\n",
            "\n",
            "# Get ground truth labels\n",
            "val_labels = []\n",
            "for _, lbls in val_loader:\n",
            "    val_labels.append(lbls.numpy())\n",
            "val_labels = np.concatenate(val_labels, axis=0)\n",
            "\n",
            "metrics = TemporalEvaluator.evaluate(val_labels, val_probs)\n",
            "print(\"Validation Metrics:\")\n",
            "for k, v in metrics.items():\n",
            "    print(f\"- {k}: {v:.4f}\")"
        ]
    })

    # Cell 17: Saliency Markdown
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 8. Saliency Interpretability\n",
            "We extract a gradient-based attribution map to highlight what leads and timestamps are most important for predicting the target diagnostic class."
        ]
    })

    # Cell 18: Saliency Code
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from temporal_encoder.explainer import TemporalSaliencyExplainer\n",
            "\n",
            "explainer = TemporalSaliencyExplainer(model)\n",
            "single_ecg = signals[0]\n",
            "\n",
            "# Compute saliency attribution map for Class 0\n",
            "saliency_map = explainer.explain(single_ecg, class_idx=0)\n",
            "\n",
            "plt.figure(figsize=(15, 6))\n",
            "plt.imshow(saliency_map, aspect='auto', cmap='hot', interpolation='nearest')\n",
            "plt.colorbar(label='Gradient Magnitude Attribution')\n",
            "plt.title(\"Lead-wise Temporal Saliency Map (Class 0)\")\n",
            "plt.xlabel(\"Time-steps\")\n",
            "plt.ylabel(\"Leads (0-11)\")\n",
            "plt.show()"
        ]
    })

    notebook_data = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.11"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook_data, f, indent=1)
        
    print(f"Notebook successfully generated at: {notebook_path}")

if __name__ == "__main__":
    generate_notebook_02()
