# Industrial Defect Detection using CNNs & Vision Transformers

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active%20Development-orange?style=flat-square)

---

## Overview

This project implements a production-ready deep learning pipeline for automated industrial defect classification, benchmarking **Convolutional Neural Networks (CNN)** against **Vision Transformers (ViT)**.

The goal is to establish a reproducible, modular, and scalable experimentation framework for vision-based defect inspection — applicable to real-world industrial quality control systems.

**Key objectives:**
- Benchmark CNN vs ViT architectures on defect classification tasks
- Analyze generalization behavior across architectural paradigms
- Build a modular pipeline adaptable to custom industrial datasets

---

## Repository Structure

```
industrial-defect-detection/
│
├── models/
│   ├── cnn_model.py          # Custom CNN architecture
│   └── vit_model.py          # Vision Transformer (ViT) with fine-tuned head
│
├── src/
│   ├── train.py              # Training loop and configuration
│   ├── evaluate.py           # Evaluation utilities and metrics
│   └── dataset.py            # Dataset loading and preprocessing
│
├── notebooks/
│   └── exploration.ipynb     # EDA and experiment visualization
│
├── assets/
│   └── model_comparison.png  # Architecture benchmark plots
│
├── requirements.txt
└── README.md
```

---

## Architecture

### Baseline: Custom CNN

A lightweight CNN built from scratch as a performance baseline:

- Stacked convolutional blocks with batch normalization
- Max pooling for spatial downsampling
- Dropout regularization
- Fully connected classification head

### Transformer: Vision Transformer (ViT)

Pretrained ViT backbone sourced via [`timm`](https://github.com/huggingface/pytorch-image-models):

- Backbone: `vit_base_patch16_224` (ImageNet-pretrained)
- Modified classification head for downstream defect classes
- Fine-tuned end-to-end for domain adaptation

---

## Pipeline Design

| Component | Description |
|---|---|
| `models/` | Modular model definitions (CNN + ViT) |
| `src/train.py` | Reproducible training loop with configurable hyperparameters |
| `src/evaluate.py` | Evaluation metrics and architecture comparison utilities |
| `src/dataset.py` | OpenCV-based preprocessing, augmentation, and data loading |
| GPU Support | CUDA-compatible training throughout |

---

## Dataset

Current baseline experiments use **CIFAR-10** (auto-downloaded via `torchvision`).

The pipeline is **dataset-agnostic** and designed for drop-in replacement with industrial datasets:

| Dataset | Domain | Notes |
|---|---|---|
| CIFAR-10 | General | Baseline benchmarking |
| [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) | Industrial anomaly detection | Primary target dataset |
| [PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset) | Agricultural defect detection | Transfer learning candidate |
| Custom industrial datasets | Domain-specific | Fully supported via custom DataLoader |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/industrial-defect-detection.git
cd industrial-defect-detection

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Requirements (`requirements.txt`):**

```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
opencv-python>=4.8.0
numpy>=1.24.0
matplotlib>=3.7.0
```

---

## Training

```bash
python src/train.py
```

**Configurable parameters:**

| Parameter | Default | Description |
|---|---|---|
| `--model` | `cnn` | Model architecture: `cnn` or `vit` |
| `--epochs` | `25` | Number of training epochs |
| `--batch_size` | `64` | Batch size |
| `--lr` | `1e-4` | Learning rate |
| `--device` | `cuda` | Training device: `cuda` or `cpu` |

**Example:**

```bash
# Train CNN baseline
python src/train.py --model cnn --epochs 30 --batch_size 64

# Fine-tune ViT
python src/train.py --model vit --epochs 20 --batch_size 32 --lr 1e-4
```

---

## Evaluation

```bash
python src/evaluate.py
```

**Reported metrics:**

- Classification accuracy (Top-1)
- Per-class precision, recall, F1-score
- Validation loss curve
- Architecture benchmark comparison (CNN vs ViT)

---

## Experiment Results

> Baseline experiments on CIFAR-10. Industrial dataset experiments in progress.

| Model | Backbone | Dataset | Val Accuracy | Notes |
|---|---|---|---|---|
| CNN | Custom | CIFAR-10 | ~72% | Baseline |
| ViT | `vit_base_patch16_224` | CIFAR-10 | ~85% | Pretrained, fine-tuned |

Ablation experiments and hyperparameter sweeps are ongoing.

---

## Tech Stack

| Library | Version | Purpose |
|---|---|---|
| PyTorch | 2.0+ | Core deep learning framework |
| Torchvision | 0.15+ | Dataset utilities and transforms |
| timm | 0.9+ | Pretrained ViT backbone |
| OpenCV | 4.8+ | Image preprocessing and augmentation |
| NumPy | 1.24+ | Numerical operations |
| Matplotlib | 3.7+ | Visualization and result plotting |

---

## Roadmap

- [x] CNN baseline architecture
- [x] ViT fine-tuning pipeline
- [x] CIFAR-10 benchmarking
- [ ] Integrate MVTec Anomaly Detection dataset
- [ ] Implement Grad-CAM for prediction interpretability
- [ ] Add experiment tracking (Weights & Biases / MLflow)
- [ ] REST API inference endpoint (FastAPI)
- [ ] Dockerize inference pipeline
- [ ] CI/CD for model evaluation on pull requests

---

## Contributing

Contributions, issues, and feature requests are welcome.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Contact

Built and maintained by **[Your Name]**  
[LinkedIn](https://linkedin.com/in/your-profile) · [GitHub](https://github.com/your-username) · [Email](mailto:your@email.com)

---

> *Designed as a modular foundation for industrial AI inspection systems — built for extensibility, reproducibility, and production readiness.*
