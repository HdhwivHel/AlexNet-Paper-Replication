# AlexNet (2012) – Paper Replication

## Overview

This implementation is based on the architecture described in the paper:

**Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton — _ImageNet Classification with Deep Convolutional Neural Networks_ (2012).**

AlexNet is one of the most influential convolutional neural networks in deep learning history and was the winning model of the **ImageNet Large Scale Visual Recognition Challenge (ILSVRC) 2012**. The architecture demonstrated that deep convolutional neural networks trained on GPUs could dramatically outperform traditional computer vision methods.

This repository provides **two ways to interact with the AlexNet replication**:

1. **Modular Python Implementation** – designed for reproducible experiments, training, and evaluation.
2. **Interactive Notebook Version** – designed for easier exploration, visualization, and experimentation.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/HdhwivHel/AlexNet-Paper-Replication
cd AlexNet-Paper-Replication
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Recommended Python version:

```
Python ≥ 3.10
```

---

## 1. Modular Implementation

The modular version is intended for **structured experiments and reproducibility**.
It separates the model architecture, dataset handling, configuration, and training scripts.

### Train the model

```bash
python train.py
```

Training parameters such as **epochs, batch size, and learning rate** can be modified in:

```
configs/config.yaml
```

### Evaluate the trained model

```bash
python evaluate.py
```

---

## 2. Notebook Implementation

The notebook (`main.ipynb`) provides a **step-by-step implementation of AlexNet** that is easier to follow and experiment with.

The notebook includes:

- Model construction
- Training loop
- Evaluation
- Visualization of training metrics
- Exploration of predictions on validation images

---

# Differences from Original AlexNet Paper

## 1. Architecture Differences

| Component                | Original Paper                     | This Implementation       | Reason                                        |
| ------------------------ | ---------------------------------- | ------------------------- | --------------------------------------------- |
| GPU Architecture         | Model split across **two GPUs**    | Single-GPU implementation | Modern GPUs have sufficient memory            |
| Output Layer             | `Linear(4096 → 1000)` for ImageNet | `Linear(4096 → 100)`      | Training performed on **ImageNet-100 subset** |
| Framework Implementation | Custom CUDA code                   | PyTorch implementation    | Modern deep learning framework                |

All other architectural components, including **Local Response Normalization (LRN)** and **dropout**, are preserved.

---

## 2. Training Differences

| Component              | Original Paper                                    | This Implementation        | Reason                              |
| ---------------------- | ------------------------------------------------- | -------------------------- | ----------------------------------- |
| Epochs                 | ~90 epochs                                        | **40 epochs**              | Smaller dataset (ImageNet-100)      |
| Learning Rate Schedule | LR reduced by 10× when validation error plateaued | **Constant learning rate** | Simpler training setup              |
| Hardware Setup         | Two GPUs                                          | Single GPU                 | Modern hardware capability          |
| Compilation            | Not available                                     | `torch.compile()` used     | Faster training with modern PyTorch |

---

## 3. Input Preprocessing Differences

| Component           | Original Paper                           | This Implementation             | Reason                                |
| ------------------- | ---------------------------------------- | ------------------------------- | ------------------------------------- |
| Training Crops      | Random 224/227 crops from resized images | `Resize(256) → RandomCrop(227)` | Standard torchvision pipeline         |
| Horizontal Flipping | Used                                     | Same                            | Data augmentation                     |
| Color Augmentation  | PCA color jitter                         | Not implemented                 | Rarely used in modern implementations |
| Tensor Conversion   | Custom preprocessing                     | `ToTensor()`                    | PyTorch pipeline                      |

---

## 4. Model Architecture (Identical Components)

Despite the implementation differences above, the following architectural properties remain identical to the original AlexNet design.

| Property       | Value                              |
| -------------- | ---------------------------------- |
| Input size     | 227 × 227                          |
| Conv1          | 96 filters, 11×11 kernel, stride 4 |
| Pool1          | 3×3 max pooling, stride 2          |
| Conv2          | 256 filters, 5×5 kernel, padding 2 |
| Pool2          | 3×3 max pooling, stride 2          |
| Conv3          | 384 filters, 3×3 kernel            |
| Conv4          | 384 filters, 3×3 kernel            |
| Conv5          | 256 filters, 3×3 kernel            |
| Pool3          | 3×3 max pooling, stride 2          |
| FC1            | 4096 neurons                       |
| FC2            | 4096 neurons                       |
| Output classes | 100                                |

Dropout is applied to the first two fully connected layers with:

```
Dropout probability = 0.5
```

---

# Training Configuration

| Parameter     | Value            |
| ------------- | ---------------- |
| Optimizer     | SGD              |
| Learning Rate | 0.01             |
| Momentum      | 0.9              |
| Weight Decay  | 0.0005           |
| Batch Size    | 128              |
| Epochs        | 40               |
| Loss Function | CrossEntropyLoss |

---

# Dataset

Training is performed on **ImageNet-100**, a subset of the ImageNet ILSVRC 2012 dataset.

Dataset statistics:

```
Number of classes: 100
Training images: ~130k
Validation images: ~5k
```

Images are resized and cropped to match the **227×227 input resolution expected by AlexNet**.

---

# Summary

This implementation preserves the **core AlexNet convolutional architecture**, including **ReLU activations, Local Response Normalization, overlapping pooling, and dropout**, while adopting modern deep learning practices such as:

- PyTorch implementation
- `torch.compile()` acceleration
- simplified training pipeline
- reduced dataset size

The goal of this project is to **faithfully reproduce the AlexNet architecture and training setup** while making the model accessible for experimentation on modern hardware.

---
