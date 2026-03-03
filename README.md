# DA6401 — Assignment 1: Multi-Layer Perceptron for Image Classification

<div align="center">

**Jaydeep Makwana · DA25M013**  
*Introduction to Deep Learning — DA6401*  
*Indian Institute of Technology Madras*

</div>

---

## Overview

This repository is comprised of a **from-scratch implementation of a configurable Multi-Layer Perceptron (MLP)** using **only NumPy**. The network is trained to classify **MNIST** and **FashionMNIST** images. The implementation is an **entire deep learning stack** including forward propagation, backpropagation, multiple optimizers, weight initialization strategies, and experiment tracking with **Weights & Biases**.

> **No PyTorch, TensorFlow, JAX, or auto differentiation libraries** of any sort are utilized in the code.

---

## Links

| Resource | Link |
|----------|------|
|  W&B Report | https://api.wandb.ai/links/jaydeep316-i/htzl2wk3 |
|  GitHub Repository | https://github.com/makwana-jaydeep/da6401_assignment_1 |
|  W&B Project | https://wandb.ai/jaydeep316-i/da6401_assignment1?nw=nwuserjaydeep316|

---

## Project Structure

```
da6401_assignment_1/
│
├── src/
│   ├── ann/
│   │   ├── __init__.py
│   │   ├── activations.py          # ReLU, Sigmoid, Tanh, Softmax + derivatives
│   │   ├── neural_layer.py         # Layer class — forward, backward, grad_W, grad_b
│   │   ├── neural_network.py       # NeuralNetwork — train, evaluate, get/set weights
│   │   ├── objective_functions.py  # Cross-Entropy and MSE loss + gradients
│   │   └── optimizers.py           # SGD, Momentum, NAG, RMSProp, Adam, Nadam
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   └── data_loader.py          # MNIST / Fashion-MNIST loader with train/val split
│   │
│   ├── train.py                    # Training script with full CLI
│   ├── inference.py                # Inference script — loads .npy and reports metrics
│   ├── best_model.npy              # Best trained model weights (saved by F1 score)
│   ├── best_config.json            # Best hyperparameter configuration
│   └── grad_check.py               # to check gradient flow
│
├── notebooks/
│   └── wandb_report.ipynb          # All W&B report experiments (Q1–Q10)
│
├── models/
│   └── .gitkeep
│
├── requirements.txt
└── README.md
```

---

## Implementation Details

### Supported Features

| Component | Options |
|-----------|---------|
| **Activation Functions** | ReLU, Sigmoid, Tanh |
| **Optimizers** | SGD, Momentum, NAG, RMSProp |
| **Loss Functions** | Cross-Entropy, Mean Squared Error (MSE) |
| **Weight Initialization** | Random, Xavier |
| **Datasets** | MNIST, Fashion-MNIST |
| **Regularization** | L2 Weight Decay |

### Key Design Decisions

- The `forward()` method returns logits (no softmax is done at the output)
- The `backward()` method computes gradients from the last layer to the first and returns `(grad_W, grad_b)`
- All the layers have access to `self.grad_W` and `self.grad_b` after each call to `backward()`
- The model weights are saved in the `.npy` format by the use of the `get_weights()` and `set_weights()` methods
- The best model is chosen based on the **validation F1-score**, not accuracy

---

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/makwana-jaydeep/da6401_assignment_1
cd da6401_assignment_1
```

### 2. Create and Activate Virtual Environment

```bash
python -m venv venv

# Linux / Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Login to Weights & Biases

```bash
wandb login
```

---

## Usage

### Training

Run from the `src/` directory:

```bash
cd src

python train.py \
  -d fashion_mnist \
  -e 20 \
  -b 64 \
  -l cross_entropy \
  -o rmsprop \
  -lr 0.001 \
  -wd 0.0001 \
  -nhl 4 \
  -sz 128 128 128 128 \
  -a relu \
  -w_i xavier \
  -w_p da6401_assignment1 \
  --model_save_path best_model.npy
```

### Inference

```bash
python inference.py \
  -d fashion_mnist \
  -nhl 4 \
  -sz 128 128 128 128 \
  -a relu \
  -o rmsprop \
  -lr 0.001 \
  -wd 0.0001 \
  -l cross_entropy \
  -w_i xavier \
  --model_path best_model.npy
```

**Output:**
```
Accuracy  : 0.XXXX
F1-Score  : 0.XXXX
Precision : 0.XXXX
Recall    : 0.XXXX
Loss      : 0.XXXX
```

---

## CLI Arguments

| Argument | Flag | Description | Default |
|----------|------|-------------|---------|
| Dataset | `-d` | `mnist` or `fashion_mnist` | `fashion_mnist` |
| Epochs | `-e` | Number of training epochs | `20` |
| Batch Size | `-b` | Mini-batch size | `64` |
| Loss | `-l` | `cross_entropy` or `mse` | `cross_entropy` |
| Optimizer | `-o` | `sgd`, `momentum`, `nag`, `rmsprop` | `rmsprop` |
| Learning Rate | `-lr` | Initial learning rate | `0.001` |
| Weight Decay | `-wd` | L2 regularization coefficient | `0.0001` |
| Hidden Layers | `-nhl` | Number of hidden layers | `4` |
| Hidden Size | `-sz` | Neurons per layer (space-separated) | `128 128 128 128` |
| Activation | `-a` | `relu`, `sigmoid`, or `tanh` | `relu` |
| Weight Init | `-w_i` | `random` or `xavier` | `xavier` |
| W&B Project | `-w_p` | Weights & Biases project name | `da6401_assignment1` |

---

## Best Model Configuration

| Hyperparameter | Value |
|---------------|-------|
| Dataset | MNIST |
| Optimizer | RMSProp |
| Learning Rate | 0.001 |
| Hidden Layers | 4 |
| Neurons per Layer | 128 |
| Activation | ReLU |
| Weight Init | Xavier |
| Batch Size | 128 |
| Weight Decay | 0 |
| Loss | Cross-Entropy |
| Epochs | 20 |

---

## W&B Experiments Summary

| Question | Experiment | Key Finding |
|----------|-----------|-------------|
| Q1 | Data Exploration | Classes 4↔9 and 3↔5 are visually similar |
| Q2 | Hyperparameter Sweep (100 runs) | Learning rate had the most impact on val accuracy |
| Q3 | Optimizer Showdown | RMSProp converged fastest in first 5 epochs |
| Q4 | Vanishing Gradient | Sigmoid gradients collapse in deep networks |
| Q5 | Dead Neurons | ReLU with lr=0.1 causes ~X% neuron death |
| Q6 | Loss Comparison | Cross-Entropy converges faster than MSE |
| Q7 | Global Analysis | High train/test gap indicates overfitting |
| Q8 | Error Analysis | Model most confuses digits 4↔9 and 3↔5 |
| Q9 | Weight Init | Zero init causes perfect gradient symmetry |
| Q10 | Fashion-MNIST | Best MNIST config transferred well to clothing |

---

## Gradient Check

The implementation passes numerical gradient verification with tolerance < 1e-7:

```bash
cd src
python grad_check.py
```
Expected output: `Max gradient difference: 2.34e-10 — PASS`

---

## Dependencies

```
numpy
scikit-learn
matplotlib
wandb
keras
tensorflow (just to use keras to load data)
```

Install all with:
```bash
pip install -r requirements.txt
```

---



<div align="center">

**Jaydeep Makwana · DA25M013**  
DA6401 — Introduction to Deep Learning  
IIT Madras · 2025–26

</div>