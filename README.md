# PyTorch-Reimplementation-of-Super-resolution-reconstruction-of-turbulent-flows-with-machine-learning

## Overview

This project is a PyTorch reimplementation of a deep learning model originally proposed in "Super-resolution reconstruction of turbulent flows with machine learning by Kai Fukami, Koji Fukagata, and Kunihiko Taira". The architecture was re-implemented from scratch based on the original paper and using a Keras-based reference implementation provided by Professor Kai Fukami [hDSC_MS.py](http://www.seas.ucla.edu/fluidflow/lib/hDSC_MS.py) — from UCLA FluidFlow Lab. 
> Fukami, K., Fukagata, K., & Taira, K. (2019). Super-resolution reconstruction of turbulent flows with machine learning. Journal of Fluid Mechanics, 870, 106–120. DOI: 10.1017/jfm.2019.238

While the original code was written in Keras, this version was developed using PyTorch to offer greater flexibility, easier integration with custom training loops, and better GPU handling in research environments.

This repository presents a physics-inspired convolutional neural network architecture for super-resolution reconstruction of turbulent 2D velocity fields. The model is trained to learn a mapping from irregularly sampled low-resolution (LR) flow fields to fully resolved high-resolution (HR) solutions.

The code includes data loading, model definition (DSM-MSM hybrid architecture), training routines, and qualitative evaluation functions. The approach is targeted at applications in data-driven turbulence modeling, flow reconstruction, and reduced-order modeling.

## Key Features

- Full reimplementation of the original model in PyTorch
- Based on the structure from the paper and Professor Fukami’s Keras code
- Trained on a different dataset (not the original one)
- Dataset obtained from a public repository (linked below)
- Modular and easy-to-modify codebase

## Dataset

Instead of using the dataset used in the original paper, this project uses a different dataset from a publicly available source:

- **Source repository**: [Link to dataset repository](https://github.com/Ali-AliAkbari/Diffusion-based-Fluid-Super-resolution)

## 🧠 Model Architecture

The proposed model is composed of two complementary modules:

### 🔹 **DSM (Downsampling Module)**  
Captures hierarchical flow features through a multi-stage encoder-decoder structure using:

- Multi-scale pooling
- Residual convolution blocks
- Feature concatenation and bilinear upsampling

### 🔹 **MSM (Multi-Scale Module)**  
Employs multiple convolutional pathways with varying kernel sizes (5×5, 9×9, 13×13) to aggregate multi-scale contextual information.

The outputs of DSM and MSM are concatenated and passed through a final convolution layer to reconstruct the HR field.

---

## 🔧 Code Structure

| File / Section      | Description                                           |
|---------------------|-------------------------------------------------------|
| `CustomDataset`     | Loads and preprocesses LR-HR paired flow fields       |
| `conv_block`        | Defines convolutional building blocks with BN+ReLU    |
| `dsm`, `msm`        | Define DSM and MSM sub-networks                       |
| `full_model`        | Integrates DSM and MSM and produces final output      |
| `train_model`       | Training loop using MSE loss and learning rate decay  |
| `output`            | Visualization routine for comparing LR, output, and HR|


- **Loss Function**: Mean Squared Error (MSE)
- **Training Scheduler**: ReduceLROnPlateau (monitors loss)
- **Batch Size**: 32
- **Input / Output Size**: (1, 256, 256)
---

## 📊 Visualization Example

The `output()` function shows qualitative performance by plotting:

- Low-Resolution Input (LR)
  ![Low-Resolution Input](/Images/LR.png)
- Super-Resolved Prediction (Model Output)
  ![Super-Resolved Prediction](/Images/HR.png)
- High-Resolution Ground Truth (HR)
  ![High-Resolution Ground Truth](/Images/GT.png)
  
## Acknowledgements

- The model architecture and training methodology are based on the work presented in [Super-resolution reconstruction of turbulent flows with machine learning by Kai Fukami, Koji Fukagata, and Kunihiko Taira].
- Reference Keras implementation by Professor Kai Fukami was used as a guide [hDSC_MS.py](http://www.seas.ucla.edu/fluidflow/lib/hDSC_MS.py).
- Dataset from [Diffusion-based-Fluid-Super-resolution](https://github.com/Ali-AliAkbari/Diffusion-based-Fluid-Super-resolution).


