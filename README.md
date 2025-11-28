# numpy-fashion-mnist-nn
## Project Overview
This repository contains a complete implementation of a two-layer, fully connected neural network (FCNN) built exclusively using the NumPy library. It solves the Fashion MNIST image classification problem.

**Goal:** To demonstrate a deep understanding of the core mathematical principles of machine learning, including matrix multiplication, activation functions, and gradient descent.

## Architecture
The network architecture is:
- **Input Layer:** 784 neurons (28x28 pixels)
- **Hidden Layer:** 256 neurons with **ReLU** activation
- **Output Layer:** 10 neurons with **Softmax** activation (for 10 classes)

## Performance
After 2000 iterations of training, the model achieves an accuracy of approximately **[Insert Your Final Test Accuracy Here, e.g., 82.50%]** on the test set.

## Key Implementations (The Impressive Part)
All core ML components are vectorized and implemented manually:

1.  **Forward Propagation:** Calculation of weighted sums and activations.
2.  **Backpropagation:** Implementation of the chain rule to calculate gradients.
3.  **Cross-Entropy Derivative:** Integration of the loss function derivative with the Softmax derivative (`dZ2 = A2 - one_hot_Y`).
4.  **Parameter Updates:** Standard Gradient Descent.
5.  **He Initialization:** Used to ensure stable weight scaling.

## How to Run Locally
1.  **Get the Data:** Download the `fashion-mnist_train.csv` and `fashion-mnist_test.csv` files from the Kaggle Fashion MNIST dataset page and place them in the correct input structure or modify the paths in `fashion_mnist_nn.py`.
2.  **Execute:**
    ```bash
    python fashion_mnist_nn.py
    ```
---
