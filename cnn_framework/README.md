# CNN Framework - Pure C Deep Learning

A comprehensive, production-grade deep learning framework written in **pure C (C99)** from scratch, designed for image detection tasks. No external ML libraries (TensorFlow, PyTorch, BLAS) are used -- only the C standard library.

## Architecture

```
cnn_framework/
├── include/
│   ├── tensor.h       # Tensor/Matrix engine (3D/4D support)
│   ├── layers.h       # Neural network layer definitions
│   ├── training.h     # Training mechanism (loss, optimizers, model)
│   └── image_io.h     # Image processing utilities
├── src/
│   ├── tensor.c       # Tensor operations (matmul, transpose, etc.)
│   ├── conv2d.c       # Conv2D layer (forward + backward)
│   ├── pooling.c      # MaxPooling layer
│   ├── activation.c   # ReLU and Softmax activations
│   ├── dense.c        # Fully connected layer
│   ├── flatten.c      # Flatten layer (4D -> 2D)
│   ├── layers.c       # Layer factory, dispatch, serialization
│   ├── training.c     # Model, loss, optimizers, training loop
│   ├── image_io.c     # Image I/O and dummy data generation
│   └── main.c         # Demo: build, train, save, load, infer
├── Makefile
└── README.md
```

## Features

### Tensor/Matrix Engine
- N-dimensional tensor struct (up to 4D for batch processing)
- Manual matrix multiplication with cache-friendly ikj loop order
- Transpose, element-wise add/multiply/scale operations
- Xavier/Glorot initialization
- Binary serialization (save/load)

### Neural Network Layers
- **Conv2D**: Sliding-window convolution with configurable filters, kernel size, stride, and padding
- **MaxPooling**: 2D max pooling with gradient routing via stored max indices
- **ReLU**: Element-wise rectified linear unit
- **Softmax**: Numerically stable per-sample normalization
- **Dense (Fully Connected)**: Standard matrix-multiply layer
- **Flatten**: Reshapes 4D conv output to 2D for dense layers

### Training
- **Forward propagation**: Dynamic sequential pass through all layers
- **Backpropagation**: Full chain-rule gradient computation for every layer
- **Loss**: Categorical Cross-Entropy with combined softmax gradient
- **Optimizers**: SGD and Adam (with bias-corrected moment estimates)
- **Mini-batch training**: Configurable batch size

### Model Persistence
- Save trained weights to binary file
- Load and restore model for inference
- Deterministic results after save/load cycle

## Building

```bash
# Standard build (optimized)
make

# Build and run demo
make run

# Debug build with AddressSanitizer
make debug

# Clean build artifacts
make clean
```

### Requirements
- GCC (or any C99-compatible compiler)
- GNU Make
- No external libraries needed (only `-lm` for math functions)

## Demo Output

The demo in `main.c` builds a CNN, trains on synthetic data, and demonstrates save/load:

```
Conv2D(1->4, 3x3) -> ReLU -> MaxPool(2x2) -> Flatten -> Dense(36->3) -> Softmax

Epoch  1/20  |  Loss: 1.134696
Epoch  2/20  |  Loss: 1.119932
...
Epoch 20/20  |  Loss: 1.054943

Inference on sample 0: [0.3107, 0.4294, 0.2599]
Predicted class: 1 (confidence: 0.4294)
```

## Conv2D Backpropagation Math

The backward pass for the convolutional layer computes three gradients (documented in detail in `src/conv2d.c`):

1. **Gradient w.r.t. biases**: `db[f] = sum over (n, i, j) of dL/dY[n, f, i, j]`
2. **Gradient w.r.t. filters**: `dW[f,c,kh,kw] = sum over (n, i, j) of dL/dY[n, f, i, j] * X[n, c, i*s+kh, j*s+kw]`
3. **Gradient w.r.t. input**: `dX[n,c,h,w] = sum over (f, kh, kw) of dL/dY[n, f, ...] * W[f, c, kh, kw]`

This is equivalent to a full convolution of the upstream gradient with 180-degree rotated filters.
