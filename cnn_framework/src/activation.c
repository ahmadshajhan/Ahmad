/*
 * activation.c - Activation Functions (ReLU, Softmax)
 *
 * ReLU: element-wise max(0, x). Backward masks gradient where input <= 0.
 * Softmax: per-sample normalization for classification output.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include "../include/tensor.h"
#include "../include/layers.h"

/* ==========================================================================
 * ReLU
 * ========================================================================== */

/*
 * ReLU Forward: out[i] = max(0, input[i])
 * Works on any tensor shape (element-wise).
 */
Tensor *relu_forward(ReLUParams *p, const Tensor *input) {
    /* Cache input for backward */
    if (p->input_cache) tensor_free(p->input_cache);
    p->input_cache = tensor_clone(input);

    Tensor *output = tensor_clone(input);
    for (size_t i = 0; i < output->size; i++) {
        if (output->data[i] < 0.0f) {
            output->data[i] = 0.0f;
        }
    }
    return output;
}

/*
 * ReLU Backward: d_input[i] = d_output[i] if input[i] > 0, else 0
 *
 * The derivative of ReLU is a step function:
 *   d(ReLU)/dx = 1 if x > 0, 0 otherwise
 * So the gradient passes through unchanged where the input was positive.
 */
Tensor *relu_backward(ReLUParams *p, const Tensor *d_output) {
    Tensor *d_input = tensor_clone(d_output);
    const Tensor *cached = p->input_cache;

    for (size_t i = 0; i < d_input->size; i++) {
        if (cached->data[i] <= 0.0f) {
            d_input->data[i] = 0.0f;
        }
    }
    return d_input;
}

/* ==========================================================================
 * Softmax
 * ========================================================================== */

/*
 * Softmax Forward: applied per-sample (row-wise on a 2D tensor).
 * Input/Output: [batch_size, num_classes]
 *
 * softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
 *
 * The subtraction of max(x) is for numerical stability to prevent overflow.
 */
Tensor *softmax_forward(SoftmaxParams *p, const Tensor *input) {
    int batch = input->dims[0];
    int classes = input->dims[1];

    Tensor *output = tensor_clone(input);

    for (int n = 0; n < batch; n++) {
        float *row = &output->data[n * classes];

        /* Find max for numerical stability */
        float max_val = -FLT_MAX;
        for (int c = 0; c < classes; c++) {
            if (row[c] > max_val) max_val = row[c];
        }

        /* Compute exp(x - max) and sum */
        float sum = 0.0f;
        for (int c = 0; c < classes; c++) {
            row[c] = expf(row[c] - max_val);
            sum += row[c];
        }

        /* Normalize */
        for (int c = 0; c < classes; c++) {
            row[c] /= sum;
        }
    }

    /* Cache output for backward */
    if (p->output_cache) tensor_free(p->output_cache);
    p->output_cache = tensor_clone(output);

    return output;
}

/*
 * Softmax Backward
 *
 * When combined with categorical cross-entropy loss, the gradient
 * simplifies to: d_input = predictions - targets
 *
 * This simplified gradient is computed in categorical_cross_entropy_grad()
 * in training.c, so here we just pass through the gradient as-is
 * (the combined softmax+CCE gradient is already computed upstream).
 *
 * For a standalone softmax backward (Jacobian), the full formula is:
 *   dL/dx_i = sum_j [ dL/dy_j * dy_j/dx_i ]
 *   where dy_j/dx_i = y_i*(delta_ij - y_j)
 *
 * But since we always pair softmax with cross-entropy, we use the
 * simplified pass-through.
 */
Tensor *softmax_backward(SoftmaxParams *p, const Tensor *d_output) {
    (void)p; /* Unused -- gradient already combined with loss */
    return tensor_clone(d_output);
}
