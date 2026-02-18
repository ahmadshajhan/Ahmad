/*
 * dense.c - Fully Connected (Dense) Layer Implementation
 *
 * Forward:  Y = X * W + b
 * Backward: dW = X^T * dY,  db = sum(dY, axis=0),  dX = dY * W^T
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/tensor.h"
#include "../include/layers.h"

/* --------------------------------------------------------------------------
 * Dense Forward Pass
 *
 * Input:  [batch_size, in_features]
 * Output: [batch_size, out_features]
 * -------------------------------------------------------------------------- */
Tensor *dense_forward(DenseParams *p, const Tensor *input) {
    int batch = input->dims[0];
    int out_f = p->out_features;

    /* Cache input for backward */
    if (p->input_cache) tensor_free(p->input_cache);
    p->input_cache = tensor_clone(input);

    /* output = input * weights   ([batch, in_f] x [in_f, out_f] = [batch, out_f]) */
    int out_dims[] = {batch, out_f};
    Tensor *output = tensor_create(2, out_dims);

    tensor_matmul(input, p->weights, output);

    /* Add bias to each row: output[n][j] += biases[j] */
    for (int n = 0; n < batch; n++) {
        for (int j = 0; j < out_f; j++) {
            output->data[n * out_f + j] += p->biases->data[j];
        }
    }

    return output;
}

/* --------------------------------------------------------------------------
 * Dense Backward Pass
 *
 * d_output: [batch_size, out_features]
 * Returns:  d_input [batch_size, in_features]
 *
 * Gradients:
 *   dW = X^T * d_output        [in_features, out_features]
 *   db = sum(d_output, axis=0)  [1, out_features]
 *   dX = d_output * W^T         [batch_size, in_features]
 * -------------------------------------------------------------------------- */
Tensor *dense_backward(DenseParams *p, const Tensor *d_output) {
    const Tensor *input = p->input_cache;
    int batch = input->dims[0];
    int in_f  = p->in_features;
    int out_f = p->out_features;

    /* --- dW = X^T * d_output --- */
    int xt_dims[] = {in_f, batch};
    Tensor *input_T = tensor_create(2, xt_dims);
    tensor_transpose(input, input_T);

    tensor_zeros(p->d_weights);
    tensor_matmul(input_T, d_output, p->d_weights);
    tensor_free(input_T);

    /* --- db = sum of d_output over batch dimension --- */
    tensor_zeros(p->d_biases);
    for (int n = 0; n < batch; n++) {
        for (int j = 0; j < out_f; j++) {
            p->d_biases->data[j] += d_output->data[n * out_f + j];
        }
    }

    /* --- dX = d_output * W^T --- */
    int wt_dims[] = {out_f, in_f};
    Tensor *weights_T = tensor_create(2, wt_dims);
    tensor_transpose(p->weights, weights_T);

    int dx_dims[] = {batch, in_f};
    Tensor *d_input = tensor_create(2, dx_dims);
    tensor_matmul(d_output, weights_T, d_input);
    tensor_free(weights_T);

    return d_input;
}
