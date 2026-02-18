/*
 * pooling.c - MaxPooling Layer Implementation
 *
 * Forward: 2D max pooling with configurable window size and stride.
 * Backward: routes gradient only to the max-value positions (winner-take-all).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

#include "../include/tensor.h"
#include "../include/layers.h"

/* --------------------------------------------------------------------------
 * MaxPool Forward Pass
 *
 * Input:  [batch, channels, H_in, W_in]
 * Output: [batch, channels, H_out, W_out]
 *
 * Also stores a mask tensor recording which element was the max in each
 * pooling window, needed for backpropagation.
 * -------------------------------------------------------------------------- */
Tensor *maxpool_forward(MaxPoolParams *p, const Tensor *input) {
    int batch = input->dims[0];
    int ch    = input->dims[1];
    int H_in  = input->dims[2];
    int W_in  = input->dims[3];
    int ps    = p->pool_size;
    int stride = p->stride;

    int H_out = (H_in - ps) / stride + 1;
    int W_out = (W_in - ps) / stride + 1;

    /* Cache input for backward */
    if (p->input_cache) tensor_free(p->input_cache);
    p->input_cache = tensor_clone(input);

    /* Allocate output */
    int out_dims[] = {batch, ch, H_out, W_out};
    Tensor *output = tensor_create(4, out_dims);

    /* Allocate mask (same shape as output, stores flattened index into the
     * pooling window of the max element) */
    if (p->mask) tensor_free(p->mask);
    p->mask = tensor_create(4, out_dims);

    for (int n = 0; n < batch; n++) {
        for (int c = 0; c < ch; c++) {
            for (int i = 0; i < H_out; i++) {
                for (int j = 0; j < W_out; j++) {
                    float max_val = -FLT_MAX;
                    int max_idx = 0;

                    for (int ph = 0; ph < ps; ph++) {
                        for (int pw = 0; pw < ps; pw++) {
                            int ih = i * stride + ph;
                            int iw = j * stride + pw;
                            float val = tensor_get4d(input, n, c, ih, iw);
                            if (val > max_val) {
                                max_val = val;
                                max_idx = ph * ps + pw;
                            }
                        }
                    }

                    tensor_set4d(output, n, c, i, j, max_val);
                    tensor_set4d(p->mask, n, c, i, j, (float)max_idx);
                }
            }
        }
    }

    return output;
}

/* --------------------------------------------------------------------------
 * MaxPool Backward Pass
 *
 * d_output: [batch, channels, H_out, W_out]
 * Returns:  d_input [batch, channels, H_in, W_in]
 *
 * The gradient flows only through the position that was the max during
 * the forward pass (all other positions get zero gradient).
 * -------------------------------------------------------------------------- */
Tensor *maxpool_backward(MaxPoolParams *p, const Tensor *d_output) {
    const Tensor *input = p->input_cache;

    int batch  = input->dims[0];
    int ch     = input->dims[1];
    int H_in   = input->dims[2];
    int W_in   = input->dims[3];
    int ps     = p->pool_size;
    int stride = p->stride;
    int H_out  = d_output->dims[2];
    int W_out  = d_output->dims[3];

    int din_dims[] = {batch, ch, H_in, W_in};
    Tensor *d_input = tensor_create(4, din_dims);

    for (int n = 0; n < batch; n++) {
        for (int c = 0; c < ch; c++) {
            for (int i = 0; i < H_out; i++) {
                for (int j = 0; j < W_out; j++) {
                    float dout = tensor_get4d(d_output, n, c, i, j);
                    int max_idx = (int)tensor_get4d(p->mask, n, c, i, j);
                    int ph = max_idx / ps;
                    int pw = max_idx % ps;
                    int ih = i * stride + ph;
                    int iw = j * stride + pw;

                    /* Route gradient only to the max position */
                    float *dx = &d_input->data[
                        ((n * ch + c) * H_in + ih) * W_in + iw];
                    *dx += dout;
                }
            }
        }
    }

    return d_input;
}
