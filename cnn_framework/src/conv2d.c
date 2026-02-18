/*
 * conv2d.c - 2D Convolutional Layer Implementation
 *
 * Forward pass: sliding window convolution with multiple filters.
 * Backward pass: computes gradients w.r.t. filters, biases, and input.
 *
 * =========================================================================
 * MATH BEHIND CONV2D BACKPROPAGATION
 * =========================================================================
 *
 * Given:
 *   - Input X:   shape [batch, C_in, H_in, W_in]
 *   - Filters W: shape [C_out, C_in, kH, kW]
 *   - Biases b:  shape [C_out]
 *   - Output Y:  shape [batch, C_out, H_out, W_out]
 *
 * Forward:
 *   Y[n, f, i, j] = b[f] + SUM over c,kh,kw of
 *       W[f, c, kh, kw] * X[n, c, i*stride + kh, j*stride + kw]
 *
 * where H_out = (H_in + 2*pad - kH) / stride + 1
 *       W_out = (W_in + 2*pad - kW) / stride + 1
 *
 * Backward (given dL/dY, the upstream gradient):
 *
 * 1) Gradient w.r.t. biases (db):
 *    db[f] = SUM over n,i,j of dL/dY[n, f, i, j]
 *
 *    Each bias contributes to every spatial position in the output,
 *    so we simply sum the gradient across batch and spatial dims.
 *
 * 2) Gradient w.r.t. filters (dW):
 *    dW[f, c, kh, kw] = SUM over n,i,j of
 *        dL/dY[n, f, i, j] * X[n, c, i*stride + kh, j*stride + kw]
 *
 *    This is a correlation between the upstream gradient and the input.
 *    For each filter weight position, we sum the products of the
 *    gradient and the corresponding input patch across all samples
 *    and spatial positions.
 *
 * 3) Gradient w.r.t. input (dX):
 *    dX[n, c, h, w] = SUM over f,kh,kw of
 *        dL/dY[n, f, (h - kh) / stride, (w - kw) / stride]
 *            * W[f, c, kh, kw]
 *        (where the index into dL/dY must be valid and divisible)
 *
 *    Equivalently, this is a "full convolution" of the upstream
 *    gradient with the 180-degree rotated filters. In practice,
 *    we iterate over output positions and scatter gradients back.
 *
 * =========================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/tensor.h"
#include "../include/layers.h"

/* --------------------------------------------------------------------------
 * Helper: get padded input value (zero if out of bounds)
 * -------------------------------------------------------------------------- */
static inline float get_padded(const Tensor *input, int n, int c,
                               int h, int w, int pad) {
    int H = input->dims[2];
    int W = input->dims[3];
    int rh = h - pad;
    int rw = w - pad;
    if (rh < 0 || rh >= H || rw < 0 || rw >= W) return 0.0f;
    return tensor_get4d(input, n, c, rh, rw);
}

/* --------------------------------------------------------------------------
 * Conv2D Forward Pass
 *
 * Input:  [batch, in_channels, H_in, W_in]
 * Output: [batch, num_filters, H_out, W_out]
 * -------------------------------------------------------------------------- */
Tensor *conv2d_forward(Conv2DParams *p, const Tensor *input) {
    int batch      = input->dims[0];
    int in_ch      = input->dims[1];
    int H_in       = input->dims[2];
    int W_in       = input->dims[3];
    int num_f      = p->num_filters;
    int ks         = p->kernel_size;
    int stride     = p->stride;
    int pad        = p->padding;

    /* Cache input for backward */
    if (p->input_cache) tensor_free(p->input_cache);
    p->input_cache = tensor_clone(input);
    p->in_channels = in_ch;
    p->in_h        = H_in;
    p->in_w        = W_in;

    int H_out = (H_in + 2 * pad - ks) / stride + 1;
    int W_out = (W_in + 2 * pad - ks) / stride + 1;

    int out_dims[] = {batch, num_f, H_out, W_out};
    Tensor *output = tensor_create(4, out_dims);

    for (int n = 0; n < batch; n++) {
        for (int f = 0; f < num_f; f++) {
            /* Get bias for this filter */
            float bias = p->biases->data[f];

            for (int i = 0; i < H_out; i++) {
                for (int j = 0; j < W_out; j++) {
                    float sum = bias;

                    /* Convolve: sum over input channels and kernel window */
                    for (int c = 0; c < in_ch; c++) {
                        for (int kh = 0; kh < ks; kh++) {
                            for (int kw = 0; kw < ks; kw++) {
                                int ih = i * stride + kh;
                                int iw = j * stride + kw;
                                float x_val = get_padded(input, n, c, ih, iw, pad);
                                float w_val = tensor_get4d(p->filters, f, c, kh, kw);
                                sum += x_val * w_val;
                            }
                        }
                    }
                    tensor_set4d(output, n, f, i, j, sum);
                }
            }
        }
    }

    return output;
}

/* --------------------------------------------------------------------------
 * Conv2D Backward Pass
 *
 * d_output: [batch, num_filters, H_out, W_out]
 * Returns:  d_input [batch, in_channels, H_in, W_in]
 *
 * Also accumulates d_filters and d_biases.
 * -------------------------------------------------------------------------- */
Tensor *conv2d_backward(Conv2DParams *p, const Tensor *d_output) {
    const Tensor *input = p->input_cache;

    int batch  = input->dims[0];
    int in_ch  = p->in_channels;
    int H_in   = p->in_h;
    int W_in   = p->in_w;
    int num_f  = p->num_filters;
    int ks     = p->kernel_size;
    int stride = p->stride;
    int pad    = p->padding;
    int H_out  = d_output->dims[2];
    int W_out  = d_output->dims[3];

    /* Allocate d_input (gradient w.r.t. input) */
    int din_dims[] = {batch, in_ch, H_in, W_in};
    Tensor *d_input = tensor_create(4, din_dims);

    /* Zero gradient accumulators (they accumulate across the batch) */
    tensor_zeros(p->d_filters);
    tensor_zeros(p->d_biases);

    for (int n = 0; n < batch; n++) {
        for (int f = 0; f < num_f; f++) {
            for (int i = 0; i < H_out; i++) {
                for (int j = 0; j < W_out; j++) {
                    float dout = tensor_get4d(d_output, n, f, i, j);

                    /*
                     * (1) Gradient w.r.t. bias:
                     *     db[f] += dL/dY[n, f, i, j]
                     */
                    p->d_biases->data[f] += dout;

                    for (int c = 0; c < in_ch; c++) {
                        for (int kh = 0; kh < ks; kh++) {
                            for (int kw = 0; kw < ks; kw++) {
                                int ih = i * stride + kh;
                                int iw = j * stride + kw;

                                float x_val = get_padded(input, n, c, ih, iw, pad);

                                /*
                                 * (2) Gradient w.r.t. filter weight:
                                 *     dW[f,c,kh,kw] += dout * X[n,c,ih,iw]
                                 */
                                float *dw = &p->d_filters->data[
                                    ((f * in_ch + c) * ks + kh) * ks + kw];
                                *dw += dout * x_val;

                                /*
                                 * (3) Gradient w.r.t. input:
                                 *     dX[n,c,rh,rw] += dout * W[f,c,kh,kw]
                                 *
                                 * Only if the position maps back to a valid
                                 * (unpadded) input location.
                                 */
                                int rh = ih - pad;
                                int rw = iw - pad;
                                if (rh >= 0 && rh < H_in && rw >= 0 && rw < W_in) {
                                    float w_val = tensor_get4d(p->filters, f, c, kh, kw);
                                    float *dx = &d_input->data[
                                        ((n * in_ch + c) * H_in + rh) * W_in + rw];
                                    *dx += dout * w_val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return d_input;
}
