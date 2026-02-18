/*
 * flatten.c - Flatten Layer Implementation
 *
 * Reshapes a 4D tensor [batch, channels, H, W] into 2D [batch, channels*H*W]
 * for feeding into Dense layers. Backward reverses the reshape.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/tensor.h"
#include "../include/layers.h"

/* --------------------------------------------------------------------------
 * Flatten Forward Pass
 *
 * Input:  [batch, channels, H, W]
 * Output: [batch, channels * H * W]
 *
 * Data is already contiguous in row-major order, so flattening is just
 * a reshape -- no data movement required, but we create a new tensor
 * with 2D shape for type safety.
 * -------------------------------------------------------------------------- */
Tensor *flatten_forward(FlattenParams *p, const Tensor *input) {
    int batch = input->dims[0];
    int ch    = input->dims[1];
    int h     = input->dims[2];
    int w     = input->dims[3];

    /* Cache original shape for backward */
    p->batch_size = batch;
    p->channels   = ch;
    p->h          = h;
    p->w          = w;

    int flat_size = ch * h * w;
    int out_dims[] = {batch, flat_size};
    Tensor *output = tensor_create(2, out_dims);

    /* Copy data (same layout, just different logical shape) */
    memcpy(output->data, input->data, input->size * sizeof(float));

    return output;
}

/* --------------------------------------------------------------------------
 * Flatten Backward Pass
 *
 * d_output: [batch, flat_size]
 * Returns:  d_input [batch, channels, H, W]
 *
 * Just reshape back to the original 4D shape.
 * -------------------------------------------------------------------------- */
Tensor *flatten_backward(FlattenParams *p, const Tensor *d_output) {
    int dims[] = {p->batch_size, p->channels, p->h, p->w};
    Tensor *d_input = tensor_create(4, dims);

    memcpy(d_input->data, d_output->data, d_output->size * sizeof(float));

    return d_input;
}
