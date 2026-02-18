/*
 * layers.c - Layer Factory, Dispatch, and Serialization
 *
 * Central dispatch for layer_forward / layer_backward based on layer type.
 * Also handles layer creation, cleanup, and weight serialization.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/tensor.h"
#include "../include/layers.h"

/* --------------------------------------------------------------------------
 * Layer Creation
 * -------------------------------------------------------------------------- */

Layer *layer_create_conv2d(int in_channels, int num_filters, int kernel_size,
                           int stride, int padding) {
    Layer *l = (Layer *)calloc(1, sizeof(Layer));
    if (!l) return NULL;
    l->type = LAYER_CONV2D;

    Conv2DParams *p = &l->params.conv2d;
    p->num_filters  = num_filters;
    p->kernel_size  = kernel_size;
    p->stride       = stride;
    p->padding      = padding;
    p->in_channels  = in_channels;
    p->in_h = p->in_w = 0;
    p->input_cache  = NULL;

    /* Filters: [num_filters, in_channels, kernel_size, kernel_size] */
    int f_dims[] = {num_filters, in_channels, kernel_size, kernel_size};
    p->filters   = tensor_create(4, f_dims);
    p->d_filters = tensor_create(4, f_dims);

    /* Xavier initialization for filters */
    int fan_in  = in_channels * kernel_size * kernel_size;
    int fan_out = num_filters * kernel_size * kernel_size;
    tensor_xavier_init(p->filters, fan_in, fan_out);

    /* Biases: stored as 1D but using 2D tensor [1, num_filters] for compat */
    int b_dims[] = {1, num_filters};
    p->biases   = tensor_create(2, b_dims);
    p->d_biases = tensor_create(2, b_dims);
    /* biases start at zero (calloc) */

    return l;
}

Layer *layer_create_maxpool(int pool_size, int stride) {
    Layer *l = (Layer *)calloc(1, sizeof(Layer));
    if (!l) return NULL;
    l->type = LAYER_MAXPOOL;

    MaxPoolParams *p = &l->params.maxpool;
    p->pool_size    = pool_size;
    p->stride       = stride;
    p->input_cache  = NULL;
    p->mask         = NULL;

    return l;
}

Layer *layer_create_relu(void) {
    Layer *l = (Layer *)calloc(1, sizeof(Layer));
    if (!l) return NULL;
    l->type = LAYER_RELU;
    l->params.relu.input_cache = NULL;
    return l;
}

Layer *layer_create_softmax(void) {
    Layer *l = (Layer *)calloc(1, sizeof(Layer));
    if (!l) return NULL;
    l->type = LAYER_SOFTMAX;
    l->params.softmax.output_cache = NULL;
    return l;
}

Layer *layer_create_dense(int in_features, int out_features) {
    Layer *l = (Layer *)calloc(1, sizeof(Layer));
    if (!l) return NULL;
    l->type = LAYER_DENSE;

    DenseParams *p = &l->params.dense;
    p->in_features  = in_features;
    p->out_features = out_features;
    p->input_cache  = NULL;

    /* Weights: [in_features, out_features] */
    int w_dims[] = {in_features, out_features};
    p->weights   = tensor_create(2, w_dims);
    p->d_weights = tensor_create(2, w_dims);
    tensor_xavier_init(p->weights, in_features, out_features);

    /* Biases: [1, out_features] */
    int b_dims[] = {1, out_features};
    p->biases   = tensor_create(2, b_dims);
    p->d_biases = tensor_create(2, b_dims);

    return l;
}

Layer *layer_create_flatten(void) {
    Layer *l = (Layer *)calloc(1, sizeof(Layer));
    if (!l) return NULL;
    l->type = LAYER_FLATTEN;
    memset(&l->params.flatten, 0, sizeof(FlattenParams));
    return l;
}

/* --------------------------------------------------------------------------
 * Forward / Backward dispatch
 * -------------------------------------------------------------------------- */

Tensor *layer_forward(Layer *layer, const Tensor *input) {
    switch (layer->type) {
        case LAYER_CONV2D:   return conv2d_forward(&layer->params.conv2d, input);
        case LAYER_MAXPOOL:  return maxpool_forward(&layer->params.maxpool, input);
        case LAYER_RELU:     return relu_forward(&layer->params.relu, input);
        case LAYER_SOFTMAX:  return softmax_forward(&layer->params.softmax, input);
        case LAYER_DENSE:    return dense_forward(&layer->params.dense, input);
        case LAYER_FLATTEN:  return flatten_forward(&layer->params.flatten, input);
        default:
            fprintf(stderr, "ERROR: Unknown layer type %d\n", layer->type);
            return NULL;
    }
}

Tensor *layer_backward(Layer *layer, const Tensor *d_output) {
    switch (layer->type) {
        case LAYER_CONV2D:   return conv2d_backward(&layer->params.conv2d, d_output);
        case LAYER_MAXPOOL:  return maxpool_backward(&layer->params.maxpool, d_output);
        case LAYER_RELU:     return relu_backward(&layer->params.relu, d_output);
        case LAYER_SOFTMAX:  return softmax_backward(&layer->params.softmax, d_output);
        case LAYER_DENSE:    return dense_backward(&layer->params.dense, d_output);
        case LAYER_FLATTEN:  return flatten_backward(&layer->params.flatten, d_output);
        default:
            fprintf(stderr, "ERROR: Unknown layer type %d\n", layer->type);
            return NULL;
    }
}

/* --------------------------------------------------------------------------
 * Layer cleanup
 * -------------------------------------------------------------------------- */

void layer_free(Layer *layer) {
    if (!layer) return;

    switch (layer->type) {
        case LAYER_CONV2D: {
            Conv2DParams *p = &layer->params.conv2d;
            tensor_free(p->filters);
            tensor_free(p->biases);
            tensor_free(p->d_filters);
            tensor_free(p->d_biases);
            tensor_free(p->input_cache);
            break;
        }
        case LAYER_MAXPOOL: {
            MaxPoolParams *p = &layer->params.maxpool;
            tensor_free(p->input_cache);
            tensor_free(p->mask);
            break;
        }
        case LAYER_RELU: {
            tensor_free(layer->params.relu.input_cache);
            break;
        }
        case LAYER_SOFTMAX: {
            tensor_free(layer->params.softmax.output_cache);
            break;
        }
        case LAYER_DENSE: {
            DenseParams *p = &layer->params.dense;
            tensor_free(p->weights);
            tensor_free(p->biases);
            tensor_free(p->d_weights);
            tensor_free(p->d_biases);
            tensor_free(p->input_cache);
            break;
        }
        case LAYER_FLATTEN:
            /* No dynamically allocated members */
            break;
    }

    free(layer);
}

/* --------------------------------------------------------------------------
 * Serialization
 * -------------------------------------------------------------------------- */

int layer_save(const Layer *layer, FILE *fp) {
    if (!layer || !fp) return -1;

    /* Write layer type */
    fwrite(&layer->type, sizeof(LayerType), 1, fp);

    switch (layer->type) {
        case LAYER_CONV2D: {
            const Conv2DParams *p = &layer->params.conv2d;
            fwrite(&p->num_filters, sizeof(int), 1, fp);
            fwrite(&p->kernel_size, sizeof(int), 1, fp);
            fwrite(&p->stride, sizeof(int), 1, fp);
            fwrite(&p->padding, sizeof(int), 1, fp);
            fwrite(&p->in_channels, sizeof(int), 1, fp);
            tensor_save(p->filters, fp);
            tensor_save(p->biases, fp);
            break;
        }
        case LAYER_MAXPOOL: {
            const MaxPoolParams *p = &layer->params.maxpool;
            fwrite(&p->pool_size, sizeof(int), 1, fp);
            fwrite(&p->stride, sizeof(int), 1, fp);
            break;
        }
        case LAYER_DENSE: {
            const DenseParams *p = &layer->params.dense;
            fwrite(&p->in_features, sizeof(int), 1, fp);
            fwrite(&p->out_features, sizeof(int), 1, fp);
            tensor_save(p->weights, fp);
            tensor_save(p->biases, fp);
            break;
        }
        case LAYER_RELU:
        case LAYER_SOFTMAX:
        case LAYER_FLATTEN:
            /* No parameters to save */
            break;
    }

    return 0;
}

Layer *layer_load(FILE *fp) {
    if (!fp) return NULL;

    LayerType type;
    if (fread(&type, sizeof(LayerType), 1, fp) != 1) return NULL;

    Layer *layer = NULL;

    switch (type) {
        case LAYER_CONV2D: {
            int nf, ks, stride, pad, in_ch;
            fread(&nf, sizeof(int), 1, fp);
            fread(&ks, sizeof(int), 1, fp);
            fread(&stride, sizeof(int), 1, fp);
            fread(&pad, sizeof(int), 1, fp);
            fread(&in_ch, sizeof(int), 1, fp);

            layer = layer_create_conv2d(in_ch, nf, ks, stride, pad);
            /* Overwrite randomly-initialized weights with saved ones */
            tensor_free(layer->params.conv2d.filters);
            tensor_free(layer->params.conv2d.biases);
            layer->params.conv2d.filters = tensor_load(fp);
            layer->params.conv2d.biases  = tensor_load(fp);
            break;
        }
        case LAYER_MAXPOOL: {
            int ps, stride;
            fread(&ps, sizeof(int), 1, fp);
            fread(&stride, sizeof(int), 1, fp);
            layer = layer_create_maxpool(ps, stride);
            break;
        }
        case LAYER_DENSE: {
            int in_f, out_f;
            fread(&in_f, sizeof(int), 1, fp);
            fread(&out_f, sizeof(int), 1, fp);

            layer = layer_create_dense(in_f, out_f);
            tensor_free(layer->params.dense.weights);
            tensor_free(layer->params.dense.biases);
            layer->params.dense.weights = tensor_load(fp);
            layer->params.dense.biases  = tensor_load(fp);
            break;
        }
        case LAYER_RELU:
            layer = layer_create_relu();
            break;
        case LAYER_SOFTMAX:
            layer = layer_create_softmax();
            break;
        case LAYER_FLATTEN:
            layer = layer_create_flatten();
            break;
        default:
            fprintf(stderr, "ERROR: Unknown layer type %d in file\n", type);
            return NULL;
    }

    return layer;
}
