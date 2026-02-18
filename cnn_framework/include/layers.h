/*
 * layers.h - Neural Network Layer Definitions
 *
 * Each layer is represented as a tagged union (Layer struct with a type enum).
 * Every layer stores its own weights/biases and cached activations needed
 * for backpropagation.
 */

#ifndef LAYERS_H
#define LAYERS_H

#include "tensor.h"

/* --------------------------------------------------------------------------
 * Layer type enum
 * -------------------------------------------------------------------------- */
typedef enum {
    LAYER_CONV2D,
    LAYER_MAXPOOL,
    LAYER_RELU,
    LAYER_SOFTMAX,
    LAYER_DENSE,
    LAYER_FLATTEN
} LayerType;

/* --------------------------------------------------------------------------
 * Conv2D parameters
 * -------------------------------------------------------------------------- */
typedef struct {
    int num_filters;    /* Number of output channels / kernels            */
    int kernel_size;    /* Square kernel side length                      */
    int stride;         /* Stride of the convolution                     */
    int padding;        /* Zero-padding added to each side               */
    int in_channels;    /* Number of input channels                      */
    int in_h, in_w;     /* Input spatial dimensions (cached on forward)  */

    Tensor *filters;    /* Shape: [num_filters, in_channels, kH, kW]    */
    Tensor *biases;     /* Shape: [num_filters]  (1D stored as 2D 1xN)  */
    Tensor *d_filters;  /* Gradient accumulators (same shape as filters) */
    Tensor *d_biases;   /* Gradient accumulators (same shape as biases)  */

    Tensor *input_cache; /* Cached input for backward pass               */
} Conv2DParams;

/* --------------------------------------------------------------------------
 * MaxPool parameters
 * -------------------------------------------------------------------------- */
typedef struct {
    int pool_size;      /* Square pool window side                       */
    int stride;         /* Stride of the pooling window                  */

    Tensor *input_cache; /* Cached input for backward pass               */
    Tensor *mask;        /* Indices of max values for gradient routing    */
} MaxPoolParams;

/* --------------------------------------------------------------------------
 * Dense (Fully Connected) parameters
 * -------------------------------------------------------------------------- */
typedef struct {
    int in_features;    /* Number of input features                      */
    int out_features;   /* Number of output features                     */

    Tensor *weights;    /* Shape: [in_features, out_features]            */
    Tensor *biases;     /* Shape: [1, out_features]                      */
    Tensor *d_weights;  /* Gradient accumulators                         */
    Tensor *d_biases;   /* Gradient accumulators                         */

    Tensor *input_cache; /* Cached input for backward pass               */
} DenseParams;

/* --------------------------------------------------------------------------
 * Flatten parameters
 * -------------------------------------------------------------------------- */
typedef struct {
    int batch_size;         /* Cached batch size                         */
    int channels, h, w;     /* Original shape before flattening          */
} FlattenParams;

/* --------------------------------------------------------------------------
 * ReLU parameters
 * -------------------------------------------------------------------------- */
typedef struct {
    Tensor *input_cache;    /* Cached input for backward (mask)          */
} ReLUParams;

/* --------------------------------------------------------------------------
 * Softmax parameters
 * -------------------------------------------------------------------------- */
typedef struct {
    Tensor *output_cache;   /* Cached softmax output for backward        */
} SoftmaxParams;

/* --------------------------------------------------------------------------
 * Generic Layer struct (tagged union)
 * -------------------------------------------------------------------------- */
typedef struct {
    LayerType type;
    union {
        Conv2DParams   conv2d;
        MaxPoolParams  maxpool;
        DenseParams    dense;
        FlattenParams  flatten;
        ReLUParams     relu;
        SoftmaxParams  softmax;
    } params;
} Layer;

/* --------------------------------------------------------------------------
 * Layer creation functions
 * -------------------------------------------------------------------------- */
Layer *layer_create_conv2d(int in_channels, int num_filters, int kernel_size,
                           int stride, int padding);
Layer *layer_create_maxpool(int pool_size, int stride);
Layer *layer_create_relu(void);
Layer *layer_create_softmax(void);
Layer *layer_create_dense(int in_features, int out_features);
Layer *layer_create_flatten(void);

/* --------------------------------------------------------------------------
 * Forward / Backward pass (dispatched by layer type)
 * -------------------------------------------------------------------------- */

/* Forward: takes ownership of nothing; returns newly allocated output tensor. */
Tensor *layer_forward(Layer *layer, const Tensor *input);

/* Backward: given d_output (gradient of loss w.r.t. this layer's output),
 * returns gradient of loss w.r.t. this layer's input (newly allocated).
 * Also accumulates parameter gradients inside the layer. */
Tensor *layer_backward(Layer *layer, const Tensor *d_output);

/* --------------------------------------------------------------------------
 * Per-layer forward/backward implementations
 * -------------------------------------------------------------------------- */

/* Conv2D */
Tensor *conv2d_forward(Conv2DParams *p, const Tensor *input);
Tensor *conv2d_backward(Conv2DParams *p, const Tensor *d_output);

/* MaxPool */
Tensor *maxpool_forward(MaxPoolParams *p, const Tensor *input);
Tensor *maxpool_backward(MaxPoolParams *p, const Tensor *d_output);

/* ReLU */
Tensor *relu_forward(ReLUParams *p, const Tensor *input);
Tensor *relu_backward(ReLUParams *p, const Tensor *d_output);

/* Softmax */
Tensor *softmax_forward(SoftmaxParams *p, const Tensor *input);
Tensor *softmax_backward(SoftmaxParams *p, const Tensor *d_output);

/* Dense */
Tensor *dense_forward(DenseParams *p, const Tensor *input);
Tensor *dense_backward(DenseParams *p, const Tensor *d_output);

/* Flatten */
Tensor *flatten_forward(FlattenParams *p, const Tensor *input);
Tensor *flatten_backward(FlattenParams *p, const Tensor *d_output);

/* --------------------------------------------------------------------------
 * Cleanup
 * -------------------------------------------------------------------------- */
void layer_free(Layer *layer);

/* --------------------------------------------------------------------------
 * Serialization (save/load weights)
 * -------------------------------------------------------------------------- */
int  layer_save(const Layer *layer, FILE *fp);
Layer *layer_load(FILE *fp);

#endif /* LAYERS_H */
