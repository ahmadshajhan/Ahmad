/*
 * training.h - Training Mechanism (Loss, Optimizers, Training Loop)
 *
 * Provides categorical cross-entropy loss, SGD and Adam optimizers,
 * and a sequential model abstraction for forward/backward/update cycles.
 */

#ifndef TRAINING_H
#define TRAINING_H

#include "layers.h"
#include "tensor.h"

/* --------------------------------------------------------------------------
 * Optimizer types
 * -------------------------------------------------------------------------- */
typedef enum {
    OPTIMIZER_SGD,
    OPTIMIZER_ADAM
} OptimizerType;

/* --------------------------------------------------------------------------
 * Adam state for a single parameter tensor
 * -------------------------------------------------------------------------- */
typedef struct {
    Tensor *m;  /* First moment estimate  */
    Tensor *v;  /* Second moment estimate */
} AdamState;

/* --------------------------------------------------------------------------
 * Optimizer configuration
 * -------------------------------------------------------------------------- */
typedef struct {
    OptimizerType type;
    float learning_rate;

    /* Adam-specific hyperparameters */
    float beta1;
    float beta2;
    float epsilon;
    int   timestep;

    /* Adam states: one pair (m, v) per trainable parameter tensor.
     * Allocated dynamically based on the model. */
    AdamState *states;
    int        num_states;
} Optimizer;

/* --------------------------------------------------------------------------
 * Sequential Model
 * -------------------------------------------------------------------------- */
typedef struct {
    Layer **layers;
    int     num_layers;
    int     capacity;
} Model;

/* --------------------------------------------------------------------------
 * Model lifecycle
 * -------------------------------------------------------------------------- */
Model *model_create(void);
void   model_add_layer(Model *model, Layer *layer);
void   model_free(Model *model);

/* --------------------------------------------------------------------------
 * Forward / Backward through the whole model
 * -------------------------------------------------------------------------- */

/* Run forward pass through all layers. Returns final output tensor. */
Tensor *model_forward(Model *model, const Tensor *input);

/* Run backward pass through all layers given loss gradient. */
void model_backward(Model *model, const Tensor *d_loss);

/* --------------------------------------------------------------------------
 * Loss functions
 * -------------------------------------------------------------------------- */

/* Categorical cross-entropy loss.
 * predictions: [batch_size, num_classes]  (softmax output)
 * targets:     [batch_size, num_classes]  (one-hot encoded)
 * Returns scalar loss value. */
float categorical_cross_entropy(const Tensor *predictions, const Tensor *targets);

/* Gradient of categorical cross-entropy w.r.t. softmax output.
 * For softmax + CCE combined, gradient simplifies to (predictions - targets).
 * Returns newly allocated gradient tensor. */
Tensor *categorical_cross_entropy_grad(const Tensor *predictions,
                                       const Tensor *targets);

/* --------------------------------------------------------------------------
 * Optimizer lifecycle
 * -------------------------------------------------------------------------- */
Optimizer *optimizer_create_sgd(float learning_rate);
Optimizer *optimizer_create_adam(float learning_rate, float beta1,
                                float beta2, float epsilon);
void optimizer_free(Optimizer *opt);

/* Initialize Adam states for all trainable parameters in the model. */
void optimizer_init_adam_states(Optimizer *opt, Model *model);

/* --------------------------------------------------------------------------
 * Parameter update
 * -------------------------------------------------------------------------- */

/* Apply one optimization step: update all trainable parameters using
 * their accumulated gradients, then zero the gradients. */
void optimizer_step(Optimizer *opt, Model *model);

/* Zero all parameter gradients in the model. */
void model_zero_grad(Model *model);

/* --------------------------------------------------------------------------
 * Training loop helper
 * -------------------------------------------------------------------------- */

/* Run one training epoch over the given data.
 * images:  [num_samples, channels, height, width]
 * labels:  [num_samples, num_classes]
 * batch_size: mini-batch size
 * Returns average loss for the epoch. */
float train_epoch(Model *model, Optimizer *opt,
                  const Tensor *images, const Tensor *labels,
                  int batch_size);

/* --------------------------------------------------------------------------
 * Model save / load
 * -------------------------------------------------------------------------- */
int   model_save(const Model *model, const char *filepath);
Model *model_load(const char *filepath);

#endif /* TRAINING_H */
