/*
 * training.c - Training Mechanism Implementation
 *
 * Implements:
 *   - Sequential model (add layers, forward, backward)
 *   - Categorical Cross-Entropy loss + gradient
 *   - SGD and Adam optimizers
 *   - Training loop with mini-batch support
 *   - Model save/load
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../include/tensor.h"
#include "../include/layers.h"
#include "../include/training.h"

/* --------------------------------------------------------------------------
 * Model lifecycle
 * -------------------------------------------------------------------------- */

Model *model_create(void) {
    Model *m = (Model *)malloc(sizeof(Model));
    if (!m) return NULL;
    m->capacity   = 16;
    m->num_layers = 0;
    m->layers     = (Layer **)malloc(sizeof(Layer *) * (size_t)m->capacity);
    if (!m->layers) { free(m); return NULL; }
    return m;
}

void model_add_layer(Model *model, Layer *layer) {
    if (model->num_layers >= model->capacity) {
        model->capacity *= 2;
        model->layers = (Layer **)realloc(model->layers,
                                          sizeof(Layer *) * (size_t)model->capacity);
    }
    model->layers[model->num_layers++] = layer;
}

void model_free(Model *model) {
    if (!model) return;
    for (int i = 0; i < model->num_layers; i++) {
        layer_free(model->layers[i]);
    }
    free(model->layers);
    free(model);
}

/* --------------------------------------------------------------------------
 * Model forward pass
 * -------------------------------------------------------------------------- */

Tensor *model_forward(Model *model, const Tensor *input) {
    Tensor *current = tensor_clone(input);

    for (int i = 0; i < model->num_layers; i++) {
        Tensor *next = layer_forward(model->layers[i], current);
        tensor_free(current);
        current = next;
    }

    return current;
}

/* --------------------------------------------------------------------------
 * Model backward pass
 * -------------------------------------------------------------------------- */

void model_backward(Model *model, const Tensor *d_loss) {
    Tensor *grad = tensor_clone(d_loss);

    for (int i = model->num_layers - 1; i >= 0; i--) {
        Tensor *prev_grad = layer_backward(model->layers[i], grad);
        tensor_free(grad);
        grad = prev_grad;
    }

    tensor_free(grad);
}

/* --------------------------------------------------------------------------
 * Categorical Cross-Entropy Loss
 *
 * L = -1/N * SUM_n SUM_c [ targets[n][c] * log(predictions[n][c] + eps) ]
 *
 * where N is batch size, eps is a small constant to avoid log(0).
 * -------------------------------------------------------------------------- */

float categorical_cross_entropy(const Tensor *predictions, const Tensor *targets) {
    int batch   = predictions->dims[0];
    int classes = predictions->dims[1];
    float eps   = 1e-7f;
    float loss  = 0.0f;

    for (int n = 0; n < batch; n++) {
        for (int c = 0; c < classes; c++) {
            float t = targets->data[n * classes + c];
            float p = predictions->data[n * classes + c];
            if (t > 0.0f) {
                loss -= t * logf(p + eps);
            }
        }
    }

    return loss / (float)batch;
}

/*
 * Gradient of CCE w.r.t. softmax output (combined softmax + CCE gradient):
 *
 *   dL/dy[n][c] = (predictions[n][c] - targets[n][c]) / batch_size
 *
 * This elegant simplification comes from combining the softmax Jacobian
 * with the cross-entropy derivative. The derivation:
 *   dL/dz_i = sum_j (dL/dy_j * dy_j/dz_i)
 *           = sum_j (-t_j/y_j * y_j*(delta_ij - y_i))
 *           = -t_i + y_i * sum_j(t_j)
 *           = y_i - t_i   (since sum(t_j) = 1 for one-hot)
 */
Tensor *categorical_cross_entropy_grad(const Tensor *predictions,
                                       const Tensor *targets) {
    Tensor *grad = tensor_clone(predictions);
    int batch = predictions->dims[0];

    for (size_t i = 0; i < grad->size; i++) {
        grad->data[i] = (grad->data[i] - targets->data[i]) / (float)batch;
    }

    return grad;
}

/* --------------------------------------------------------------------------
 * Optimizer creation
 * -------------------------------------------------------------------------- */

Optimizer *optimizer_create_sgd(float learning_rate) {
    Optimizer *opt = (Optimizer *)calloc(1, sizeof(Optimizer));
    if (!opt) return NULL;
    opt->type          = OPTIMIZER_SGD;
    opt->learning_rate = learning_rate;
    return opt;
}

Optimizer *optimizer_create_adam(float learning_rate, float beta1,
                                float beta2, float epsilon) {
    Optimizer *opt = (Optimizer *)calloc(1, sizeof(Optimizer));
    if (!opt) return NULL;
    opt->type          = OPTIMIZER_ADAM;
    opt->learning_rate = learning_rate;
    opt->beta1         = beta1;
    opt->beta2         = beta2;
    opt->epsilon        = epsilon;
    opt->timestep      = 0;
    opt->states        = NULL;
    opt->num_states    = 0;
    return opt;
}

void optimizer_free(Optimizer *opt) {
    if (!opt) return;
    for (int i = 0; i < opt->num_states; i++) {
        tensor_free(opt->states[i].m);
        tensor_free(opt->states[i].v);
    }
    free(opt->states);
    free(opt);
}

/* --------------------------------------------------------------------------
 * Adam state initialization
 *
 * We need one (m, v) pair per trainable parameter tensor.
 * Conv2D has 2 (filters, biases), Dense has 2 (weights, biases).
 * -------------------------------------------------------------------------- */

static int count_trainable_params(const Model *model) {
    int count = 0;
    for (int i = 0; i < model->num_layers; i++) {
        switch (model->layers[i]->type) {
            case LAYER_CONV2D: count += 2; break;
            case LAYER_DENSE:  count += 2; break;
            default: break;
        }
    }
    return count;
}

void optimizer_init_adam_states(Optimizer *opt, Model *model) {
    int n = count_trainable_params(model);
    opt->num_states = n;
    opt->states = (AdamState *)calloc((size_t)n, sizeof(AdamState));

    int idx = 0;
    for (int i = 0; i < model->num_layers; i++) {
        Layer *l = model->layers[i];
        if (l->type == LAYER_CONV2D) {
            Conv2DParams *p = &l->params.conv2d;
            opt->states[idx].m = tensor_create(p->filters->ndim, p->filters->dims);
            opt->states[idx].v = tensor_create(p->filters->ndim, p->filters->dims);
            idx++;
            opt->states[idx].m = tensor_create(p->biases->ndim, p->biases->dims);
            opt->states[idx].v = tensor_create(p->biases->ndim, p->biases->dims);
            idx++;
        } else if (l->type == LAYER_DENSE) {
            DenseParams *p = &l->params.dense;
            opt->states[idx].m = tensor_create(p->weights->ndim, p->weights->dims);
            opt->states[idx].v = tensor_create(p->weights->ndim, p->weights->dims);
            idx++;
            opt->states[idx].m = tensor_create(p->biases->ndim, p->biases->dims);
            opt->states[idx].v = tensor_create(p->biases->ndim, p->biases->dims);
            idx++;
        }
    }
}

/* --------------------------------------------------------------------------
 * SGD update: param -= lr * grad
 * -------------------------------------------------------------------------- */
static void sgd_update(Tensor *param, const Tensor *grad, float lr) {
    for (size_t i = 0; i < param->size; i++) {
        param->data[i] -= lr * grad->data[i];
    }
}

/* --------------------------------------------------------------------------
 * Adam update
 *
 * m = beta1 * m + (1 - beta1) * grad
 * v = beta2 * v + (1 - beta2) * grad^2
 * m_hat = m / (1 - beta1^t)
 * v_hat = v / (1 - beta2^t)
 * param -= lr * m_hat / (sqrt(v_hat) + epsilon)
 * -------------------------------------------------------------------------- */
static void adam_update(Tensor *param, const Tensor *grad,
                        AdamState *state, float lr,
                        float beta1, float beta2, float eps, int t) {
    float b1_corr = 1.0f - powf(beta1, (float)t);
    float b2_corr = 1.0f - powf(beta2, (float)t);

    for (size_t i = 0; i < param->size; i++) {
        float g = grad->data[i];
        state->m->data[i] = beta1 * state->m->data[i] + (1.0f - beta1) * g;
        state->v->data[i] = beta2 * state->v->data[i] + (1.0f - beta2) * g * g;

        float m_hat = state->m->data[i] / b1_corr;
        float v_hat = state->v->data[i] / b2_corr;

        param->data[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
    }
}

/* --------------------------------------------------------------------------
 * Optimizer step: update all trainable parameters
 * -------------------------------------------------------------------------- */

void optimizer_step(Optimizer *opt, Model *model) {
    if (opt->type == OPTIMIZER_ADAM) {
        opt->timestep++;
    }

    int state_idx = 0;
    for (int i = 0; i < model->num_layers; i++) {
        Layer *l = model->layers[i];

        if (l->type == LAYER_CONV2D) {
            Conv2DParams *p = &l->params.conv2d;
            if (opt->type == OPTIMIZER_SGD) {
                sgd_update(p->filters, p->d_filters, opt->learning_rate);
                sgd_update(p->biases, p->d_biases, opt->learning_rate);
            } else {
                adam_update(p->filters, p->d_filters, &opt->states[state_idx],
                            opt->learning_rate, opt->beta1, opt->beta2,
                            opt->epsilon, opt->timestep);
                state_idx++;
                adam_update(p->biases, p->d_biases, &opt->states[state_idx],
                            opt->learning_rate, opt->beta1, opt->beta2,
                            opt->epsilon, opt->timestep);
                state_idx++;
            }
        } else if (l->type == LAYER_DENSE) {
            DenseParams *p = &l->params.dense;
            if (opt->type == OPTIMIZER_SGD) {
                sgd_update(p->weights, p->d_weights, opt->learning_rate);
                sgd_update(p->biases, p->d_biases, opt->learning_rate);
            } else {
                adam_update(p->weights, p->d_weights, &opt->states[state_idx],
                            opt->learning_rate, opt->beta1, opt->beta2,
                            opt->epsilon, opt->timestep);
                state_idx++;
                adam_update(p->biases, p->d_biases, &opt->states[state_idx],
                            opt->learning_rate, opt->beta1, opt->beta2,
                            opt->epsilon, opt->timestep);
                state_idx++;
            }
        }
    }
}

/* --------------------------------------------------------------------------
 * Zero all gradients
 * -------------------------------------------------------------------------- */

void model_zero_grad(Model *model) {
    for (int i = 0; i < model->num_layers; i++) {
        Layer *l = model->layers[i];
        if (l->type == LAYER_CONV2D) {
            tensor_zeros(l->params.conv2d.d_filters);
            tensor_zeros(l->params.conv2d.d_biases);
        } else if (l->type == LAYER_DENSE) {
            tensor_zeros(l->params.dense.d_weights);
            tensor_zeros(l->params.dense.d_biases);
        }
    }
}

/* --------------------------------------------------------------------------
 * Training epoch
 *
 * Splits data into mini-batches, runs forward + loss + backward + update
 * for each batch. Returns average loss over the epoch.
 * -------------------------------------------------------------------------- */

float train_epoch(Model *model, Optimizer *opt,
                  const Tensor *images, const Tensor *labels,
                  int batch_size) {
    int num_samples = images->dims[0];
    int channels    = images->dims[1];
    int height      = images->dims[2];
    int width       = images->dims[3];
    int num_classes = labels->dims[1];

    int num_batches   = num_samples / batch_size;
    float total_loss  = 0.0f;
    int img_stride    = channels * height * width;
    int lbl_stride    = num_classes;

    for (int b = 0; b < num_batches; b++) {
        int start = b * batch_size;

        /* Extract mini-batch for images */
        int img_dims[] = {batch_size, channels, height, width};
        Tensor *batch_img = tensor_create(4, img_dims);
        memcpy(batch_img->data,
               images->data + start * img_stride,
               (size_t)(batch_size * img_stride) * sizeof(float));

        /* Extract mini-batch for labels */
        int lbl_dims[] = {batch_size, num_classes};
        Tensor *batch_lbl = tensor_create(2, lbl_dims);
        memcpy(batch_lbl->data,
               labels->data + start * lbl_stride,
               (size_t)(batch_size * lbl_stride) * sizeof(float));

        /* Zero gradients */
        model_zero_grad(model);

        /* Forward pass */
        Tensor *predictions = model_forward(model, batch_img);

        /* Compute loss */
        float loss = categorical_cross_entropy(predictions, batch_lbl);
        total_loss += loss;

        /* Compute loss gradient */
        Tensor *d_loss = categorical_cross_entropy_grad(predictions, batch_lbl);

        /* Backward pass */
        model_backward(model, d_loss);

        /* Update parameters */
        optimizer_step(opt, model);

        /* Cleanup batch tensors */
        tensor_free(batch_img);
        tensor_free(batch_lbl);
        tensor_free(predictions);
        tensor_free(d_loss);
    }

    return total_loss / (float)num_batches;
}

/* --------------------------------------------------------------------------
 * Model save / load
 * -------------------------------------------------------------------------- */

int model_save(const Model *model, const char *filepath) {
    FILE *fp = fopen(filepath, "wb");
    if (!fp) {
        fprintf(stderr, "ERROR: Cannot open '%s' for writing\n", filepath);
        return -1;
    }

    /* Write magic number and layer count */
    int magic = 0xC4400;
    fwrite(&magic, sizeof(int), 1, fp);
    fwrite(&model->num_layers, sizeof(int), 1, fp);

    for (int i = 0; i < model->num_layers; i++) {
        if (layer_save(model->layers[i], fp) != 0) {
            fclose(fp);
            return -1;
        }
    }

    fclose(fp);
    return 0;
}

Model *model_load(const char *filepath) {
    FILE *fp = fopen(filepath, "rb");
    if (!fp) {
        fprintf(stderr, "ERROR: Cannot open '%s' for reading\n", filepath);
        return NULL;
    }

    int magic, num_layers;
    fread(&magic, sizeof(int), 1, fp);
    if (magic != 0xC4400) {
        fprintf(stderr, "ERROR: Invalid model file magic number\n");
        fclose(fp);
        return NULL;
    }
    fread(&num_layers, sizeof(int), 1, fp);

    Model *model = model_create();

    for (int i = 0; i < num_layers; i++) {
        Layer *layer = layer_load(fp);
        if (!layer) {
            fprintf(stderr, "ERROR: Failed to load layer %d\n", i);
            model_free(model);
            fclose(fp);
            return NULL;
        }
        model_add_layer(model, layer);
    }

    fclose(fp);
    return model;
}
