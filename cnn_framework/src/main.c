/*
 * main.c - CNN Framework Demo
 *
 * Demonstrates:
 *   1. Building a CNN architecture:
 *      Conv2D -> ReLU -> MaxPool -> Flatten -> Dense -> Softmax
 *   2. Training on synthetic (dummy) image data
 *   3. Saving the trained model to a binary file
 *   4. Loading the model back and running inference
 *
 * Configuration:
 *   - Input: 8x8 grayscale images (1 channel), simulating a tiny dataset
 *   - Classes: 3 (arbitrary for demo)
 *   - Optimizer: Adam
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "../include/tensor.h"
#include "../include/layers.h"
#include "../include/training.h"
#include "../include/image_io.h"

/* --------------------------------------------------------------------------
 * Configuration constants
 * -------------------------------------------------------------------------- */
#define IMG_CHANNELS  1
#define IMG_HEIGHT    8
#define IMG_WIDTH     8
#define NUM_CLASSES   3
#define NUM_SAMPLES   64
#define BATCH_SIZE    8
#define NUM_EPOCHS    20
#define LEARNING_RATE 0.001f
#define MODEL_PATH    "model.bin"

/* --------------------------------------------------------------------------
 * Build the CNN model
 *
 * Architecture:
 *   [Input: 1x8x8]
 *   -> Conv2D(in=1, filters=4, kernel=3, stride=1, pad=0) => 4x6x6
 *   -> ReLU
 *   -> MaxPool(size=2, stride=2) => 4x3x3
 *   -> Flatten => 36
 *   -> Dense(36, NUM_CLASSES)
 *   -> Softmax
 * -------------------------------------------------------------------------- */
static Model *build_model(void) {
    Model *model = model_create();

    /* Conv2D: 1 input channel, 4 filters, 3x3 kernel, stride 1, no padding */
    model_add_layer(model, layer_create_conv2d(IMG_CHANNELS, 4, 3, 1, 0));

    /* ReLU activation */
    model_add_layer(model, layer_create_relu());

    /* MaxPool: 2x2 window, stride 2 */
    model_add_layer(model, layer_create_maxpool(2, 2));

    /* Flatten: 4 channels * 3 * 3 = 36 features */
    model_add_layer(model, layer_create_flatten());

    /* Dense: 36 -> NUM_CLASSES */
    model_add_layer(model, layer_create_dense(4 * 3 * 3, NUM_CLASSES));

    /* Softmax for classification output */
    model_add_layer(model, layer_create_softmax());

    return model;
}

/* --------------------------------------------------------------------------
 * Run inference on a single sample and print predictions
 * -------------------------------------------------------------------------- */
static void run_inference(Model *model, const Tensor *images, int sample_idx) {
    int img_size = IMG_CHANNELS * IMG_HEIGHT * IMG_WIDTH;

    /* Extract one sample as a batch of 1 */
    int dims[] = {1, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH};
    Tensor *single = tensor_create(4, dims);
    for (int i = 0; i < img_size; i++) {
        single->data[i] = images->data[sample_idx * img_size + i];
    }

    Tensor *pred = model_forward(model, single);

    printf("  Inference on sample %d: [", sample_idx);
    for (int c = 0; c < NUM_CLASSES; c++) {
        printf("%.4f%s", pred->data[c], (c < NUM_CLASSES - 1) ? ", " : "");
    }
    printf("]\n");

    /* Find predicted class */
    int best = 0;
    float best_val = pred->data[0];
    for (int c = 1; c < NUM_CLASSES; c++) {
        if (pred->data[c] > best_val) {
            best_val = pred->data[c];
            best = c;
        }
    }
    printf("  Predicted class: %d (confidence: %.4f)\n", best, best_val);

    tensor_free(single);
    tensor_free(pred);
}

/* --------------------------------------------------------------------------
 * Main
 * -------------------------------------------------------------------------- */
int main(void) {
    printf("=== CNN Framework - Pure C Deep Learning ===\n\n");

    srand((unsigned int)time(NULL));

    /* ---- Step 1: Generate dummy dataset ---- */
    printf("[1] Generating dummy dataset (%d samples, %dx%dx%d, %d classes)...\n",
           NUM_SAMPLES, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES);

    Tensor *images = NULL;
    Tensor *labels = NULL;
    if (generate_dummy_dataset(NUM_SAMPLES, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH,
                               NUM_CLASSES, &images, &labels) != 0) {
        fprintf(stderr, "Failed to generate dataset\n");
        return 1;
    }
    tensor_print_shape(images);
    tensor_print_shape(labels);

    /* ---- Step 2: Build model ---- */
    printf("\n[2] Building CNN model...\n");
    printf("    Conv2D(1->4, 3x3) -> ReLU -> MaxPool(2x2) -> Flatten -> Dense(36->%d) -> Softmax\n",
           NUM_CLASSES);
    Model *model = build_model();
    printf("    Model created with %d layers\n", model->num_layers);

    /* ---- Step 3: Create optimizer ---- */
    printf("\n[3] Creating Adam optimizer (lr=%.4f)...\n", LEARNING_RATE);
    Optimizer *opt = optimizer_create_adam(LEARNING_RATE, 0.9f, 0.999f, 1e-8f);
    optimizer_init_adam_states(opt, model);

    /* ---- Step 4: Training loop ---- */
    printf("\n[4] Training for %d epochs (batch_size=%d)...\n\n", NUM_EPOCHS, BATCH_SIZE);
    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        float avg_loss = train_epoch(model, opt, images, labels, BATCH_SIZE);
        printf("    Epoch %2d/%d  |  Loss: %.6f\n", epoch + 1, NUM_EPOCHS, avg_loss);
    }

    /* ---- Step 5: Run inference before saving ---- */
    printf("\n[5] Running inference (before save)...\n");
    run_inference(model, images, 0);
    run_inference(model, images, 1);

    /* ---- Step 6: Save model ---- */
    printf("\n[6] Saving model to '%s'...\n", MODEL_PATH);
    if (model_save(model, MODEL_PATH) == 0) {
        printf("    Model saved successfully\n");
    } else {
        fprintf(stderr, "    Failed to save model\n");
    }

    /* ---- Step 7: Load model and run inference again ---- */
    printf("\n[7] Loading model from '%s'...\n", MODEL_PATH);
    Model *loaded = model_load(MODEL_PATH);
    if (loaded) {
        printf("    Model loaded successfully (%d layers)\n", loaded->num_layers);
        printf("    Running inference (after load)...\n");
        run_inference(loaded, images, 0);
        run_inference(loaded, images, 1);
        model_free(loaded);
    } else {
        fprintf(stderr, "    Failed to load model\n");
    }

    /* ---- Cleanup ---- */
    printf("\n[8] Cleaning up...\n");
    model_free(model);
    optimizer_free(opt);
    tensor_free(images);
    tensor_free(labels);

    /* Remove the model file */
    remove(MODEL_PATH);

    printf("\nDone.\n");
    return 0;
}
