/*
 * image_io.c - Basic Image Processing Utilities
 *
 * Provides raw image loading, saving, dummy dataset generation,
 * and pixel normalization.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../include/tensor.h"
#include "../include/image_io.h"

/* --------------------------------------------------------------------------
 * Load raw binary image
 *
 * Expects a file containing exactly channels * height * width float values
 * in CHW (channel-first) order.
 * -------------------------------------------------------------------------- */
Tensor *image_load_raw(const char *filepath, int channels, int height, int width) {
    FILE *fp = fopen(filepath, "rb");
    if (!fp) {
        fprintf(stderr, "ERROR: Cannot open image file '%s'\n", filepath);
        return NULL;
    }

    int dims[] = {channels, height, width};
    Tensor *img = tensor_create(3, dims);
    if (!img) { fclose(fp); return NULL; }

    size_t expected = (size_t)(channels * height * width);
    size_t read = fread(img->data, sizeof(float), expected, fp);
    fclose(fp);

    if (read != expected) {
        fprintf(stderr, "ERROR: Expected %zu floats, got %zu from '%s'\n",
                expected, read, filepath);
        tensor_free(img);
        return NULL;
    }

    return img;
}

/* --------------------------------------------------------------------------
 * Save tensor as raw binary image
 * -------------------------------------------------------------------------- */
int image_save_raw(const Tensor *img, const char *filepath) {
    if (!img || !filepath) return -1;

    FILE *fp = fopen(filepath, "wb");
    if (!fp) {
        fprintf(stderr, "ERROR: Cannot open '%s' for writing\n", filepath);
        return -1;
    }

    fwrite(img->data, sizeof(float), img->size, fp);
    fclose(fp);
    return 0;
}

/* --------------------------------------------------------------------------
 * Generate dummy dataset
 *
 * Creates random images in [0, 1] and random one-hot labels.
 * Useful for testing the training pipeline without real data.
 * -------------------------------------------------------------------------- */
int generate_dummy_dataset(int num_samples, int channels, int height, int width,
                           int num_classes, Tensor **out_images, Tensor **out_labels) {
    if (!out_images || !out_labels) return -1;

    /* Images: [num_samples, channels, height, width] */
    int img_dims[] = {num_samples, channels, height, width};
    Tensor *images = tensor_create(4, img_dims);
    if (!images) return -1;

    /* Fill with random values in [0, 1] */
    for (size_t i = 0; i < images->size; i++) {
        images->data[i] = (float)rand() / (float)RAND_MAX;
    }

    /* Labels: [num_samples, num_classes] (one-hot) */
    int lbl_dims[] = {num_samples, num_classes};
    Tensor *labels = tensor_create(2, lbl_dims);
    if (!labels) {
        tensor_free(images);
        return -1;
    }

    /* Assign random one-hot labels */
    for (int n = 0; n < num_samples; n++) {
        int cls = rand() % num_classes;
        labels->data[n * num_classes + cls] = 1.0f;
    }

    *out_images = images;
    *out_labels = labels;
    return 0;
}

/* --------------------------------------------------------------------------
 * Normalize pixel values from [0, 255] to [0, 1]
 * -------------------------------------------------------------------------- */
void image_normalize(Tensor *img) {
    if (!img) return;
    for (size_t i = 0; i < img->size; i++) {
        img->data[i] /= 255.0f;
    }
}
