/*
 * image_io.h - Basic Image Processing Utilities
 *
 * Functions to load raw pixel data into Tensor structs and generate
 * synthetic / dummy image datasets for testing.
 */

#ifndef IMAGE_IO_H
#define IMAGE_IO_H

#include "tensor.h"

/* Load a raw binary image file into a tensor.
 * The file is expected to contain width*height*channels float values
 * in row-major order (CHW layout).
 * Returns a 3D tensor [channels, height, width] or NULL on failure. */
Tensor *image_load_raw(const char *filepath, int channels, int height, int width);

/* Save a tensor as a raw binary image file.
 * Tensor must be 3D [channels, height, width].
 * Returns 0 on success. */
int image_save_raw(const Tensor *img, const char *filepath);

/* Generate a dummy dataset of random images and one-hot labels.
 * out_images: [num_samples, channels, height, width]
 * out_labels: [num_samples, num_classes]
 * Returns 0 on success. */
int generate_dummy_dataset(int num_samples, int channels, int height, int width,
                           int num_classes, Tensor **out_images, Tensor **out_labels);

/* Normalize pixel values from [0, 255] range to [0, 1]. In-place. */
void image_normalize(Tensor *img);

#endif /* IMAGE_IO_H */
