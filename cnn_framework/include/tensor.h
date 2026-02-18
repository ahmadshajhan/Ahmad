/*
 * tensor.h - Tensor/Matrix Engine for Deep Learning Framework
 *
 * Provides a flexible N-dimensional tensor structure and core linear algebra
 * operations (matmul, transpose, element-wise ops) implemented from scratch.
 *
 * Memory layout: row-major, contiguous float array.
 * Dimensions are stored as (batch, channels/depth, height, width) for 4D,
 * or fewer dims as needed.
 */

#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>

/* Maximum number of dimensions supported */
#define TENSOR_MAX_DIMS 4

/* --------------------------------------------------------------------------
 * Core Tensor structure
 * -------------------------------------------------------------------------- */
typedef struct {
    float *data;                    /* Contiguous data buffer              */
    int    dims[TENSOR_MAX_DIMS];   /* Shape: [batch, channels, h, w]     */
    int    ndim;                    /* Number of active dimensions         */
    size_t size;                    /* Total number of elements            */
} Tensor;

/* --------------------------------------------------------------------------
 * Lifecycle
 * -------------------------------------------------------------------------- */

/* Create a tensor with the given number of dimensions and shape.
 * All elements are initialized to zero. Returns NULL on failure. */
Tensor *tensor_create(int ndim, const int *dims);

/* Create a deep copy of src. Returns NULL on failure. */
Tensor *tensor_clone(const Tensor *src);

/* Free all memory associated with t (including the struct itself). */
void tensor_free(Tensor *t);

/* --------------------------------------------------------------------------
 * Initialization helpers
 * -------------------------------------------------------------------------- */

/* Fill every element with val. */
void tensor_fill(Tensor *t, float val);

/* Fill with zeros. */
void tensor_zeros(Tensor *t);

/* Xavier / Glorot uniform initialization given fan_in and fan_out. */
void tensor_xavier_init(Tensor *t, int fan_in, int fan_out);

/* Simple random uniform in [-range, +range]. */
void tensor_rand_uniform(Tensor *t, float range);

/* --------------------------------------------------------------------------
 * Element access  (bounds-unchecked for speed)
 * -------------------------------------------------------------------------- */

/* 2D index (row, col) into a 2D tensor */
static inline float tensor_get2d(const Tensor *t, int r, int c) {
    return t->data[r * t->dims[1] + c];
}
static inline void tensor_set2d(Tensor *t, int r, int c, float v) {
    t->data[r * t->dims[1] + c] = v;
}

/* 4D index (batch, channel, row, col) */
static inline float tensor_get4d(const Tensor *t, int b, int ch, int r, int c) {
    return t->data[((b * t->dims[1] + ch) * t->dims[2] + r) * t->dims[3] + c];
}
static inline void tensor_set4d(Tensor *t, int b, int ch, int r, int c, float v) {
    t->data[((b * t->dims[1] + ch) * t->dims[2] + r) * t->dims[3] + c] = v;
}

/* --------------------------------------------------------------------------
 * Linear algebra operations
 * -------------------------------------------------------------------------- */

/* C = A * B  (2D matrix multiply). A is (M x K), B is (K x N), C is (M x N).
 * C must be pre-allocated. */
void tensor_matmul(const Tensor *A, const Tensor *B, Tensor *C);

/* out = transpose of src (2D only). out must be pre-allocated (N x M). */
void tensor_transpose(const Tensor *src, Tensor *out);

/* --------------------------------------------------------------------------
 * Element-wise operations  (in-place where noted)
 * -------------------------------------------------------------------------- */

/* C = A + B  (element-wise, same shape). */
void tensor_add(const Tensor *A, const Tensor *B, Tensor *C);

/* A += B  (in-place). */
void tensor_add_inplace(Tensor *A, const Tensor *B);

/* A *= scalar  (in-place). */
void tensor_scale(Tensor *A, float scalar);

/* C = A (element-wise) * B (Hadamard product). */
void tensor_elementwise_mul(const Tensor *A, const Tensor *B, Tensor *C);

/* --------------------------------------------------------------------------
 * Serialization
 * -------------------------------------------------------------------------- */

/* Write tensor to a binary file. Returns 0 on success. */
int tensor_save(const Tensor *t, FILE *fp);

/* Read tensor from a binary file. Returns newly allocated tensor or NULL. */
Tensor *tensor_load(FILE *fp);

/* --------------------------------------------------------------------------
 * Debug
 * -------------------------------------------------------------------------- */
void tensor_print_shape(const Tensor *t);

#endif /* TENSOR_H */
