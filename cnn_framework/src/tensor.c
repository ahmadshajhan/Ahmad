/*
 * tensor.c - Tensor/Matrix Engine Implementation
 *
 * All tensor operations are implemented using raw pointer arithmetic
 * on a contiguous float buffer. No external BLAS or linear algebra
 * libraries are used.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "../include/tensor.h"

/* --------------------------------------------------------------------------
 * Internal: seed the RNG once
 * -------------------------------------------------------------------------- */
static int g_rng_seeded = 0;

static void ensure_rng_seeded(void) {
    if (!g_rng_seeded) {
        srand((unsigned int)time(NULL));
        g_rng_seeded = 1;
    }
}

/* Uniform random in [0, 1) */
static float rand_uniform(void) {
    return (float)rand() / ((float)RAND_MAX + 1.0f);
}

/* --------------------------------------------------------------------------
 * Lifecycle
 * -------------------------------------------------------------------------- */

Tensor *tensor_create(int ndim, const int *dims) {
    if (ndim <= 0 || ndim > TENSOR_MAX_DIMS || !dims) return NULL;

    Tensor *t = (Tensor *)malloc(sizeof(Tensor));
    if (!t) return NULL;

    t->ndim = ndim;
    t->size = 1;
    for (int i = 0; i < TENSOR_MAX_DIMS; i++) {
        if (i < ndim) {
            t->dims[i] = dims[i];
            t->size *= (size_t)dims[i];
        } else {
            t->dims[i] = 1;
        }
    }

    t->data = (float *)calloc(t->size, sizeof(float));
    if (!t->data) {
        free(t);
        return NULL;
    }
    return t;
}

Tensor *tensor_clone(const Tensor *src) {
    if (!src) return NULL;
    Tensor *dst = tensor_create(src->ndim, src->dims);
    if (!dst) return NULL;
    memcpy(dst->data, src->data, src->size * sizeof(float));
    return dst;
}

void tensor_free(Tensor *t) {
    if (!t) return;
    free(t->data);
    free(t);
}

/* --------------------------------------------------------------------------
 * Initialization helpers
 * -------------------------------------------------------------------------- */

void tensor_fill(Tensor *t, float val) {
    if (!t) return;
    for (size_t i = 0; i < t->size; i++) {
        t->data[i] = val;
    }
}

void tensor_zeros(Tensor *t) {
    if (!t) return;
    memset(t->data, 0, t->size * sizeof(float));
}

void tensor_xavier_init(Tensor *t, int fan_in, int fan_out) {
    if (!t) return;
    ensure_rng_seeded();
    float limit = sqrtf(6.0f / (float)(fan_in + fan_out));
    for (size_t i = 0; i < t->size; i++) {
        t->data[i] = (rand_uniform() * 2.0f - 1.0f) * limit;
    }
}

void tensor_rand_uniform(Tensor *t, float range) {
    if (!t) return;
    ensure_rng_seeded();
    for (size_t i = 0; i < t->size; i++) {
        t->data[i] = (rand_uniform() * 2.0f - 1.0f) * range;
    }
}

/* --------------------------------------------------------------------------
 * Linear algebra: Matrix Multiplication
 *
 * C[M x N] = A[M x K] * B[K x N]
 *
 * Standard triple-loop with row-major access patterns optimized by
 * iterating (i, k, j) to improve cache locality on A's rows.
 * -------------------------------------------------------------------------- */

void tensor_matmul(const Tensor *A, const Tensor *B, Tensor *C) {
    int M = A->dims[0];
    int K = A->dims[1];
    int N = B->dims[1];

    /* Zero out C first */
    memset(C->data, 0, C->size * sizeof(float));

    /*
     * ikj loop order: for each row i of A, iterate over k (columns of A /
     * rows of B), then scatter A[i][k] * B[k][j] into C[i][j].
     * This gives sequential access on B's rows and good cache reuse on A.
     */
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            float a_ik = A->data[i * K + k];
            for (int j = 0; j < N; j++) {
                C->data[i * N + j] += a_ik * B->data[k * N + j];
            }
        }
    }
}

/* --------------------------------------------------------------------------
 * Transpose (2D only)
 * -------------------------------------------------------------------------- */

void tensor_transpose(const Tensor *src, Tensor *out) {
    int rows = src->dims[0];
    int cols = src->dims[1];

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            out->data[j * rows + i] = src->data[i * cols + j];
        }
    }
}

/* --------------------------------------------------------------------------
 * Element-wise operations
 * -------------------------------------------------------------------------- */

void tensor_add(const Tensor *A, const Tensor *B, Tensor *C) {
    for (size_t i = 0; i < A->size; i++) {
        C->data[i] = A->data[i] + B->data[i];
    }
}

void tensor_add_inplace(Tensor *A, const Tensor *B) {
    for (size_t i = 0; i < A->size; i++) {
        A->data[i] += B->data[i];
    }
}

void tensor_scale(Tensor *A, float scalar) {
    for (size_t i = 0; i < A->size; i++) {
        A->data[i] *= scalar;
    }
}

void tensor_elementwise_mul(const Tensor *A, const Tensor *B, Tensor *C) {
    for (size_t i = 0; i < A->size; i++) {
        C->data[i] = A->data[i] * B->data[i];
    }
}

/* --------------------------------------------------------------------------
 * Serialization
 * -------------------------------------------------------------------------- */

int tensor_save(const Tensor *t, FILE *fp) {
    if (!t || !fp) return -1;
    fwrite(&t->ndim, sizeof(int), 1, fp);
    fwrite(t->dims, sizeof(int), TENSOR_MAX_DIMS, fp);
    fwrite(t->data, sizeof(float), t->size, fp);
    return 0;
}

Tensor *tensor_load(FILE *fp) {
    if (!fp) return NULL;

    int ndim;
    int dims[TENSOR_MAX_DIMS];

    if (fread(&ndim, sizeof(int), 1, fp) != 1) return NULL;
    if (fread(dims, sizeof(int), TENSOR_MAX_DIMS, fp) != (size_t)TENSOR_MAX_DIMS) return NULL;

    Tensor *t = tensor_create(ndim, dims);
    if (!t) return NULL;

    if (fread(t->data, sizeof(float), t->size, fp) != t->size) {
        tensor_free(t);
        return NULL;
    }
    return t;
}

/* --------------------------------------------------------------------------
 * Debug
 * -------------------------------------------------------------------------- */

void tensor_print_shape(const Tensor *t) {
    if (!t) {
        printf("Tensor(NULL)\n");
        return;
    }
    printf("Tensor(ndim=%d, shape=[", t->ndim);
    for (int i = 0; i < t->ndim; i++) {
        printf("%d%s", t->dims[i], (i < t->ndim - 1) ? ", " : "");
    }
    printf("], size=%zu)\n", t->size);
}
