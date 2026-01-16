/**
 * Batched Sparse Matrix-Vector Products (SpMV) on GPU
 *
 * Provides batched SpMV operations for residual computation:
 *   - Px: P is shared across all problems, x is different per problem
 *   - Ax: A is shared across all problems, x is different per problem
 *   - Aty: A' (transpose), shared A, different y per problem
 */

#ifndef SPMV_BATCH_GPU_H
#define SPMV_BATCH_GPU_H

#ifdef __cplusplus
extern "C" {
#endif

#include "types.h"

/**
 * Sparse matrix structure for GPU (CSC format, shared across batch)
 * Used for P and A matrices which have the same sparsity pattern
 */
typedef struct {
    c_int* d_p;       // Column pointers [n+1] or [m+1] on device
    c_int* d_i;       // Row indices [nnz] on device
    c_float* d_x;     // Values [nnz] on device (shared across batch)

    c_int nrows;      // Number of rows
    c_int ncols;      // Number of columns
    c_int nnz;        // Number of nonzeros
} GPUSparseMatrix;

/**
 * Allocate GPU sparse matrix from host CSC data
 *
 * @param h_p      Host column pointers [ncols+1]
 * @param h_i      Host row indices [nnz]
 * @param h_x      Host values [nnz]
 * @param nrows    Number of rows
 * @param ncols    Number of columns
 * @param nnz      Number of nonzeros
 * @return         GPU sparse matrix (caller must free with free_gpu_sparse_matrix)
 */
GPUSparseMatrix* alloc_gpu_sparse_matrix(
    const c_int* h_p,
    const c_int* h_i,
    const c_float* h_x,
    c_int nrows,
    c_int ncols,
    c_int nnz
);

/**
 * Free GPU sparse matrix
 */
void free_gpu_sparse_matrix(GPUSparseMatrix* mat);

/**
 * Batched Px computation: y[b] = P * x[b] for all b
 *
 * P is symmetric upper triangular, stored in CSC.
 * This computes the full symmetric product.
 *
 * @param P           Shared P matrix [n x n] on GPU
 * @param d_x         Device: input vectors [batch_size * n]
 * @param d_Px        Device: output vectors [batch_size * n]
 * @param n           Vector dimension
 * @param batch_size  Number of problems
 * @return            0 on success
 */
int gpu_batch_Px(
    const GPUSparseMatrix* P,
    const c_float* d_x,
    c_float* d_Px,
    int n,
    int batch_size
);

/**
 * Batched Ax computation: y[b] = A * x[b] for all b
 *
 * A is stored in CSC format.
 *
 * @param A           Shared A matrix [m x n] on GPU
 * @param d_x         Device: input vectors [batch_size * n]
 * @param d_Ax        Device: output vectors [batch_size * m]
 * @param n           Input dimension
 * @param m           Output dimension
 * @param batch_size  Number of problems
 * @return            0 on success
 */
int gpu_batch_Ax(
    const GPUSparseMatrix* A,
    const c_float* d_x,
    c_float* d_Ax,
    int n,
    int m,
    int batch_size
);

/**
 * Batched A'y computation: x[b] = A' * y[b] for all b
 *
 * A is stored in CSC format. This computes the transpose product.
 *
 * @param A           Shared A matrix [m x n] on GPU
 * @param d_y         Device: input vectors [batch_size * m]
 * @param d_Aty       Device: output vectors [batch_size * n]
 * @param n           Output dimension
 * @param m           Input dimension
 * @param batch_size  Number of problems
 * @return            0 on success
 */
int gpu_batch_Aty(
    const GPUSparseMatrix* A,
    const c_float* d_y,
    c_float* d_Aty,
    int n,
    int m,
    int batch_size
);

/**
 * Batched vector addition: y[b] = alpha * x[b] + beta * y[b]
 *
 * @param alpha       Scalar multiplier for x
 * @param d_x         Device: input vectors [batch_size * n]
 * @param beta        Scalar multiplier for y
 * @param d_y         Device: in/out vectors [batch_size * n]
 * @param n           Vector dimension
 * @param batch_size  Number of problems
 * @return            0 on success
 */
int gpu_batch_axpy(
    c_float alpha,
    const c_float* d_x,
    c_float beta,
    c_float* d_y,
    int n,
    int batch_size
);

/**
 * Batched infinity norm: norm[b] = ||x[b]||_inf
 *
 * @param d_x         Device: input vectors [batch_size * n]
 * @param d_norm      Device: output norms [batch_size]
 * @param n           Vector dimension
 * @param batch_size  Number of problems
 * @return            0 on success
 */
int gpu_batch_inf_norm(
    const c_float* d_x,
    c_float* d_norm,
    int n,
    int batch_size
);

/**
 * Batched vector subtraction: z[b] = x[b] - y[b]
 *
 * @param d_x         Device: input vectors [batch_size * n]
 * @param d_y         Device: input vectors [batch_size * n]
 * @param d_z         Device: output vectors [batch_size * n]
 * @param n           Vector dimension
 * @param batch_size  Number of problems
 * @return            0 on success
 */
int gpu_batch_vec_sub(
    const c_float* d_x,
    const c_float* d_y,
    c_float* d_z,
    int n,
    int batch_size
);

/**
 * Batched vector copy: y[b] = x[b]
 *
 * @param d_x         Device: input vectors [batch_size * n]
 * @param d_y         Device: output vectors [batch_size * n]
 * @param n           Vector dimension
 * @param batch_size  Number of problems
 * @return            0 on success
 */
int gpu_batch_vec_copy(
    const c_float* d_x,
    c_float* d_y,
    int n,
    int batch_size
);

#ifdef __cplusplus
}
#endif

#endif // SPMV_BATCH_GPU_H
