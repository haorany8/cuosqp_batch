/**
 * Batched Sparse Matrix-Vector Products (SpMV) on GPU - Implementation
 */

#include "spmv_batch_gpu.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            return -1; \
        } \
    } while(0)

#define THREADS_PER_BLOCK 256

//=============================================================================
// CUDA Kernels
//=============================================================================

/**
 * Kernel: Batched CSC SpMV (y = A*x)
 * Each thread block handles one problem in the batch
 * Threads within a block cooperate on the SpMV
 */
__global__ void kernel_batch_csc_spmv(
    const c_int* __restrict__ Ap,
    const c_int* __restrict__ Ai,
    const c_float* __restrict__ Ax,
    const c_float* __restrict__ d_x,
    c_float* __restrict__ d_y,
    int nrows,
    int ncols,
    int batch_size
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    const c_float* x = d_x + batch_idx * ncols;
    c_float* y = d_y + batch_idx * nrows;

    // Initialize output to zero (collaborative within block)
    for (int i = threadIdx.x; i < nrows; i += blockDim.x) {
        y[i] = 0.0f;
    }
    __syncthreads();

    // Each thread processes some columns
    for (int col = threadIdx.x; col < ncols; col += blockDim.x) {
        c_float x_col = x[col];
        for (c_int k = Ap[col]; k < Ap[col + 1]; k++) {
            c_int row = Ai[k];
            atomicAdd(&y[row], Ax[k] * x_col);
        }
    }
}

/**
 * Kernel: Batched CSC SpMV transpose (y = A'*x)
 * A is m x n, so A' is n x m
 * Input x is [m], output y is [n]
 */
__global__ void kernel_batch_csc_spmv_transpose(
    const c_int* __restrict__ Ap,
    const c_int* __restrict__ Ai,
    const c_float* __restrict__ Ax,
    const c_float* __restrict__ d_x,
    c_float* __restrict__ d_y,
    int nrows,   // m
    int ncols,   // n
    int batch_size
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    const c_float* x = d_x + batch_idx * nrows;   // input is [m]
    c_float* y = d_y + batch_idx * ncols;         // output is [n]

    // A' * x: for each column j of A (row j of A'), sum A[i,j] * x[i]
    // In CSC: column j has entries at Ai[Ap[j]:Ap[j+1]] with values Ax[...]
    // A'[j,i] = A[i,j], so y[j] = sum_i A[i,j] * x[i]
    for (int col = threadIdx.x; col < ncols; col += blockDim.x) {
        c_float sum = 0.0f;
        for (c_int k = Ap[col]; k < Ap[col + 1]; k++) {
            c_int row = Ai[k];  // row index in A
            sum += Ax[k] * x[row];
        }
        y[col] = sum;
    }
}

/**
 * Kernel: Batched symmetric SpMV for upper triangular P (y = P*x)
 * P is stored as upper triangular in CSC format
 * For symmetric matrix: y = (P + P' - diag(P)) * x
 */
__global__ void kernel_batch_symmetric_spmv(
    const c_int* __restrict__ Pp,
    const c_int* __restrict__ Pi,
    const c_float* __restrict__ Px,
    const c_float* __restrict__ d_x,
    c_float* __restrict__ d_y,
    int n,
    int batch_size
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    const c_float* x = d_x + batch_idx * n;
    c_float* y = d_y + batch_idx * n;

    // Initialize output to zero
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        y[i] = 0.0f;
    }
    __syncthreads();

    // Process each column (upper triangular)
    for (int col = threadIdx.x; col < n; col += blockDim.x) {
        c_float x_col = x[col];
        c_float diag_contrib = 0.0f;

        for (c_int k = Pp[col]; k < Pp[col + 1]; k++) {
            c_int row = Pi[k];
            c_float val = Px[k];

            if (row == col) {
                // Diagonal: add once
                diag_contrib = val * x_col;
            } else if (row < col) {
                // Off-diagonal in upper triangle: add P[row,col]*x[col] to y[row]
                // and P[col,row]*x[row] to y[col] (symmetric)
                atomicAdd(&y[row], val * x_col);
                atomicAdd(&y[col], val * x[row]);
            }
        }

        // Add diagonal contribution
        atomicAdd(&y[col], diag_contrib);
    }
}

/**
 * Kernel: Batched axpy (y = alpha*x + beta*y)
 */
__global__ void kernel_batch_axpy(
    c_float alpha,
    const c_float* __restrict__ d_x,
    c_float beta,
    c_float* __restrict__ d_y,
    int n,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * n;

    if (idx < total) {
        d_y[idx] = alpha * d_x[idx] + beta * d_y[idx];
    }
}

/**
 * Kernel: Batched infinity norm
 * Each block computes norm for one problem
 */
__global__ void kernel_batch_inf_norm(
    const c_float* __restrict__ d_x,
    c_float* __restrict__ d_norm,
    int n,
    int batch_size
) {
    extern __shared__ c_float sdata[];

    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    const c_float* x = d_x + batch_idx * n;

    // Each thread finds max of its assigned elements
    c_float local_max = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        c_float abs_val = fabsf(x[i]);
        if (abs_val > local_max) local_max = abs_val;
    }

    sdata[threadIdx.x] = local_max;
    __syncthreads();

    // Parallel reduction to find maximum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (sdata[threadIdx.x + stride] > sdata[threadIdx.x]) {
                sdata[threadIdx.x] = sdata[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        d_norm[batch_idx] = sdata[0];
    }
}

/**
 * Kernel: Batched vector subtraction (z = x - y)
 */
__global__ void kernel_batch_vec_sub(
    const c_float* __restrict__ d_x,
    const c_float* __restrict__ d_y,
    c_float* __restrict__ d_z,
    int n,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * n;

    if (idx < total) {
        d_z[idx] = d_x[idx] - d_y[idx];
    }
}

/**
 * Kernel: Batched vector copy
 */
__global__ void kernel_batch_vec_copy(
    const c_float* __restrict__ d_x,
    c_float* __restrict__ d_y,
    int n,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * n;

    if (idx < total) {
        d_y[idx] = d_x[idx];
    }
}

//=============================================================================
// Host Functions
//=============================================================================

extern "C" {

GPUSparseMatrix* alloc_gpu_sparse_matrix(
    const c_int* h_p,
    const c_int* h_i,
    const c_float* h_x,
    c_int nrows,
    c_int ncols,
    c_int nnz
) {
    GPUSparseMatrix* mat = (GPUSparseMatrix*)malloc(sizeof(GPUSparseMatrix));
    if (!mat) return NULL;

    mat->nrows = nrows;
    mat->ncols = ncols;
    mat->nnz = nnz;

    cudaError_t err;

    err = cudaMalloc(&mat->d_p, (ncols + 1) * sizeof(c_int));
    if (err != cudaSuccess) { free(mat); return NULL; }

    err = cudaMalloc(&mat->d_i, nnz * sizeof(c_int));
    if (err != cudaSuccess) { cudaFree(mat->d_p); free(mat); return NULL; }

    err = cudaMalloc(&mat->d_x, nnz * sizeof(c_float));
    if (err != cudaSuccess) { cudaFree(mat->d_p); cudaFree(mat->d_i); free(mat); return NULL; }

    cudaMemcpy(mat->d_p, h_p, (ncols + 1) * sizeof(c_int), cudaMemcpyHostToDevice);
    cudaMemcpy(mat->d_i, h_i, nnz * sizeof(c_int), cudaMemcpyHostToDevice);
    cudaMemcpy(mat->d_x, h_x, nnz * sizeof(c_float), cudaMemcpyHostToDevice);

    return mat;
}

void free_gpu_sparse_matrix(GPUSparseMatrix* mat) {
    if (mat) {
        if (mat->d_p) cudaFree(mat->d_p);
        if (mat->d_i) cudaFree(mat->d_i);
        if (mat->d_x) cudaFree(mat->d_x);
        free(mat);
    }
}

int gpu_batch_Px(
    const GPUSparseMatrix* P,
    const c_float* d_x,
    c_float* d_Px,
    int n,
    int batch_size
) {
    if (!P || !d_x || !d_Px) return -1;

    // One block per batch element
    kernel_batch_symmetric_spmv<<<batch_size, THREADS_PER_BLOCK>>>(
        P->d_p,
        P->d_i,
        P->d_x,
        d_x,
        d_Px,
        n,
        batch_size
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in gpu_batch_Px: %s\n", cudaGetErrorString(err));
        return -1;
    }

    return 0;
}

int gpu_batch_Ax(
    const GPUSparseMatrix* A,
    const c_float* d_x,
    c_float* d_Ax,
    int n,
    int m,
    int batch_size
) {
    if (!A || !d_x || !d_Ax) return -1;

    // One block per batch element
    kernel_batch_csc_spmv<<<batch_size, THREADS_PER_BLOCK>>>(
        A->d_p,
        A->d_i,
        A->d_x,
        d_x,
        d_Ax,
        m,    // nrows
        n,    // ncols
        batch_size
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in gpu_batch_Ax: %s\n", cudaGetErrorString(err));
        return -1;
    }

    return 0;
}

int gpu_batch_Aty(
    const GPUSparseMatrix* A,
    const c_float* d_y,
    c_float* d_Aty,
    int n,
    int m,
    int batch_size
) {
    if (!A || !d_y || !d_Aty) return -1;

    // One block per batch element
    kernel_batch_csc_spmv_transpose<<<batch_size, THREADS_PER_BLOCK>>>(
        A->d_p,
        A->d_i,
        A->d_x,
        d_y,
        d_Aty,
        m,    // nrows of A (input dimension)
        n,    // ncols of A (output dimension)
        batch_size
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in gpu_batch_Aty: %s\n", cudaGetErrorString(err));
        return -1;
    }

    return 0;
}

int gpu_batch_axpy(
    c_float alpha,
    const c_float* d_x,
    c_float beta,
    c_float* d_y,
    int n,
    int batch_size
) {
    if (!d_x || !d_y) return -1;

    int total = batch_size * n;
    int num_blocks = (total + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    kernel_batch_axpy<<<num_blocks, THREADS_PER_BLOCK>>>(
        alpha, d_x, beta, d_y, n, batch_size
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in gpu_batch_axpy: %s\n", cudaGetErrorString(err));
        return -1;
    }

    return 0;
}

int gpu_batch_inf_norm(
    const c_float* d_x,
    c_float* d_norm,
    int n,
    int batch_size
) {
    if (!d_x || !d_norm) return -1;

    // One block per batch element, shared memory for reduction
    int shared_mem_size = THREADS_PER_BLOCK * sizeof(c_float);

    kernel_batch_inf_norm<<<batch_size, THREADS_PER_BLOCK, shared_mem_size>>>(
        d_x, d_norm, n, batch_size
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in gpu_batch_inf_norm: %s\n", cudaGetErrorString(err));
        return -1;
    }

    return 0;
}

int gpu_batch_vec_sub(
    const c_float* d_x,
    const c_float* d_y,
    c_float* d_z,
    int n,
    int batch_size
) {
    if (!d_x || !d_y || !d_z) return -1;

    int total = batch_size * n;
    int num_blocks = (total + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    kernel_batch_vec_sub<<<num_blocks, THREADS_PER_BLOCK>>>(
        d_x, d_y, d_z, n, batch_size
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in gpu_batch_vec_sub: %s\n", cudaGetErrorString(err));
        return -1;
    }

    return 0;
}

int gpu_batch_vec_copy(
    const c_float* d_x,
    c_float* d_y,
    int n,
    int batch_size
) {
    if (!d_x || !d_y) return -1;

    int total = batch_size * n;
    int num_blocks = (total + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    kernel_batch_vec_copy<<<num_blocks, THREADS_PER_BLOCK>>>(
        d_x, d_y, n, batch_size
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in gpu_batch_vec_copy: %s\n", cudaGetErrorString(err));
        return -1;
    }

    return 0;
}

} // extern "C"
