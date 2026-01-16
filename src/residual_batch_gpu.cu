/**
 * Batched Residual Computation on GPU - Implementation
 */

#include "residual_batch_gpu.h"
#include "spmv_batch_gpu.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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
 * Kernel: Compute tolerance from norms
 * eps = eps_abs * sqrt(dim) + eps_rel * norm
 */
__global__ void kernel_compute_tolerance(
    const c_float* __restrict__ d_norm,
    c_float eps_abs,
    c_float eps_rel,
    c_float sqrt_dim,
    c_float* __restrict__ d_eps,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    d_eps[idx] = eps_abs * sqrt_dim + eps_rel * d_norm[idx];
}

/**
 * Kernel: Element-wise max of two vectors, store result
 */
__global__ void kernel_element_max(
    const c_float* __restrict__ d_a,
    const c_float* __restrict__ d_b,
    c_float* __restrict__ d_c,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    c_float a_val = fabsf(d_a[idx]);
    c_float b_val = fabsf(d_b[idx]);
    d_c[idx] = (a_val > b_val) ? a_val : b_val;
}

/**
 * Kernel: Batched max of two scalars
 */
__global__ void kernel_batch_scalar_max(
    const c_float* __restrict__ d_a,
    const c_float* __restrict__ d_b,
    c_float* __restrict__ d_c,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    d_c[idx] = (d_a[idx] > d_b[idx]) ? d_a[idx] : d_b[idx];
}

/**
 * Kernel: Batched max of three scalars
 */
__global__ void kernel_batch_scalar_max3(
    const c_float* __restrict__ d_a,
    const c_float* __restrict__ d_b,
    const c_float* __restrict__ d_c,
    c_float* __restrict__ d_out,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    c_float max_val = d_a[idx];
    if (d_b[idx] > max_val) max_val = d_b[idx];
    if (d_c[idx] > max_val) max_val = d_c[idx];
    d_out[idx] = max_val;
}

/**
 * Kernel: Batched vector addition (z = x + y)
 */
__global__ void kernel_batch_vec_add(
    const c_float* __restrict__ d_x,
    const c_float* __restrict__ d_y,
    c_float* __restrict__ d_z,
    int n,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * n;

    if (idx < total) {
        d_z[idx] = d_x[idx] + d_y[idx];
    }
}

/**
 * Kernel: Batched three-term sum (w = x + y + z)
 */
__global__ void kernel_batch_vec_add3(
    const c_float* __restrict__ d_x,
    const c_float* __restrict__ d_y,
    const c_float* __restrict__ d_z,
    c_float* __restrict__ d_w,
    int n,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * n;

    if (idx < total) {
        d_w[idx] = d_x[idx] + d_y[idx] + d_z[idx];
    }
}

//=============================================================================
// Host Functions
//=============================================================================

extern "C" {

GPUBatchResidualWorkspace* alloc_gpu_residual_workspace(
    int n,
    int m,
    int batch_size
) {
    GPUBatchResidualWorkspace* ws = (GPUBatchResidualWorkspace*)calloc(1, sizeof(GPUBatchResidualWorkspace));
    if (!ws) return NULL;

    ws->n = n;
    ws->m = m;
    ws->batch_size = batch_size;

    cudaError_t err;

    // Allocate temporary buffers
    err = cudaMalloc(&ws->d_Ax, batch_size * m * sizeof(c_float));
    if (err != cudaSuccess) { free(ws); return NULL; }

    err = cudaMalloc(&ws->d_Px, batch_size * n * sizeof(c_float));
    if (err != cudaSuccess) { cudaFree(ws->d_Ax); free(ws); return NULL; }

    err = cudaMalloc(&ws->d_Aty, batch_size * n * sizeof(c_float));
    if (err != cudaSuccess) { cudaFree(ws->d_Ax); cudaFree(ws->d_Px); free(ws); return NULL; }

    int max_dim = (n > m) ? n : m;
    err = cudaMalloc(&ws->d_temp, batch_size * max_dim * sizeof(c_float));
    if (err != cudaSuccess) { cudaFree(ws->d_Ax); cudaFree(ws->d_Px); cudaFree(ws->d_Aty); free(ws); return NULL; }

    // Allocate residual outputs
    err = cudaMalloc(&ws->d_pri_res, batch_size * sizeof(c_float));
    if (err != cudaSuccess) { cudaFree(ws->d_Ax); cudaFree(ws->d_Px); cudaFree(ws->d_Aty); cudaFree(ws->d_temp); free(ws); return NULL; }

    err = cudaMalloc(&ws->d_dua_res, batch_size * sizeof(c_float));
    if (err != cudaSuccess) { cudaFree(ws->d_Ax); cudaFree(ws->d_Px); cudaFree(ws->d_Aty); cudaFree(ws->d_temp); cudaFree(ws->d_pri_res); free(ws); return NULL; }

    ws->P = NULL;
    ws->A = NULL;

    return ws;
}

void free_gpu_residual_workspace(GPUBatchResidualWorkspace* ws) {
    if (ws) {
        if (ws->d_Ax) cudaFree(ws->d_Ax);
        if (ws->d_Px) cudaFree(ws->d_Px);
        if (ws->d_Aty) cudaFree(ws->d_Aty);
        if (ws->d_temp) cudaFree(ws->d_temp);
        if (ws->d_pri_res) cudaFree(ws->d_pri_res);
        if (ws->d_dua_res) cudaFree(ws->d_dua_res);
        if (ws->P) free_gpu_sparse_matrix(ws->P);
        if (ws->A) free_gpu_sparse_matrix(ws->A);
        free(ws);
    }
}

int gpu_residual_set_matrices(
    GPUBatchResidualWorkspace* ws,
    const c_int* h_Pp, const c_int* h_Pi, const c_float* h_Px, c_int nnz_P,
    const c_int* h_Ap, const c_int* h_Ai, const c_float* h_Ax, c_int nnz_A
) {
    if (!ws) return -1;

    // Free old matrices if they exist
    if (ws->P) { free_gpu_sparse_matrix(ws->P); ws->P = NULL; }
    if (ws->A) { free_gpu_sparse_matrix(ws->A); ws->A = NULL; }

    // Allocate P matrix
    ws->P = alloc_gpu_sparse_matrix(h_Pp, h_Pi, h_Px, ws->n, ws->n, nnz_P);
    if (!ws->P) return -1;

    // Allocate A matrix
    ws->A = alloc_gpu_sparse_matrix(h_Ap, h_Ai, h_Ax, ws->m, ws->n, nnz_A);
    if (!ws->A) {
        free_gpu_sparse_matrix(ws->P);
        ws->P = NULL;
        return -1;
    }

    return 0;
}

int gpu_batch_primal_residual(
    GPUBatchResidualWorkspace* ws,
    const c_float* d_x,
    const c_float* d_z,
    c_float* d_pri_res
) {
    if (!ws || !ws->A || !d_x || !d_z || !d_pri_res) return -1;

    int n = ws->n;
    int m = ws->m;
    int batch_size = ws->batch_size;

    // Compute Ax
    int ret = gpu_batch_Ax(ws->A, d_x, ws->d_Ax, n, m, batch_size);
    if (ret != 0) return ret;

    // Compute Ax - z
    ret = gpu_batch_vec_sub(ws->d_Ax, d_z, ws->d_temp, m, batch_size);
    if (ret != 0) return ret;

    // Compute ||Ax - z||_inf
    ret = gpu_batch_inf_norm(ws->d_temp, d_pri_res, m, batch_size);
    return ret;
}

int gpu_batch_dual_residual(
    GPUBatchResidualWorkspace* ws,
    const c_float* d_x,
    const c_float* d_q,
    const c_float* d_y,
    c_float* d_dua_res
) {
    if (!ws || !ws->P || !ws->A || !d_x || !d_q || !d_y || !d_dua_res) return -1;

    int n = ws->n;
    int m = ws->m;
    int batch_size = ws->batch_size;
    int ret;

    // Compute Px
    ret = gpu_batch_Px(ws->P, d_x, ws->d_Px, n, batch_size);
    if (ret != 0) return ret;

    // Compute A'y
    ret = gpu_batch_Aty(ws->A, d_y, ws->d_Aty, n, m, batch_size);
    if (ret != 0) return ret;

    // Compute Px + q + A'y
    int total = batch_size * n;
    int num_blocks = (total + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    kernel_batch_vec_add3<<<num_blocks, THREADS_PER_BLOCK>>>(
        ws->d_Px, d_q, ws->d_Aty, ws->d_temp, n, batch_size
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in gpu_batch_dual_residual: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Compute ||Px + q + A'y||_inf
    ret = gpu_batch_inf_norm(ws->d_temp, d_dua_res, n, batch_size);
    return ret;
}

int gpu_batch_compute_residuals(
    GPUBatchResidualWorkspace* ws,
    const c_float* d_x,
    const c_float* d_z,
    const c_float* d_y,
    const c_float* d_q,
    c_float* d_pri_res,
    c_float* d_dua_res
) {
    int ret;

    // Compute primal residual
    ret = gpu_batch_primal_residual(ws, d_x, d_z, d_pri_res);
    if (ret != 0) return ret;

    // Compute dual residual
    ret = gpu_batch_dual_residual(ws, d_x, d_q, d_y, d_dua_res);
    return ret;
}

int gpu_batch_compute_tolerances(
    GPUBatchResidualWorkspace* ws,
    const c_float* d_x,
    const c_float* d_z,
    const c_float* d_y,
    const c_float* d_q,
    c_float eps_abs,
    c_float eps_rel,
    c_float* d_eps_pri,
    c_float* d_eps_dua
) {
    if (!ws || !ws->P || !ws->A) return -1;

    int n = ws->n;
    int m = ws->m;
    int batch_size = ws->batch_size;
    int ret;
    int num_blocks = (batch_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Temporary buffers for norms
    c_float* d_norm1;
    c_float* d_norm2;
    c_float* d_norm3;
    cudaMalloc(&d_norm1, batch_size * sizeof(c_float));
    cudaMalloc(&d_norm2, batch_size * sizeof(c_float));
    cudaMalloc(&d_norm3, batch_size * sizeof(c_float));

    // --- Primal tolerance ---
    // eps_pri = eps_abs * sqrt(m) + eps_rel * max(||Ax||, ||z||)

    // Compute ||Ax||
    ret = gpu_batch_Ax(ws->A, d_x, ws->d_Ax, n, m, batch_size);
    if (ret != 0) goto cleanup;
    ret = gpu_batch_inf_norm(ws->d_Ax, d_norm1, m, batch_size);
    if (ret != 0) goto cleanup;

    // Compute ||z||
    ret = gpu_batch_inf_norm(d_z, d_norm2, m, batch_size);
    if (ret != 0) goto cleanup;

    // max(||Ax||, ||z||)
    kernel_batch_scalar_max<<<num_blocks, THREADS_PER_BLOCK>>>(
        d_norm1, d_norm2, d_norm3, batch_size
    );

    // eps_pri = eps_abs * sqrt(m) + eps_rel * norm
    kernel_compute_tolerance<<<num_blocks, THREADS_PER_BLOCK>>>(
        d_norm3, eps_abs, eps_rel, sqrtf((float)m), d_eps_pri, batch_size
    );

    // --- Dual tolerance ---
    // eps_dua = eps_abs * sqrt(n) + eps_rel * max(||Px||, ||A'y||, ||q||)

    // Compute ||Px||
    ret = gpu_batch_Px(ws->P, d_x, ws->d_Px, n, batch_size);
    if (ret != 0) goto cleanup;
    ret = gpu_batch_inf_norm(ws->d_Px, d_norm1, n, batch_size);
    if (ret != 0) goto cleanup;

    // Compute ||A'y||
    ret = gpu_batch_Aty(ws->A, d_y, ws->d_Aty, n, m, batch_size);
    if (ret != 0) goto cleanup;
    ret = gpu_batch_inf_norm(ws->d_Aty, d_norm2, n, batch_size);
    if (ret != 0) goto cleanup;

    // Compute ||q||
    ret = gpu_batch_inf_norm(d_q, d_norm3, n, batch_size);
    if (ret != 0) goto cleanup;

    // max(||Px||, ||A'y||, ||q||)
    kernel_batch_scalar_max3<<<num_blocks, THREADS_PER_BLOCK>>>(
        d_norm1, d_norm2, d_norm3, d_norm1, batch_size
    );

    // eps_dua = eps_abs * sqrt(n) + eps_rel * norm
    kernel_compute_tolerance<<<num_blocks, THREADS_PER_BLOCK>>>(
        d_norm1, eps_abs, eps_rel, sqrtf((float)n), d_eps_dua, batch_size
    );

    ret = 0;

cleanup:
    cudaFree(d_norm1);
    cudaFree(d_norm2);
    cudaFree(d_norm3);
    return ret;
}

int gpu_residual_copy_to_host(
    GPUBatchResidualWorkspace* ws,
    c_float* h_pri_res,
    c_float* h_dua_res
) {
    if (!ws) return -1;

    if (h_pri_res) {
        CUDA_CHECK(cudaMemcpy(h_pri_res, ws->d_pri_res, ws->batch_size * sizeof(c_float), cudaMemcpyDeviceToHost));
    }
    if (h_dua_res) {
        CUDA_CHECK(cudaMemcpy(h_dua_res, ws->d_dua_res, ws->batch_size * sizeof(c_float), cudaMemcpyDeviceToHost));
    }

    return 0;
}

} // extern "C"
