/**
 * Batched ADMM GPU Kernels - Implementation
 *
 * CUDA kernels for parallelizing ADMM update functions across multiple QP problems.
 */

#include "admm_batch_gpu.h"
#include "qdldl_batch_gpu.h"
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

//=============================================================================
// Batched ADMM Kernels
//=============================================================================

/**
 * Kernel: Compute RHS for batched KKT solve
 * Each thread handles one problem in the batch
 *
 * RHS structure: [xtilde_part (n), ztilde_part (m)]
 *   xtilde_part = sigma * x_prev - q
 *   ztilde_part = z_prev - rho_inv * y  (scalar rho)
 *              or z_prev - rho_inv_vec .* y (vector rho)
 *              or z_prev - rho_inv_batch[idx] * y (per-problem rho)
 */
__global__ void kernel_batch_compute_rhs(
    int n,
    int m,
    c_float sigma,
    c_float rho_inv,
    int rho_is_vec,
    int use_per_problem_rho,
    const c_float* d_x_prev,       // [batch_size * n]
    const c_float* d_q,            // [batch_size * n]
    const c_float* d_z_prev,       // [batch_size * m]
    const c_float* d_y,            // [batch_size * m]
    const c_float* d_rho_inv_vec,  // [m] or NULL
    const c_float* d_rho_inv_batch, // [batch_size] or NULL (per-problem rho)
    c_float* d_rhs,                // [batch_size * (n+m)] output
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    int kkt_dim = n + m;

    // Pointers for this problem
    const c_float* x_prev = d_x_prev + idx * n;
    const c_float* q      = d_q + idx * n;
    const c_float* z_prev = d_z_prev + idx * m;
    const c_float* y      = d_y + idx * m;
    c_float* rhs          = d_rhs + idx * kkt_dim;

    // xtilde part: sigma * x_prev - q
    for (int i = 0; i < n; i++) {
        rhs[i] = sigma * x_prev[i] - q[i];
    }

    // ztilde part: z_prev - rho_inv * y
    // Priority: per-problem rho > vector rho > scalar rho
    c_float rho_inv_to_use;
    if (use_per_problem_rho && d_rho_inv_batch) {
        rho_inv_to_use = d_rho_inv_batch[idx];
        for (int i = 0; i < m; i++) {
            rhs[n + i] = z_prev[i] - rho_inv_to_use * y[i];
        }
    } else if (rho_is_vec && d_rho_inv_vec) {
        for (int i = 0; i < m; i++) {
            rhs[n + i] = z_prev[i] - d_rho_inv_vec[i] * y[i];
        }
    } else {
        for (int i = 0; i < m; i++) {
            rhs[n + i] = z_prev[i] - rho_inv * y[i];
        }
    }
}

/**
 * Kernel: Extract xtilde from KKT solution buffer (strided) to contiguous buffer
 * KKT solution has layout [batch_size * (n+m)], xtilde is first n elements per problem
 */
__global__ void kernel_batch_extract_xtilde(
    int n,
    int kkt_dim,
    const c_float* d_kkt_solution,   // [batch_size * kkt_dim] strided input
    c_float* d_xtilde,               // [batch_size * n] contiguous output
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const c_float* kkt_sol = d_kkt_solution + idx * kkt_dim;  // stride = n+m
    c_float* xtilde = d_xtilde + idx * n;                     // stride = n

    for (int i = 0; i < n; i++) {
        xtilde[i] = kkt_sol[i];
    }
}

/**
 * Kernel: Batched update_x
 *   x = alpha * xtilde + (1 - alpha) * x_prev
 *   delta_x = x - x_prev
 */
__global__ void kernel_batch_update_x(
    int n,
    c_float alpha,
    const c_float* d_xtilde,   // [batch_size * n] (from RHS buffer, first n)
    const c_float* d_x_prev,   // [batch_size * n]
    c_float* d_x,              // [batch_size * n] output
    c_float* d_delta_x,        // [batch_size * n] output
    int batch_size,
    int kkt_dim                // stride for xtilde (n+m)
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const c_float* xtilde = d_xtilde + idx * kkt_dim;  // xtilde is first n of RHS
    const c_float* x_prev = d_x_prev + idx * n;
    c_float* x            = d_x + idx * n;
    c_float* delta_x      = d_delta_x + idx * n;

    c_float one_minus_alpha = 1.0f - alpha;

    for (int i = 0; i < n; i++) {
        c_float xp = x_prev[i];
        x[i] = alpha * xtilde[i] + one_minus_alpha * xp;
        delta_x[i] = x[i] - xp;
    }
}

/**
 * Kernel: Batched update_z with projection
 *   z = alpha * ztilde + (1 - alpha) * z_prev + rho_inv * y
 *   z = clamp(z, l, u)
 *
 * NOTE: ztilde is now A * xtilde, stored in a separate buffer with stride m
 */
__global__ void kernel_batch_update_z(
    int m,
    c_float alpha,
    c_float rho_inv,
    int rho_is_vec,
    int use_per_problem_rho,
    const c_float* d_ztilde,       // [batch_size * m] = A * xtilde
    const c_float* d_z_prev,       // [batch_size * m]
    const c_float* d_y,            // [batch_size * m]
    const c_float* d_rho_inv_vec,  // [m] or NULL
    const c_float* d_rho_inv_batch, // [batch_size] or NULL (per-problem rho)
    const c_float* d_l,            // [batch_size * m]
    const c_float* d_u,            // [batch_size * m]
    c_float* d_z,                  // [batch_size * m] output
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const c_float* ztilde = d_ztilde + idx * m;  // ztilde = A * xtilde
    const c_float* z_prev = d_z_prev + idx * m;
    const c_float* y      = d_y + idx * m;
    const c_float* l      = d_l + idx * m;
    const c_float* u      = d_u + idx * m;
    c_float* z            = d_z + idx * m;

    c_float one_minus_alpha = 1.0f - alpha;

    // Get rho_inv for this problem
    c_float rho_inv_to_use = rho_inv;
    if (use_per_problem_rho && d_rho_inv_batch) {
        rho_inv_to_use = d_rho_inv_batch[idx];
    }

    for (int i = 0; i < m; i++) {
        c_float val;

        if (rho_is_vec && !use_per_problem_rho) {
            val = d_rho_inv_vec[i] * y[i];
            val = val + alpha * ztilde[i] + one_minus_alpha * z_prev[i];
        } else {
            val = alpha * ztilde[i] + one_minus_alpha * z_prev[i] + rho_inv_to_use * y[i];
        }

        // Project onto [l, u]
        if (val < l[i]) {
            z[i] = l[i];
        } else if (val > u[i]) {
            z[i] = u[i];
        } else {
            z[i] = val;
        }
    }
}

/**
 * Kernel: Batched update_y
 *   delta_y = alpha * ztilde + (1 - alpha) * z_prev - z
 *   delta_y = rho * delta_y
 *   y = y + delta_y
 *
 * NOTE: ztilde is now A * xtilde, stored in a separate buffer with stride m
 */
__global__ void kernel_batch_update_y(
    int m,
    c_float alpha,
    c_float rho,
    int rho_is_vec,
    int use_per_problem_rho,
    const c_float* d_ztilde,    // [batch_size * m] = A * xtilde
    const c_float* d_z_prev,    // [batch_size * m]
    const c_float* d_z,         // [batch_size * m]
    const c_float* d_rho_vec,   // [m] or NULL
    const c_float* d_rho_batch, // [batch_size] or NULL (per-problem rho)
    c_float* d_y,               // [batch_size * m] in/out
    c_float* d_delta_y,         // [batch_size * m] output
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const c_float* ztilde = d_ztilde + idx * m;  // ztilde = A * xtilde
    const c_float* z_prev = d_z_prev + idx * m;
    const c_float* z      = d_z + idx * m;
    c_float* y            = d_y + idx * m;
    c_float* delta_y      = d_delta_y + idx * m;

    c_float one_minus_alpha = 1.0f - alpha;

    // Get rho for this problem
    c_float rho_to_use = rho;
    if (use_per_problem_rho && d_rho_batch) {
        rho_to_use = d_rho_batch[idx];
    }

    for (int i = 0; i < m; i++) {
        // delta_y = alpha * ztilde + (1-alpha) * z_prev - z
        c_float dy = alpha * ztilde[i] + one_minus_alpha * z_prev[i] - z[i];

        // Scale by rho
        if (rho_is_vec && !use_per_problem_rho) {
            dy = d_rho_vec[i] * dy;
        } else {
            dy = rho_to_use * dy;
        }

        delta_y[i] = dy;
        y[i] = y[i] + dy;
    }
}

//=============================================================================
// Batched ADMM Host Functions
//=============================================================================

extern "C" {

GPUBatchADMMWorkspace* alloc_gpu_admm_workspace(
    GPUBatchWorkspace* base_ws,
    int n,
    int m,
    int batch_size
) {
    GPUBatchADMMWorkspace* ws = (GPUBatchADMMWorkspace*)malloc(sizeof(GPUBatchADMMWorkspace));
    if (!ws) return NULL;

    ws->base_ws = base_ws;
    ws->n = n;
    ws->m = m;
    ws->batch_size = batch_size;

    // Allocate ADMM iterate buffers
    cudaMalloc(&ws->d_x,       batch_size * n * sizeof(c_float));
    cudaMalloc(&ws->d_x_prev,  batch_size * n * sizeof(c_float));
    cudaMalloc(&ws->d_z,       batch_size * m * sizeof(c_float));
    cudaMalloc(&ws->d_z_prev,  batch_size * m * sizeof(c_float));
    cudaMalloc(&ws->d_y,       batch_size * m * sizeof(c_float));
    cudaMalloc(&ws->d_delta_x, batch_size * n * sizeof(c_float));
    cudaMalloc(&ws->d_delta_y, batch_size * m * sizeof(c_float));
    cudaMalloc(&ws->d_xtilde,  batch_size * n * sizeof(c_float));
    cudaMalloc(&ws->d_ztilde,  batch_size * m * sizeof(c_float));

    // Allocate problem data buffers
    cudaMalloc(&ws->d_q, batch_size * n * sizeof(c_float));
    cudaMalloc(&ws->d_l, batch_size * m * sizeof(c_float));
    cudaMalloc(&ws->d_u, batch_size * m * sizeof(c_float));

    // A matrix - allocated later via gpu_admm_set_A_matrix
    ws->A = NULL;

    // rho vectors are allocated on demand
    ws->d_rho_vec = NULL;
    ws->d_rho_inv_vec = NULL;
    ws->rho = 0.1f;
    ws->rho_inv = 10.0f;
    ws->sigma = 1e-6f;
    ws->alpha = 1.6f;
    ws->rho_is_vec = 0;

    // Per-problem rho support (allocated on demand)
    ws->d_rho_batch = NULL;
    ws->d_rho_inv_batch = NULL;
    ws->d_rho_inv_diag_indices = NULL;
    ws->use_per_problem_rho = 0;

    return ws;
}

void free_gpu_admm_workspace(GPUBatchADMMWorkspace* ws) {
    if (ws) {
        if (ws->d_x)       cudaFree(ws->d_x);
        if (ws->d_x_prev)  cudaFree(ws->d_x_prev);
        if (ws->d_z)       cudaFree(ws->d_z);
        if (ws->d_z_prev)  cudaFree(ws->d_z_prev);
        if (ws->d_y)       cudaFree(ws->d_y);
        if (ws->d_delta_x) cudaFree(ws->d_delta_x);
        if (ws->d_delta_y) cudaFree(ws->d_delta_y);
        if (ws->d_xtilde)  cudaFree(ws->d_xtilde);
        if (ws->d_ztilde)  cudaFree(ws->d_ztilde);
        if (ws->d_q)       cudaFree(ws->d_q);
        if (ws->d_l)       cudaFree(ws->d_l);
        if (ws->d_u)       cudaFree(ws->d_u);
        if (ws->d_rho_vec)     cudaFree(ws->d_rho_vec);
        if (ws->d_rho_inv_vec) cudaFree(ws->d_rho_inv_vec);
        // Per-problem rho arrays
        if (ws->d_rho_batch)          cudaFree(ws->d_rho_batch);
        if (ws->d_rho_inv_batch)      cudaFree(ws->d_rho_inv_batch);
        if (ws->d_rho_inv_diag_indices) cudaFree(ws->d_rho_inv_diag_indices);
        if (ws->A) free_gpu_sparse_matrix(ws->A);
        free(ws);
    }
}

int gpu_admm_set_A_matrix(
    GPUBatchADMMWorkspace* ws,
    const c_int* h_Ap,
    const c_int* h_Ai,
    const c_float* h_Ax,
    c_int nnz
) {
    if (!ws || !h_Ap || !h_Ai || !h_Ax) return -1;

    // Free existing A if any
    if (ws->A) {
        free_gpu_sparse_matrix(ws->A);
        ws->A = NULL;
    }

    // Allocate new A matrix on GPU
    ws->A = alloc_gpu_sparse_matrix(h_Ap, h_Ai, h_Ax, ws->m, ws->n, nnz);
    if (!ws->A) return -1;

    return 0;
}

int gpu_admm_copy_problem_data(
    GPUBatchADMMWorkspace* ws,
    const c_float* h_q,
    const c_float* h_l,
    const c_float* h_u,
    const c_float* h_rho_vec,
    c_float rho,
    c_float sigma,
    c_float alpha
) {
    int n = ws->n;
    int m = ws->m;
    int batch_size = ws->batch_size;

    // Copy q, l, u
    CUDA_CHECK(cudaMemcpy(ws->d_q, h_q, batch_size * n * sizeof(c_float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ws->d_l, h_l, batch_size * m * sizeof(c_float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ws->d_u, h_u, batch_size * m * sizeof(c_float), cudaMemcpyHostToDevice));

    // Store scalar parameters
    ws->sigma = sigma;
    ws->alpha = alpha;
    ws->rho = rho;
    ws->rho_inv = 1.0f / rho;

    // Handle rho vector
    if (h_rho_vec) {
        ws->rho_is_vec = 1;

        // Allocate if needed
        if (!ws->d_rho_vec) {
            CUDA_CHECK(cudaMalloc(&ws->d_rho_vec, m * sizeof(c_float)));
        }
        if (!ws->d_rho_inv_vec) {
            CUDA_CHECK(cudaMalloc(&ws->d_rho_inv_vec, m * sizeof(c_float)));
        }

        // Copy rho_vec
        CUDA_CHECK(cudaMemcpy(ws->d_rho_vec, h_rho_vec, m * sizeof(c_float), cudaMemcpyHostToDevice));

        // Compute rho_inv_vec on host and copy
        c_float* h_rho_inv_vec = (c_float*)malloc(m * sizeof(c_float));
        for (int i = 0; i < m; i++) {
            h_rho_inv_vec[i] = 1.0f / h_rho_vec[i];
        }
        CUDA_CHECK(cudaMemcpy(ws->d_rho_inv_vec, h_rho_inv_vec, m * sizeof(c_float), cudaMemcpyHostToDevice));
        free(h_rho_inv_vec);
    } else {
        ws->rho_is_vec = 0;
    }

    return 0;
}

int gpu_admm_copy_initial_iterates(
    GPUBatchADMMWorkspace* ws,
    const c_float* h_x,
    const c_float* h_z,
    const c_float* h_y
) {
    int n = ws->n;
    int m = ws->m;
    int batch_size = ws->batch_size;

    CUDA_CHECK(cudaMemcpy(ws->d_x, h_x, batch_size * n * sizeof(c_float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ws->d_z, h_z, batch_size * m * sizeof(c_float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ws->d_y, h_y, batch_size * m * sizeof(c_float), cudaMemcpyHostToDevice));

    // Also copy to _prev buffers
    CUDA_CHECK(cudaMemcpy(ws->d_x_prev, h_x, batch_size * n * sizeof(c_float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ws->d_z_prev, h_z, batch_size * m * sizeof(c_float), cudaMemcpyHostToDevice));

    return 0;
}

int gpu_admm_copy_solution(
    GPUBatchADMMWorkspace* ws,
    c_float* h_x,
    c_float* h_z,
    c_float* h_y
) {
    int n = ws->n;
    int m = ws->m;
    int batch_size = ws->batch_size;

    CUDA_CHECK(cudaMemcpy(h_x, ws->d_x, batch_size * n * sizeof(c_float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_z, ws->d_z, batch_size * m * sizeof(c_float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_y, ws->d_y, batch_size * m * sizeof(c_float), cudaMemcpyDeviceToHost));

    return 0;
}

void gpu_batch_swap_iterates(GPUBatchADMMWorkspace* ws) {
    // Pointer swap for x <-> x_prev
    c_float* tmp = ws->d_x;
    ws->d_x = ws->d_x_prev;
    ws->d_x_prev = tmp;

    // Pointer swap for z <-> z_prev
    tmp = ws->d_z;
    ws->d_z = ws->d_z_prev;
    ws->d_z_prev = tmp;
}

int gpu_batch_compute_rhs(
    const GPUFactorPattern* pattern,
    GPUBatchADMMWorkspace* ws
) {
    (void)pattern;  // unused, but kept for consistency

    int n = ws->n;
    int m = ws->m;
    int batch_size = ws->batch_size;

    int threads_per_block = 256;
    int num_blocks = (batch_size + threads_per_block - 1) / threads_per_block;

    // Get the RHS buffer from base workspace
    c_float* d_rhs = (c_float*)ws->base_ws->d_x;

    kernel_batch_compute_rhs<<<num_blocks, threads_per_block>>>(
        n, m,
        ws->sigma,
        ws->rho_inv,
        ws->rho_is_vec,
        ws->use_per_problem_rho,
        ws->d_x_prev,
        ws->d_q,
        ws->d_z_prev,
        ws->d_y,
        ws->d_rho_inv_vec,
        ws->d_rho_inv_batch,
        d_rhs,
        batch_size
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel error in compute_rhs: %s\n", cudaGetErrorString(err));
        return -1;
    }

    return 0;
}

int gpu_batch_update_x(GPUBatchADMMWorkspace* ws) {
    int n = ws->n;
    int batch_size = ws->batch_size;

    int threads_per_block = 256;
    int num_blocks = (batch_size + threads_per_block - 1) / threads_per_block;

    // Use the contiguous d_xtilde buffer (stride = n)
    kernel_batch_update_x<<<num_blocks, threads_per_block>>>(
        n,
        ws->alpha,
        ws->d_xtilde,       // Now using contiguous xtilde buffer
        ws->d_x_prev,
        ws->d_x,
        ws->d_delta_x,
        batch_size,
        n                   // stride = n (contiguous)
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel error in update_x: %s\n", cudaGetErrorString(err));
        return -1;
    }

    return 0;
}

int gpu_batch_update_z(GPUBatchADMMWorkspace* ws) {
    int m = ws->m;
    int batch_size = ws->batch_size;

    int threads_per_block = 256;
    int num_blocks = (batch_size + threads_per_block - 1) / threads_per_block;

    kernel_batch_update_z<<<num_blocks, threads_per_block>>>(
        m,
        ws->alpha,
        ws->rho_inv,
        ws->rho_is_vec,
        ws->use_per_problem_rho,
        ws->d_ztilde,    // Now using separate ztilde buffer
        ws->d_z_prev,
        ws->d_y,
        ws->d_rho_inv_vec,
        ws->d_rho_inv_batch,
        ws->d_l,
        ws->d_u,
        ws->d_z,
        batch_size
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel error in update_z: %s\n", cudaGetErrorString(err));
        return -1;
    }

    return 0;
}

int gpu_batch_update_y(GPUBatchADMMWorkspace* ws) {
    int m = ws->m;
    int batch_size = ws->batch_size;

    int threads_per_block = 256;
    int num_blocks = (batch_size + threads_per_block - 1) / threads_per_block;

    kernel_batch_update_y<<<num_blocks, threads_per_block>>>(
        m,
        ws->alpha,
        ws->rho,
        ws->rho_is_vec,
        ws->use_per_problem_rho,
        ws->d_ztilde,    // Now using separate ztilde buffer
        ws->d_z_prev,
        ws->d_z,
        ws->d_rho_vec,
        ws->d_rho_batch,
        ws->d_y,
        ws->d_delta_y,
        batch_size
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel error in update_y: %s\n", cudaGetErrorString(err));
        return -1;
    }

    return 0;
}

int gpu_batch_admm_iteration(
    const GPUFactorPattern* pattern,
    GPUBatchADMMWorkspace* ws,
    int admm_iter
) {
    (void)admm_iter;  // Not used currently, but kept for future (e.g., logging)

    int n = ws->n;
    int m = ws->m;
    int batch_size = ws->batch_size;
    int ret;

    // 1. Swap iterates: x <-> x_prev, z <-> z_prev
    gpu_batch_swap_iterates(ws);

    // 2. Compute RHS for KKT system
    ret = gpu_batch_compute_rhs(pattern, ws);
    if (ret != 0) return ret;

    // 3. Solve KKT system (factorization already done, just solve)
    // Note: Assumes factorization is done once since P is the same for all
    c_float* d_rhs = (c_float*)ws->base_ws->d_x;
    ret = gpu_batch_solve(pattern, ws->base_ws, d_rhs, batch_size);
    if (ret != 0) return ret;

    // 4. Extract xtilde from KKT solution buffer (strided) to contiguous buffer
    //    KKT solution has layout [batch_size * (n+m)], xtilde is first n elements per problem
    {
        int kkt_dim = n + m;
        int threads_per_block = 256;
        int num_blocks = (batch_size + threads_per_block - 1) / threads_per_block;
        kernel_batch_extract_xtilde<<<num_blocks, threads_per_block>>>(
            n, kkt_dim, d_rhs, ws->d_xtilde, batch_size
        );
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA kernel error in extract_xtilde: %s\n", cudaGetErrorString(err));
            return -1;
        }
    }

    // 5. Update x from xtilde (now using contiguous xtilde buffer)
    ret = gpu_batch_update_x(ws);
    if (ret != 0) return ret;

    // 6. Compute ztilde = A * xtilde (CRITICAL: use actual xtilde, not relaxed x!)
    if (ws->A) {
        // Now using the contiguous d_xtilde buffer
        ret = gpu_batch_Ax(ws->A, ws->d_xtilde, ws->d_ztilde, n, m, batch_size);
        if (ret != 0) return ret;
    } else {
        fprintf(stderr, "Warning: A matrix not set in ADMM workspace\n");
        return -1;
    }

    // 6. Update z (with projection)
    ret = gpu_batch_update_z(ws);
    if (ret != 0) return ret;

    // 7. Update y
    ret = gpu_batch_update_y(ws);
    if (ret != 0) return ret;

    // Synchronize to ensure all operations complete
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA sync error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    return 0;
}

//=============================================================================
// Per-Problem Rho Support Functions
//=============================================================================

int gpu_admm_init_per_problem_rho(
    GPUBatchADMMWorkspace* ws,
    const c_int* h_KKTp,
    const c_int* h_KKTi,
    c_int n,
    c_int m,
    c_float initial_rho
) {
    if (!ws || !h_KKTp || !h_KKTi) return -1;

    int batch_size = ws->batch_size;

    // Allocate per-problem rho arrays
    CUDA_CHECK(cudaMalloc(&ws->d_rho_batch, batch_size * sizeof(c_float)));
    CUDA_CHECK(cudaMalloc(&ws->d_rho_inv_batch, batch_size * sizeof(c_float)));
    CUDA_CHECK(cudaMalloc(&ws->d_rho_inv_diag_indices, m * sizeof(c_int)));

    // Initialize all problems with the same rho
    c_float* h_rho_batch = (c_float*)malloc(batch_size * sizeof(c_float));
    c_float* h_rho_inv_batch = (c_float*)malloc(batch_size * sizeof(c_float));
    c_float rho_inv = 1.0f / initial_rho;
    for (int i = 0; i < batch_size; i++) {
        h_rho_batch[i] = initial_rho;
        h_rho_inv_batch[i] = rho_inv;
    }
    CUDA_CHECK(cudaMemcpy(ws->d_rho_batch, h_rho_batch, batch_size * sizeof(c_float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ws->d_rho_inv_batch, h_rho_inv_batch, batch_size * sizeof(c_float), cudaMemcpyHostToDevice));
    free(h_rho_batch);
    free(h_rho_inv_batch);

    // Find the diagonal indices for -1/rho in the KKT matrix
    // KKT structure: [P+sigma*I, A'; A, -1/rho*I]
    // The -1/rho entries are on the diagonal for columns n to n+m-1
    c_int* h_rho_inv_diag_indices = (c_int*)malloc(m * sizeof(c_int));
    for (c_int j = 0; j < m; j++) {
        c_int col = n + j;  // Column in KKT matrix
        c_int row = n + j;  // Diagonal entry has row == col
        // Search for diagonal entry in this column
        c_int found = -1;
        for (c_int k = h_KKTp[col]; k < h_KKTp[col + 1]; k++) {
            if (h_KKTi[k] == row) {
                found = k;
                break;
            }
        }
        if (found < 0) {
            fprintf(stderr, "Error: -1/rho diagonal entry not found at column %d\n", col);
            free(h_rho_inv_diag_indices);
            return -1;
        }
        h_rho_inv_diag_indices[j] = found;
    }
    CUDA_CHECK(cudaMemcpy(ws->d_rho_inv_diag_indices, h_rho_inv_diag_indices, m * sizeof(c_int), cudaMemcpyHostToDevice));
    free(h_rho_inv_diag_indices);

    ws->use_per_problem_rho = 1;
    ws->rho = initial_rho;
    ws->rho_inv = rho_inv;

    return 0;
}

int gpu_admm_update_problem_rho(
    GPUBatchADMMWorkspace* ws,
    const GPUFactorPattern* gpu_pattern,
    c_int problem_idx,
    c_float new_rho
) {
    if (!ws || !ws->use_per_problem_rho || problem_idx < 0 || problem_idx >= ws->batch_size) {
        return -1;
    }

    c_float new_rho_inv = 1.0f / new_rho;

    // Update rho and rho_inv for this problem in the arrays
    CUDA_CHECK(cudaMemcpy(ws->d_rho_batch + problem_idx, &new_rho, sizeof(c_float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ws->d_rho_inv_batch + problem_idx, &new_rho_inv, sizeof(c_float), cudaMemcpyHostToDevice));

    // Update the KKT matrix diagonal for this problem
    // Using gpu_batch_update_rho with batch_size=1 and offset
    // Actually we need a dedicated kernel for single-problem update
    // For now, we'll just update the rho arrays and the full KKT update
    // will be done by gpu_admm_update_all_rho or a separate call

    return 0;
}

int gpu_admm_update_all_rho(
    GPUBatchADMMWorkspace* ws,
    const GPUFactorPattern* gpu_pattern,
    const c_float* h_new_rho
) {
    if (!ws || !ws->use_per_problem_rho || !h_new_rho) {
        return -1;
    }

    int batch_size = ws->batch_size;
    int m = ws->m;

    // Compute rho_inv for each problem
    c_float* h_rho_inv = (c_float*)malloc(batch_size * sizeof(c_float));
    for (int i = 0; i < batch_size; i++) {
        h_rho_inv[i] = 1.0f / h_new_rho[i];
    }

    // Copy to device
    CUDA_CHECK(cudaMemcpy(ws->d_rho_batch, h_new_rho, batch_size * sizeof(c_float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ws->d_rho_inv_batch, h_rho_inv, batch_size * sizeof(c_float), cudaMemcpyHostToDevice));
    free(h_rho_inv);

    // Update KKT matrix diagonals for all problems using the batched kernel
    int ret = gpu_batch_update_rho(
        gpu_pattern,
        ws->base_ws,
        ws->d_rho_batch,
        ws->n,
        m,
        ws->d_rho_inv_diag_indices,
        batch_size
    );

    return ret;
}

int gpu_admm_get_rho(
    GPUBatchADMMWorkspace* ws,
    c_float* h_rho
) {
    if (!ws || !h_rho) return -1;

    if (ws->use_per_problem_rho && ws->d_rho_batch) {
        CUDA_CHECK(cudaMemcpy(h_rho, ws->d_rho_batch, ws->batch_size * sizeof(c_float), cudaMemcpyDeviceToHost));
    } else {
        // All problems have the same rho
        for (int i = 0; i < ws->batch_size; i++) {
            h_rho[i] = ws->rho;
        }
    }

    return 0;
}

} // extern "C"
