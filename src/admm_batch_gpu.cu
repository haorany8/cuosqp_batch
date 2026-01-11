/**
 * Batched ADMM GPU Kernels - Implementation
 *
 * CUDA kernels for parallelizing ADMM update functions across multiple QP problems.
 */

#include "admm_batch_gpu.h"
#include "qdldl_batch_gpu.h"
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
 */
__global__ void kernel_batch_compute_rhs(
    int n,
    int m,
    c_float sigma,
    c_float rho_inv,
    int rho_is_vec,
    const c_float* d_x_prev,       // [batch_size * n]
    const c_float* d_q,            // [batch_size * n]
    const c_float* d_z_prev,       // [batch_size * m]
    const c_float* d_y,            // [batch_size * m]
    const c_float* d_rho_inv_vec,  // [m] or NULL
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
    if (rho_is_vec) {
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
 */
__global__ void kernel_batch_update_z(
    int n,
    int m,
    c_float alpha,
    c_float rho_inv,
    int rho_is_vec,
    const c_float* d_ztilde,       // [batch_size * m] (from RHS buffer, after n)
    const c_float* d_z_prev,       // [batch_size * m]
    const c_float* d_y,            // [batch_size * m]
    const c_float* d_rho_inv_vec,  // [m] or NULL
    const c_float* d_l,            // [batch_size * m]
    const c_float* d_u,            // [batch_size * m]
    c_float* d_z,                  // [batch_size * m] output
    int batch_size,
    int kkt_dim                    // stride for ztilde (n+m)
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const c_float* ztilde = d_ztilde + idx * kkt_dim + n;  // ztilde is after first n
    const c_float* z_prev = d_z_prev + idx * m;
    const c_float* y      = d_y + idx * m;
    const c_float* l      = d_l + idx * m;
    const c_float* u      = d_u + idx * m;
    c_float* z            = d_z + idx * m;

    c_float one_minus_alpha = 1.0f - alpha;

    for (int i = 0; i < m; i++) {
        c_float val;

        if (rho_is_vec) {
            val = d_rho_inv_vec[i] * y[i];
            val = val + alpha * ztilde[i] + one_minus_alpha * z_prev[i];
        } else {
            val = alpha * ztilde[i] + one_minus_alpha * z_prev[i] + rho_inv * y[i];
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
 */
__global__ void kernel_batch_update_y(
    int n,
    int m,
    c_float alpha,
    c_float rho,
    int rho_is_vec,
    const c_float* d_ztilde,    // [batch_size * m] (from RHS buffer, after n)
    const c_float* d_z_prev,    // [batch_size * m]
    const c_float* d_z,         // [batch_size * m]
    const c_float* d_rho_vec,   // [m] or NULL
    c_float* d_y,               // [batch_size * m] in/out
    c_float* d_delta_y,         // [batch_size * m] output
    int batch_size,
    int kkt_dim                 // stride for ztilde (n+m)
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const c_float* ztilde = d_ztilde + idx * kkt_dim + n;  // ztilde is after first n
    const c_float* z_prev = d_z_prev + idx * m;
    const c_float* z      = d_z + idx * m;
    c_float* y            = d_y + idx * m;
    c_float* delta_y      = d_delta_y + idx * m;

    c_float one_minus_alpha = 1.0f - alpha;

    for (int i = 0; i < m; i++) {
        // delta_y = alpha * ztilde + (1-alpha) * z_prev - z
        c_float dy = alpha * ztilde[i] + one_minus_alpha * z_prev[i] - z[i];

        // Scale by rho
        if (rho_is_vec) {
            dy = d_rho_vec[i] * dy;
        } else {
            dy = rho * dy;
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

    // Allocate problem data buffers
    cudaMalloc(&ws->d_q, batch_size * n * sizeof(c_float));
    cudaMalloc(&ws->d_l, batch_size * m * sizeof(c_float));
    cudaMalloc(&ws->d_u, batch_size * m * sizeof(c_float));

    // rho vectors are allocated on demand
    ws->d_rho_vec = NULL;
    ws->d_rho_inv_vec = NULL;
    ws->rho = 0.1f;
    ws->rho_inv = 10.0f;
    ws->sigma = 1e-6f;
    ws->alpha = 1.6f;
    ws->rho_is_vec = 0;

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
        if (ws->d_q)       cudaFree(ws->d_q);
        if (ws->d_l)       cudaFree(ws->d_l);
        if (ws->d_u)       cudaFree(ws->d_u);
        if (ws->d_rho_vec)     cudaFree(ws->d_rho_vec);
        if (ws->d_rho_inv_vec) cudaFree(ws->d_rho_inv_vec);
        free(ws);
    }
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
        ws->d_x_prev,
        ws->d_q,
        ws->d_z_prev,
        ws->d_y,
        ws->d_rho_inv_vec,
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
    int m = ws->m;
    int batch_size = ws->batch_size;
    int kkt_dim = n + m;

    int threads_per_block = 256;
    int num_blocks = (batch_size + threads_per_block - 1) / threads_per_block;

    c_float* d_xz_tilde = (c_float*)ws->base_ws->d_x;

    kernel_batch_update_x<<<num_blocks, threads_per_block>>>(
        n,
        ws->alpha,
        d_xz_tilde,
        ws->d_x_prev,
        ws->d_x,
        ws->d_delta_x,
        batch_size,
        kkt_dim
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel error in update_x: %s\n", cudaGetErrorString(err));
        return -1;
    }

    return 0;
}

int gpu_batch_update_z(GPUBatchADMMWorkspace* ws) {
    int n = ws->n;
    int m = ws->m;
    int batch_size = ws->batch_size;
    int kkt_dim = n + m;

    int threads_per_block = 256;
    int num_blocks = (batch_size + threads_per_block - 1) / threads_per_block;

    c_float* d_xz_tilde = (c_float*)ws->base_ws->d_x;

    kernel_batch_update_z<<<num_blocks, threads_per_block>>>(
        n, m,
        ws->alpha,
        ws->rho_inv,
        ws->rho_is_vec,
        d_xz_tilde,
        ws->d_z_prev,
        ws->d_y,
        ws->d_rho_inv_vec,
        ws->d_l,
        ws->d_u,
        ws->d_z,
        batch_size,
        kkt_dim
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel error in update_z: %s\n", cudaGetErrorString(err));
        return -1;
    }

    return 0;
}

int gpu_batch_update_y(GPUBatchADMMWorkspace* ws) {
    int n = ws->n;
    int m = ws->m;
    int batch_size = ws->batch_size;
    int kkt_dim = n + m;

    int threads_per_block = 256;
    int num_blocks = (batch_size + threads_per_block - 1) / threads_per_block;

    c_float* d_xz_tilde = (c_float*)ws->base_ws->d_x;

    kernel_batch_update_y<<<num_blocks, threads_per_block>>>(
        n, m,
        ws->alpha,
        ws->rho,
        ws->rho_is_vec,
        d_xz_tilde,
        ws->d_z_prev,
        ws->d_z,
        ws->d_rho_vec,
        ws->d_y,
        ws->d_delta_y,
        batch_size,
        kkt_dim
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

    // 4. Update x
    ret = gpu_batch_update_x(ws);
    if (ret != 0) return ret;

    // 5. Update z (with projection)
    ret = gpu_batch_update_z(ws);
    if (ret != 0) return ret;

    // 6. Update y
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

} // extern "C"
