/**
 * Batched Termination Checking on GPU - Implementation
 */

#include "termination_batch_gpu.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
 * Kernel: Initialize status to OSQP_UNSOLVED
 */
__global__ void kernel_init_status(
    c_int* __restrict__ d_status,
    c_int* __restrict__ d_converged,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    d_status[idx] = OSQP_UNSOLVED;
    d_converged[idx] = 0;
}

/**
 * Kernel: Check convergence with per-problem tolerances
 * converged[b] = (pri_res[b] <= eps_pri[b]) && (dua_res[b] <= eps_dua[b])
 */
__global__ void kernel_check_convergence(
    const c_float* __restrict__ d_pri_res,
    const c_float* __restrict__ d_dua_res,
    const c_float* __restrict__ d_eps_pri,
    const c_float* __restrict__ d_eps_dua,
    c_int* __restrict__ d_converged,
    c_int* __restrict__ d_status,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    // Only check if not already converged
    if (d_status[idx] == OSQP_SOLVED) {
        d_converged[idx] = 1;
        return;
    }

    int pri_ok = (d_pri_res[idx] <= d_eps_pri[idx]);
    int dua_ok = (d_dua_res[idx] <= d_eps_dua[idx]);

    if (pri_ok && dua_ok) {
        d_converged[idx] = 1;
        d_status[idx] = OSQP_SOLVED;
    } else {
        d_converged[idx] = 0;
    }
}

/**
 * Kernel: Check convergence with scalar tolerances
 */
__global__ void kernel_check_convergence_scalar(
    const c_float* __restrict__ d_pri_res,
    const c_float* __restrict__ d_dua_res,
    c_float eps_pri,
    c_float eps_dua,
    c_int* __restrict__ d_converged,
    c_int* __restrict__ d_status,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    // Only check if not already converged
    if (d_status[idx] == OSQP_SOLVED) {
        d_converged[idx] = 1;
        return;
    }

    int pri_ok = (d_pri_res[idx] <= eps_pri);
    int dua_ok = (d_dua_res[idx] <= eps_dua);

    if (pri_ok && dua_ok) {
        d_converged[idx] = 1;
        d_status[idx] = OSQP_SOLVED;
    } else {
        d_converged[idx] = 0;
    }
}

/**
 * Kernel: Mark max iterations reached for unconverged problems
 */
__global__ void kernel_mark_max_iter(
    c_int* __restrict__ d_status,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    if (d_status[idx] == OSQP_UNSOLVED) {
        d_status[idx] = OSQP_MAX_ITER_REACHED;
    }
}

/**
 * Kernel: Count converged problems using parallel reduction
 */
__global__ void kernel_count_converged(
    const c_int* __restrict__ d_converged,
    c_int* __restrict__ d_count,
    int batch_size
) {
    extern __shared__ c_int sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load and accumulate
    sdata[tid] = (idx < batch_size) ? d_converged[idx] : 0;
    __syncthreads();

    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(d_count, sdata[0]);
    }
}

//=============================================================================
// Host Functions
//=============================================================================

extern "C" {

GPUBatchTermination* alloc_gpu_termination_workspace(int batch_size) {
    GPUBatchTermination* ws = (GPUBatchTermination*)calloc(1, sizeof(GPUBatchTermination));
    if (!ws) return NULL;

    ws->batch_size = batch_size;

    cudaError_t err;

    // Allocate device arrays
    err = cudaMalloc(&ws->d_status, batch_size * sizeof(c_int));
    if (err != cudaSuccess) { free(ws); return NULL; }

    err = cudaMalloc(&ws->d_converged, batch_size * sizeof(c_int));
    if (err != cudaSuccess) { cudaFree(ws->d_status); free(ws); return NULL; }

    err = cudaMalloc(&ws->d_pri_res, batch_size * sizeof(c_float));
    if (err != cudaSuccess) { cudaFree(ws->d_status); cudaFree(ws->d_converged); free(ws); return NULL; }

    err = cudaMalloc(&ws->d_dua_res, batch_size * sizeof(c_float));
    if (err != cudaSuccess) { cudaFree(ws->d_status); cudaFree(ws->d_converged); cudaFree(ws->d_pri_res); free(ws); return NULL; }

    err = cudaMalloc(&ws->d_eps_pri, batch_size * sizeof(c_float));
    if (err != cudaSuccess) { cudaFree(ws->d_status); cudaFree(ws->d_converged); cudaFree(ws->d_pri_res); cudaFree(ws->d_dua_res); free(ws); return NULL; }

    err = cudaMalloc(&ws->d_eps_dua, batch_size * sizeof(c_float));
    if (err != cudaSuccess) { cudaFree(ws->d_status); cudaFree(ws->d_converged); cudaFree(ws->d_pri_res); cudaFree(ws->d_dua_res); cudaFree(ws->d_eps_pri); free(ws); return NULL; }

    // Allocate host arrays
    ws->h_status = (c_int*)calloc(batch_size, sizeof(c_int));
    ws->h_converged = (c_int*)calloc(batch_size, sizeof(c_int));
    ws->h_pri_res = (c_float*)calloc(batch_size, sizeof(c_float));
    ws->h_dua_res = (c_float*)calloc(batch_size, sizeof(c_float));

    if (!ws->h_status || !ws->h_converged || !ws->h_pri_res || !ws->h_dua_res) {
        free_gpu_termination_workspace(ws);
        return NULL;
    }

    ws->num_converged = 0;
    ws->num_pri_infeas = 0;
    ws->num_dua_infeas = 0;

    return ws;
}

void free_gpu_termination_workspace(GPUBatchTermination* ws) {
    if (ws) {
        if (ws->d_status) cudaFree(ws->d_status);
        if (ws->d_converged) cudaFree(ws->d_converged);
        if (ws->d_pri_res) cudaFree(ws->d_pri_res);
        if (ws->d_dua_res) cudaFree(ws->d_dua_res);
        if (ws->d_eps_pri) cudaFree(ws->d_eps_pri);
        if (ws->d_eps_dua) cudaFree(ws->d_eps_dua);
        if (ws->h_status) free(ws->h_status);
        if (ws->h_converged) free(ws->h_converged);
        if (ws->h_pri_res) free(ws->h_pri_res);
        if (ws->h_dua_res) free(ws->h_dua_res);
        free(ws);
    }
}

int gpu_termination_init(GPUBatchTermination* ws) {
    if (!ws) return -1;

    int num_blocks = (ws->batch_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    kernel_init_status<<<num_blocks, THREADS_PER_BLOCK>>>(
        ws->d_status, ws->d_converged, ws->batch_size
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in gpu_termination_init: %s\n", cudaGetErrorString(err));
        return -1;
    }

    ws->num_converged = 0;
    return 0;
}

int gpu_batch_check_convergence(
    GPUBatchTermination* ws,
    const c_float* d_pri_res,
    const c_float* d_dua_res,
    const c_float* d_eps_pri,
    const c_float* d_eps_dua,
    int* all_converged
) {
    if (!ws || !d_pri_res || !d_dua_res || !d_eps_pri || !d_eps_dua) return -1;

    int batch_size = ws->batch_size;
    int num_blocks = (batch_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Check convergence
    kernel_check_convergence<<<num_blocks, THREADS_PER_BLOCK>>>(
        d_pri_res, d_dua_res, d_eps_pri, d_eps_dua,
        ws->d_converged, ws->d_status, batch_size
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in gpu_batch_check_convergence: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Copy residuals to internal storage
    CUDA_CHECK(cudaMemcpy(ws->d_pri_res, d_pri_res, batch_size * sizeof(c_float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(ws->d_dua_res, d_dua_res, batch_size * sizeof(c_float), cudaMemcpyDeviceToDevice));

    // Count converged problems
    c_int* d_count;
    cudaMalloc(&d_count, sizeof(c_int));
    cudaMemset(d_count, 0, sizeof(c_int));

    int shared_mem_size = THREADS_PER_BLOCK * sizeof(c_int);
    kernel_count_converged<<<num_blocks, THREADS_PER_BLOCK, shared_mem_size>>>(
        ws->d_converged, d_count, batch_size
    );

    c_int count;
    cudaMemcpy(&count, d_count, sizeof(c_int), cudaMemcpyDeviceToHost);
    cudaFree(d_count);

    ws->num_converged = count;
    if (all_converged) {
        *all_converged = (count == batch_size) ? 1 : 0;
    }

    return 0;
}

int gpu_batch_check_convergence_scalar(
    GPUBatchTermination* ws,
    const c_float* d_pri_res,
    const c_float* d_dua_res,
    c_float eps_pri,
    c_float eps_dua,
    int* all_converged
) {
    if (!ws || !d_pri_res || !d_dua_res) return -1;

    int batch_size = ws->batch_size;
    int num_blocks = (batch_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Check convergence with scalar tolerances
    kernel_check_convergence_scalar<<<num_blocks, THREADS_PER_BLOCK>>>(
        d_pri_res, d_dua_res, eps_pri, eps_dua,
        ws->d_converged, ws->d_status, batch_size
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in gpu_batch_check_convergence_scalar: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Copy residuals to internal storage
    CUDA_CHECK(cudaMemcpy(ws->d_pri_res, d_pri_res, batch_size * sizeof(c_float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(ws->d_dua_res, d_dua_res, batch_size * sizeof(c_float), cudaMemcpyDeviceToDevice));

    // Count converged problems
    c_int* d_count;
    cudaMalloc(&d_count, sizeof(c_int));
    cudaMemset(d_count, 0, sizeof(c_int));

    int shared_mem_size = THREADS_PER_BLOCK * sizeof(c_int);
    kernel_count_converged<<<num_blocks, THREADS_PER_BLOCK, shared_mem_size>>>(
        ws->d_converged, d_count, batch_size
    );

    c_int count;
    cudaMemcpy(&count, d_count, sizeof(c_int), cudaMemcpyDeviceToHost);
    cudaFree(d_count);

    ws->num_converged = count;
    if (all_converged) {
        *all_converged = (count == batch_size) ? 1 : 0;
    }

    return 0;
}

int gpu_termination_mark_max_iter(GPUBatchTermination* ws) {
    if (!ws) return -1;

    int num_blocks = (ws->batch_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    kernel_mark_max_iter<<<num_blocks, THREADS_PER_BLOCK>>>(
        ws->d_status, ws->batch_size
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in gpu_termination_mark_max_iter: %s\n", cudaGetErrorString(err));
        return -1;
    }

    return 0;
}

int gpu_termination_mark_solved(GPUBatchTermination* ws) {
    // Already handled in check_convergence kernels
    return 0;
}

int gpu_termination_sync_to_host(GPUBatchTermination* ws) {
    if (!ws) return -1;

    int batch_size = ws->batch_size;

    CUDA_CHECK(cudaMemcpy(ws->h_status, ws->d_status, batch_size * sizeof(c_int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ws->h_converged, ws->d_converged, batch_size * sizeof(c_int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ws->h_pri_res, ws->d_pri_res, batch_size * sizeof(c_float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ws->h_dua_res, ws->d_dua_res, batch_size * sizeof(c_float), cudaMemcpyDeviceToHost));

    return 0;
}

c_int gpu_termination_get_num_converged(GPUBatchTermination* ws) {
    if (!ws) return 0;
    return ws->num_converged;
}

c_int gpu_termination_get_status(GPUBatchTermination* ws, int idx) {
    if (!ws || idx < 0 || idx >= ws->batch_size) return OSQP_UNSOLVED;
    return ws->h_status[idx];
}

void gpu_termination_get_residuals(
    GPUBatchTermination* ws,
    int idx,
    c_float* pri_res,
    c_float* dua_res
) {
    if (!ws || idx < 0 || idx >= ws->batch_size) return;
    if (pri_res) *pri_res = ws->h_pri_res[idx];
    if (dua_res) *dua_res = ws->h_dua_res[idx];
}

} // extern "C"
