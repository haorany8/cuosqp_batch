#include "qdldl_batch_gpu.h"
#include "qdldl_batch_gpu.cuh"
#include "qdldl_symbolic.h"
#include <cuda_runtime.h>
#include <stdio.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            return -1; \
        } \
    } while(0)

#define CUDA_CHECK_VOID(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            return; \
        } \
    } while(0)

//=============================================================================
// Device functions: QDLDL factor and solve (run by each thread)
//=============================================================================

#define QDLDL_UNKNOWN (-1)
#define QDLDL_USED    (1)
#define QDLDL_UNUSED  (0)

/**
 * Device function: LDL factorization for a single matrix
 * Each thread calls this with its own Ax, Lx, D, Dinv, and workspace
 */
__device__ QDLDL_int device_qdldl_factor(
    const QDLDL_int    n,
    const QDLDL_int*   Ap,
    const QDLDL_int*   Ai,
    const QDLDL_float* Ax,
    const QDLDL_int*   Lp,
    const QDLDL_int*   Li,
    QDLDL_float*       Lx,
    QDLDL_float*       D,
    QDLDL_float*       Dinv,
    const QDLDL_int*   Lnz,
    const QDLDL_int*   etree,
    QDLDL_bool*        bwork,
    QDLDL_int*         iwork,
    QDLDL_float*       fwork
) {
    QDLDL_int i, j, k, nnzY, bidx, cidx, nextIdx, nnzE, tmpIdx;
    QDLDL_int positiveValuesInD = 0;

    // Partition workspace
    QDLDL_bool*  yMarkers        = bwork;
    QDLDL_int*   yIdx            = iwork;
    QDLDL_int*   elimBuffer      = iwork + n;
    QDLDL_int*   LNextSpaceInCol = iwork + n * 2;
    QDLDL_float* yVals           = fwork;

    // Initialize
    for (i = 0; i < n; i++) {
        yMarkers[i]        = QDLDL_UNUSED;
        yVals[i]           = 0.0;
        D[i]               = 0.0;
        LNextSpaceInCol[i] = Lp[i];
    }

    // First diagonal element
    D[0] = Ax[0];
    if (D[0] == 0.0) return -1;
    if (D[0] > 0.0) positiveValuesInD++;
    Dinv[0] = 1.0 / D[0];

    // Main factorization loop
    for (k = 1; k < n; k++) {
        nnzY = 0;

        tmpIdx = Ap[k + 1];
        for (i = Ap[k]; i < tmpIdx; i++) {
            bidx = Ai[i];

            if (bidx == k) {
                D[k] = Ax[i];
                continue;
            }

            yVals[bidx] = Ax[i];
            nextIdx = bidx;

            if (yMarkers[nextIdx] == QDLDL_UNUSED) {
                yMarkers[nextIdx] = QDLDL_USED;
                elimBuffer[0]     = nextIdx;
                nnzE              = 1;
                nextIdx = etree[bidx];

                while (nextIdx != QDLDL_UNKNOWN && nextIdx < k) {
                    if (yMarkers[nextIdx] == QDLDL_USED) break;
                    yMarkers[nextIdx] = QDLDL_USED;
                    elimBuffer[nnzE]  = nextIdx;
                    nnzE++;
                    nextIdx = etree[nextIdx];
                }

                while (nnzE) {
                    yIdx[nnzY++] = elimBuffer[--nnzE];
                }
            }
        }

        // Compute L values for row k
        for (i = nnzY - 1; i >= 0; i--) {
            cidx = yIdx[i];
            tmpIdx = LNextSpaceInCol[cidx];
            QDLDL_float yVals_cidx = yVals[cidx];

            for (j = Lp[cidx]; j < tmpIdx; j++) {
                yVals[Li[j]] -= Lx[j] * yVals_cidx;
            }

            Lx[tmpIdx] = yVals_cidx * Dinv[cidx];
            D[k] -= yVals_cidx * Lx[tmpIdx];
            LNextSpaceInCol[cidx]++;

            yVals[cidx]   = 0.0;
            yMarkers[cidx] = QDLDL_UNUSED;
        }

        if (D[k] == 0.0) return -1;
        if (D[k] > 0.0) positiveValuesInD++;
        Dinv[k] = 1.0 / D[k];
    }

    return positiveValuesInD;
}

/**
 * Device function: Forward solve (L + I)x = b
 */
__device__ void device_qdldl_Lsolve(
    const QDLDL_int    n,
    const QDLDL_int*   Lp,
    const QDLDL_int*   Li,
    const QDLDL_float* Lx,
    QDLDL_float*       x
) {
    for (QDLDL_int i = 0; i < n; i++) {
        QDLDL_float val = x[i];
        for (QDLDL_int j = Lp[i]; j < Lp[i + 1]; j++) {
            x[Li[j]] -= Lx[j] * val;
        }
    }
}

/**
 * Device function: Backward solve (L + I)'x = b
 */
__device__ void device_qdldl_Ltsolve(
    const QDLDL_int    n,
    const QDLDL_int*   Lp,
    const QDLDL_int*   Li,
    const QDLDL_float* Lx,
    QDLDL_float*       x
) {
    for (QDLDL_int i = n - 1; i >= 0; i--) {
        QDLDL_float val = x[i];
        for (QDLDL_int j = Lp[i]; j < Lp[i + 1]; j++) {
            val -= Lx[j] * x[Li[j]];
        }
        x[i] = val;
    }
}

/**
 * Device function: Full LDL solve
 */
__device__ void device_qdldl_solve(
    const QDLDL_int    n,
    const QDLDL_int*   Lp,
    const QDLDL_int*   Li,
    const QDLDL_float* Lx,
    const QDLDL_float* Dinv,
    QDLDL_float*       x
) {
    device_qdldl_Lsolve(n, Lp, Li, Lx, x);
    for (QDLDL_int i = 0; i < n; i++) {
        x[i] *= Dinv[i];
    }
    device_qdldl_Ltsolve(n, Lp, Li, Lx, x);
}

//=============================================================================
// CUDA Kernels
//=============================================================================

/**
 * Kernel: Batched factorization
 * Each thread handles one matrix in the batch
 */
__global__ void kernel_batch_factor(
    const QDLDL_int    n,
    const QDLDL_int    nnz_L,
    const QDLDL_int*   Ap,
    const QDLDL_int*   Ai,
    const QDLDL_int*   Lp,
    const QDLDL_int*   Li,
    const QDLDL_int*   Lnz,
    const QDLDL_int*   etree,
    const QDLDL_float* Ax_batch,      // [batch_size * nnz_KKT]
    QDLDL_float*       Lx_batch,      // [batch_size * nnz_L]
    QDLDL_float*       D_batch,       // [batch_size * n]
    QDLDL_float*       Dinv_batch,    // [batch_size * n]
    QDLDL_bool*        bwork_batch,   // [batch_size * n]
    QDLDL_int*         iwork_batch,   // [batch_size * 3 * n]
    QDLDL_float*       fwork_batch,   // [batch_size * n]
    QDLDL_int          nnz_KKT,
    int                batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    // Pointers to this thread's data
    const QDLDL_float* Ax    = Ax_batch    + idx * nnz_KKT;
    QDLDL_float*       Lx    = Lx_batch    + idx * nnz_L;
    QDLDL_float*       D     = D_batch     + idx * n;
    QDLDL_float*       Dinv  = Dinv_batch  + idx * n;
    QDLDL_bool*        bwork = bwork_batch + idx * n;
    QDLDL_int*         iwork = iwork_batch + idx * 3 * n;
    QDLDL_float*       fwork = fwork_batch + idx * n;

    device_qdldl_factor(n, Ap, Ai, Ax, Lp, Li, Lx, D, Dinv, Lnz, etree, bwork, iwork, fwork);
}

/**
 * Kernel: Batched solve (assumes factorization already done)
 * Each thread handles one system in the batch
 */
__global__ void kernel_batch_solve(
    const QDLDL_int    n,
    const QDLDL_int*   Lp,
    const QDLDL_int*   Li,
    const QDLDL_int*   P,
    const QDLDL_float* Lx_batch,      // [batch_size * nnz_L]
    const QDLDL_float* Dinv_batch,    // [batch_size * n]
    QDLDL_float*       x_batch,       // [batch_size * n] in: RHS, out: solution
    QDLDL_float*       work_batch,    // [batch_size * n]
    QDLDL_int          nnz_L,
    int                batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    // Pointers to this thread's data
    const QDLDL_float* Lx   = Lx_batch   + idx * nnz_L;
    const QDLDL_float* Dinv = Dinv_batch + idx * n;
    QDLDL_float*       x    = x_batch    + idx * n;
    QDLDL_float*       work = work_batch + idx * n;

    // Permute: work = P * x
    for (QDLDL_int j = 0; j < n; j++) {
        work[j] = x[P[j]];
    }

    // Solve LDL'y = work
    device_qdldl_solve(n, Lp, Li, Lx, Dinv, work);

    // Unpermute: x = P' * work
    for (QDLDL_int j = 0; j < n; j++) {
        x[P[j]] = work[j];
    }
}

//=============================================================================
// Host functions
//=============================================================================

extern "C" {

GPUFactorPattern* copy_pattern_to_gpu(const void* cpu_pattern_void) {
    const FactorPattern* cpu_pattern = (const FactorPattern*)cpu_pattern_void;

    GPUFactorPattern* gpu = (GPUFactorPattern*)malloc(sizeof(GPUFactorPattern));
    if (!gpu) return NULL;

    gpu->n       = cpu_pattern->n;
    gpu->nnz_KKT = cpu_pattern->nnz_KKT;
    gpu->nnz_L   = cpu_pattern->nnz_L;

    QDLDL_int n       = cpu_pattern->n;
    QDLDL_int nnz_KKT = cpu_pattern->nnz_KKT;
    QDLDL_int nnz_L   = cpu_pattern->nnz_L;

    // Allocate and copy pattern arrays to GPU
    cudaMalloc(&gpu->d_Ap,    (n + 1) * sizeof(QDLDL_int));
    cudaMalloc(&gpu->d_Ai,    nnz_KKT * sizeof(QDLDL_int));
    cudaMalloc(&gpu->d_etree, n * sizeof(QDLDL_int));
    cudaMalloc(&gpu->d_Lnz,   n * sizeof(QDLDL_int));
    cudaMalloc(&gpu->d_Lp,    (n + 1) * sizeof(QDLDL_int));
    cudaMalloc(&gpu->d_Li,    nnz_L * sizeof(QDLDL_int));
    cudaMalloc(&gpu->d_P,     n * sizeof(QDLDL_int));

    cudaMemcpy(gpu->d_Ap,    cpu_pattern->Ap,    (n + 1) * sizeof(QDLDL_int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu->d_Ai,    cpu_pattern->Ai,    nnz_KKT * sizeof(QDLDL_int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu->d_etree, cpu_pattern->etree, n * sizeof(QDLDL_int),       cudaMemcpyHostToDevice);
    cudaMemcpy(gpu->d_Lnz,   cpu_pattern->Lnz,   n * sizeof(QDLDL_int),       cudaMemcpyHostToDevice);
    cudaMemcpy(gpu->d_Lp,    cpu_pattern->Lp,    (n + 1) * sizeof(QDLDL_int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu->d_Li,    cpu_pattern->Li,    nnz_L * sizeof(QDLDL_int),   cudaMemcpyHostToDevice);
    cudaMemcpy(gpu->d_P,     cpu_pattern->P,     n * sizeof(QDLDL_int),       cudaMemcpyHostToDevice);

    return gpu;
}

void free_gpu_pattern(GPUFactorPattern* gpu) {
    if (gpu) {
        if (gpu->d_Ap)    cudaFree(gpu->d_Ap);
        if (gpu->d_Ai)    cudaFree(gpu->d_Ai);
        if (gpu->d_etree) cudaFree(gpu->d_etree);
        if (gpu->d_Lnz)   cudaFree(gpu->d_Lnz);
        if (gpu->d_Lp)    cudaFree(gpu->d_Lp);
        if (gpu->d_Li)    cudaFree(gpu->d_Li);
        if (gpu->d_P)     cudaFree(gpu->d_P);
        free(gpu);
    }
}

GPUBatchWorkspace* alloc_gpu_workspace(const GPUFactorPattern* gpu_pattern, int batch_size) {
    GPUBatchWorkspace* ws = (GPUBatchWorkspace*)malloc(sizeof(GPUBatchWorkspace));
    if (!ws) return NULL;

    ws->batch_size = batch_size;
    ws->n          = gpu_pattern->n;
    ws->nnz_L      = gpu_pattern->nnz_L;

    QDLDL_int n     = gpu_pattern->n;
    QDLDL_int nnz_L = gpu_pattern->nnz_L;

    cudaMalloc(&ws->d_Lx,    batch_size * nnz_L * sizeof(QDLDL_float));
    cudaMalloc(&ws->d_D,     batch_size * n * sizeof(QDLDL_float));
    cudaMalloc(&ws->d_Dinv,  batch_size * n * sizeof(QDLDL_float));
    cudaMalloc(&ws->d_work,  batch_size * n * sizeof(QDLDL_float));
    cudaMalloc(&ws->d_fwork, batch_size * n * sizeof(QDLDL_float));
    cudaMalloc(&ws->d_iwork, batch_size * 3 * n * sizeof(QDLDL_int));
    cudaMalloc(&ws->d_bwork, batch_size * n * sizeof(QDLDL_bool));

    // d_Ax is allocated on demand by gpu_batch_factor_host
    ws->d_Ax = NULL;

    return ws;
}

void free_gpu_workspace(GPUBatchWorkspace* ws) {
    if (ws) {
        if (ws->d_Lx)    cudaFree(ws->d_Lx);
        if (ws->d_D)     cudaFree(ws->d_D);
        if (ws->d_Dinv)  cudaFree(ws->d_Dinv);
        if (ws->d_work)  cudaFree(ws->d_work);
        if (ws->d_fwork) cudaFree(ws->d_fwork);
        if (ws->d_iwork) cudaFree(ws->d_iwork);
        if (ws->d_bwork) cudaFree(ws->d_bwork);
        if (ws->d_Ax)    cudaFree(ws->d_Ax);
        free(ws);
    }
}

int gpu_batch_factor(
    const GPUFactorPattern* gpu_pattern,
    GPUBatchWorkspace*      workspace,
    const QDLDL_float*      d_Ax_batch,
    int                     batch_size
) {
    int threads_per_block = 256;
    int num_blocks = (batch_size + threads_per_block - 1) / threads_per_block;

    kernel_batch_factor<<<num_blocks, threads_per_block>>>(
        gpu_pattern->n,
        gpu_pattern->nnz_L,
        gpu_pattern->d_Ap,
        gpu_pattern->d_Ai,
        gpu_pattern->d_Lp,
        gpu_pattern->d_Li,
        gpu_pattern->d_Lnz,
        gpu_pattern->d_etree,
        d_Ax_batch,
        workspace->d_Lx,
        workspace->d_D,
        workspace->d_Dinv,
        workspace->d_bwork,
        workspace->d_iwork,
        workspace->d_fwork,
        gpu_pattern->nnz_KKT,
        batch_size
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    return 0;
}

int gpu_batch_solve(
    const GPUFactorPattern* gpu_pattern,
    GPUBatchWorkspace*      workspace,
    QDLDL_float*            d_x_batch,
    int                     batch_size
) {
    int threads_per_block = 256;
    int num_blocks = (batch_size + threads_per_block - 1) / threads_per_block;

    kernel_batch_solve<<<num_blocks, threads_per_block>>>(
        gpu_pattern->n,
        gpu_pattern->d_Lp,
        gpu_pattern->d_Li,
        gpu_pattern->d_P,
        workspace->d_Lx,
        workspace->d_Dinv,
        d_x_batch,
        workspace->d_work,
        gpu_pattern->nnz_L,
        batch_size
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    return 0;
}

int gpu_batch_factor_solve(
    const GPUFactorPattern* gpu_pattern,
    GPUBatchWorkspace*      workspace,
    const QDLDL_float*      d_Ax_batch,
    QDLDL_float*            d_x_batch,
    int                     batch_size
) {
    int ret = gpu_batch_factor(gpu_pattern, workspace, d_Ax_batch, batch_size);
    if (ret != 0) return ret;

    ret = gpu_batch_solve(gpu_pattern, workspace, d_x_batch, batch_size);
    return ret;
}

int gpu_batch_factor_host(
    const GPUFactorPattern* gpu_pattern,
    GPUBatchWorkspace*      workspace,
    const QDLDL_float*      h_Ax_batch,
    int                     batch_size
) {
    QDLDL_int nnz_KKT = gpu_pattern->nnz_KKT;
    size_t size = batch_size * nnz_KKT * sizeof(QDLDL_float);

    // Allocate device memory if not already allocated
    if (!workspace->d_Ax) {
        cudaError_t err = cudaMalloc(&workspace->d_Ax, size);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA malloc error: %s\n", cudaGetErrorString(err));
            return -1;
        }
    }

    // Copy KKT values to device
    cudaError_t err = cudaMemcpy(workspace->d_Ax, h_Ax_batch, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA memcpy error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Call the device memory version
    return gpu_batch_factor(gpu_pattern, workspace, workspace->d_Ax, batch_size);
}

int gpu_batch_factor_broadcast_host(
    const GPUFactorPattern* gpu_pattern,
    GPUBatchWorkspace*      workspace,
    const QDLDL_float*      h_Ax,
    int                     batch_size
) {
    QDLDL_int nnz_KKT = gpu_pattern->nnz_KKT;
    size_t single_size = nnz_KKT * sizeof(QDLDL_float);
    size_t total_size = batch_size * single_size;

    // Allocate device memory if not already allocated
    if (!workspace->d_Ax) {
        cudaError_t err = cudaMalloc(&workspace->d_Ax, total_size);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA malloc error: %s\n", cudaGetErrorString(err));
            return -1;
        }
    }

    // Broadcast: copy the single KKT to all batch elements
    for (int i = 0; i < batch_size; i++) {
        QDLDL_float* dst = workspace->d_Ax + i * nnz_KKT;
        cudaError_t err = cudaMemcpy(dst, h_Ax, single_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA memcpy error: %s\n", cudaGetErrorString(err));
            return -1;
        }
    }

    // Call the device memory version
    return gpu_batch_factor(gpu_pattern, workspace, workspace->d_Ax, batch_size);
}

int gpu_batch_solve_host(
    const GPUFactorPattern* gpu_pattern,
    GPUBatchWorkspace*      workspace,
    QDLDL_float*            h_x_batch,
    int                     batch_size
) {
    QDLDL_int n = gpu_pattern->n;
    size_t size = batch_size * n * sizeof(QDLDL_float);

    // Allocate temporary device buffer
    QDLDL_float* d_x_batch;
    cudaError_t err = cudaMalloc(&d_x_batch, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Copy RHS to device
    err = cudaMemcpy(d_x_batch, h_x_batch, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_x_batch);
        fprintf(stderr, "CUDA memcpy error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Solve on device
    int ret = gpu_batch_solve(gpu_pattern, workspace, d_x_batch, batch_size);

    if (ret == 0) {
        // Copy solution back to host
        err = cudaMemcpy(h_x_batch, d_x_batch, size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            ret = -1;
        }
    }

    cudaFree(d_x_batch);
    return ret;
}

void gpu_get_factor_values(
    const GPUFactorPattern* gpu_pattern,
    const GPUBatchWorkspace* workspace,
    QDLDL_float*            h_Lx,
    QDLDL_float*            h_Dinv,
    int                     batch_idx
) {
    QDLDL_int n = gpu_pattern->n;
    QDLDL_int nnz_L = gpu_pattern->nnz_L;

    // Copy Lx for specified batch element
    const QDLDL_float* d_Lx = workspace->d_Lx + batch_idx * nnz_L;
    cudaMemcpy(h_Lx, d_Lx, nnz_L * sizeof(QDLDL_float), cudaMemcpyDeviceToHost);

    // Copy Dinv for specified batch element
    const QDLDL_float* d_Dinv = workspace->d_Dinv + batch_idx * n;
    cudaMemcpy(h_Dinv, d_Dinv, n * sizeof(QDLDL_float), cudaMemcpyDeviceToHost);
}

void gpu_get_permutation(
    const GPUFactorPattern* gpu_pattern,
    QDLDL_int*              h_P
) {
    cudaMemcpy(h_P, gpu_pattern->d_P, gpu_pattern->n * sizeof(QDLDL_int), cudaMemcpyDeviceToHost);
}

} // extern "C"
