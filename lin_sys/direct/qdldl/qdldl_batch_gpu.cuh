#ifndef QDLDL_BATCH_GPU_CUH
#define QDLDL_BATCH_GPU_CUH

#ifdef __cplusplus
extern "C" {
#endif

#include "qdldl_types.h"

/**
 * GPU-side pattern structure (pointers are device pointers)
 */
typedef struct {
    QDLDL_int   n;              // Matrix dimension
    QDLDL_int   nnz_KKT;        // Total nonzeros in KKT
    QDLDL_int   nnz_L;          // Total nonzeros in L

    // Device pointers (shared across all batch elements)
    QDLDL_int*  d_Ap;           // Column pointers [n+1]
    QDLDL_int*  d_Ai;           // Row indices [nnz_KKT]
    QDLDL_int*  d_etree;        // Elimination tree [n]
    QDLDL_int*  d_Lnz;          // Nonzeros per column of L [n]
    QDLDL_int*  d_Lp;           // L column pointers [n+1]
    QDLDL_int*  d_Li;           // L row indices [nnz_L]
    QDLDL_int*  d_P;            // Permutation vector [n]
} GPUFactorPattern;

/**
 * Workspace for batched GPU solves
 */
typedef struct {
    int batch_size;
    QDLDL_int n;
    QDLDL_int nnz_L;

    // Per-batch arrays on device [batch_size * size]
    QDLDL_float* d_Lx;          // [batch_size * nnz_L]
    QDLDL_float* d_D;           // [batch_size * n]
    QDLDL_float* d_Dinv;        // [batch_size * n]
    QDLDL_float* d_work;        // [batch_size * n] for solve permutation
    QDLDL_float* d_fwork;       // [batch_size * n] for factorization
    QDLDL_int*   d_iwork;       // [batch_size * 3 * n]
    QDLDL_bool*  d_bwork;       // [batch_size * n]
} GPUBatchWorkspace;

/**
 * Copy pattern from CPU FactorPattern to GPU
 * @param  cpu_pattern  Pattern from record_pattern_from_qdldl_solver()
 * @return              GPU pattern (caller must free with free_gpu_pattern)
 */
GPUFactorPattern* copy_pattern_to_gpu(const void* cpu_pattern);

/**
 * Free GPU pattern
 */
void free_gpu_pattern(GPUFactorPattern* gpu_pattern);

/**
 * Allocate workspace for batched GPU solves
 * @param  gpu_pattern  GPU pattern
 * @param  batch_size   Number of problems to solve in parallel
 * @return              Workspace (caller must free with free_gpu_workspace)
 */
GPUBatchWorkspace* alloc_gpu_workspace(const GPUFactorPattern* gpu_pattern, int batch_size);

/**
 * Free GPU workspace
 */
void free_gpu_workspace(GPUBatchWorkspace* workspace);

/**
 * Batched factorization on GPU
 * @param  gpu_pattern  GPU pattern
 * @param  workspace    GPU workspace
 * @param  d_Ax_batch   Device pointer to batched KKT values [batch_size * nnz_KKT]
 * @param  batch_size   Number of problems
 * @return              0 on success
 */
int gpu_batch_factor(
    const GPUFactorPattern* gpu_pattern,
    GPUBatchWorkspace*      workspace,
    const QDLDL_float*      d_Ax_batch,
    int                     batch_size
);

/**
 * Batched solve on GPU (must call gpu_batch_factor first)
 * @param  gpu_pattern  GPU pattern
 * @param  workspace    GPU workspace (with factorization from gpu_batch_factor)
 * @param  d_x_batch    Device pointer to batched RHS/solution [batch_size * n]
 * @param  batch_size   Number of problems
 * @return              0 on success
 */
int gpu_batch_solve(
    const GPUFactorPattern* gpu_pattern,
    GPUBatchWorkspace*      workspace,
    QDLDL_float*            d_x_batch,
    int                     batch_size
);

/**
 * Combined batched factor + solve on GPU
 * @param  gpu_pattern  GPU pattern
 * @param  workspace    GPU workspace
 * @param  d_Ax_batch   Device pointer to batched KKT values [batch_size * nnz_KKT]
 * @param  d_x_batch    Device pointer to batched RHS/solution [batch_size * n]
 * @param  batch_size   Number of problems
 * @return              0 on success
 */
int gpu_batch_factor_solve(
    const GPUFactorPattern* gpu_pattern,
    GPUBatchWorkspace*      workspace,
    const QDLDL_float*      d_Ax_batch,
    QDLDL_float*            d_x_batch,
    int                     batch_size
);

#ifdef __cplusplus
}
#endif

#endif // QDLDL_BATCH_GPU_CUH
