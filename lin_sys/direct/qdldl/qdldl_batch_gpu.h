#ifndef QDLDL_BATCH_GPU_H
#define QDLDL_BATCH_GPU_H

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

    // Device memory for KKT values (allocated separately)
    QDLDL_float* d_Ax;          // [batch_size * nnz_KKT]

    // Persistent device buffer for RHS/solution (GPU-resident mode)
    QDLDL_float* d_x;           // [batch_size * n]

    // CUDA Graph for reduced kernel launch overhead
    void* cuda_graph;           // cudaGraph_t (opaque pointer for C compatibility)
    void* cuda_graph_exec;      // cudaGraphExec_t
    int   graph_captured;       // 1 if graph is ready to use
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
 * Batched factorization on GPU (device memory input)
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
 * Batched factorization on GPU (host memory input)
 * Copies data to device, factorizes, and stores results in workspace
 * @param  gpu_pattern  GPU pattern
 * @param  workspace    GPU workspace
 * @param  h_Ax_batch   Host pointer to batched KKT values [batch_size * nnz_KKT]
 * @param  batch_size   Number of problems
 * @return              0 on success
 */
int gpu_batch_factor_host(
    const GPUFactorPattern* gpu_pattern,
    GPUBatchWorkspace*      workspace,
    const QDLDL_float*      h_Ax_batch,
    int                     batch_size
);

/**
 * Batched factorization on GPU with single KKT broadcast
 * Takes a single KKT matrix and broadcasts it to all batch elements
 * @param  gpu_pattern  GPU pattern
 * @param  workspace    GPU workspace
 * @param  h_Ax         Host pointer to single KKT values [nnz_KKT]
 * @param  batch_size   Number of problems (all use same KKT)
 * @return              0 on success
 */
int gpu_batch_factor_broadcast_host(
    const GPUFactorPattern* gpu_pattern,
    GPUBatchWorkspace*      workspace,
    const QDLDL_float*      h_Ax,
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
 * Batched solve on GPU with host memory interface
 * Copies RHS to device, solves, copies solution back
 * @param  gpu_pattern  GPU pattern
 * @param  workspace    GPU workspace (with factorization from gpu_batch_factor)
 * @param  h_x_batch    Host pointer to batched RHS/solution [batch_size * n]
 * @param  batch_size   Number of problems
 * @return              0 on success
 */
int gpu_batch_solve_host(
    const GPUFactorPattern* gpu_pattern,
    GPUBatchWorkspace*      workspace,
    QDLDL_float*            h_x_batch,
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

/**
 * Debug: Copy L and Dinv from GPU to host for comparison
 * @param  workspace    GPU workspace with factorization
 * @param  h_Lx         Host buffer for Lx [nnz_L]
 * @param  h_Dinv       Host buffer for Dinv [n]
 * @param  batch_idx    Which batch element to copy (0 for first)
 */
void gpu_get_factor_values(
    const GPUFactorPattern* gpu_pattern,
    const GPUBatchWorkspace* workspace,
    QDLDL_float*            h_Lx,
    QDLDL_float*            h_Dinv,
    int                     batch_idx
);

/**
 * Debug: Copy permutation from GPU to host
 */
void gpu_get_permutation(
    const GPUFactorPattern* gpu_pattern,
    QDLDL_int*              h_P
);

//=============================================================================
// GPU-Resident Data Functions (avoid host<->device transfers)
//=============================================================================

/**
 * Copy KKT batch data from host to persistent device buffer
 * Call this once, then use gpu_batch_factor_device for repeated factorizations
 */
int gpu_copy_kkt_to_device(
    const GPUFactorPattern* gpu_pattern,
    GPUBatchWorkspace*      workspace,
    const QDLDL_float*      h_Ax_batch,
    int                     batch_size
);

/**
 * Copy RHS batch data from host to persistent device buffer
 * Call this once, then use gpu_batch_solve_device for repeated solves
 */
int gpu_copy_rhs_to_device(
    const GPUFactorPattern* gpu_pattern,
    GPUBatchWorkspace*      workspace,
    const QDLDL_float*      h_x_batch,
    int                     batch_size
);

/**
 * Copy solution batch data from device buffer to host
 */
int gpu_copy_solution_to_host(
    const GPUFactorPattern* gpu_pattern,
    GPUBatchWorkspace*      workspace,
    QDLDL_float*            h_x_batch,
    int                     batch_size
);

/**
 * Factorize using data already on device (no host transfer)
 * Must call gpu_copy_kkt_to_device first
 */
int gpu_batch_factor_device(
    const GPUFactorPattern* gpu_pattern,
    GPUBatchWorkspace*      workspace,
    int                     batch_size
);

/**
 * Solve using data already on device (no host transfer)
 * Must call gpu_copy_rhs_to_device first, factorization must be done
 */
int gpu_batch_solve_device(
    const GPUFactorPattern* gpu_pattern,
    GPUBatchWorkspace*      workspace,
    int                     batch_size
);

//=============================================================================
// CUDA Graph Functions (reduced kernel launch overhead)
//=============================================================================

/**
 * Capture factor+solve sequence into a CUDA graph
 * After capturing, use gpu_batch_factor_solve_graph for faster execution
 * @return 0 on success
 */
int gpu_capture_graph(
    const GPUFactorPattern* gpu_pattern,
    GPUBatchWorkspace*      workspace,
    int                     batch_size
);

/**
 * Execute factor+solve using captured CUDA graph (lowest overhead)
 * Must call gpu_capture_graph first
 * Data must already be on device (use gpu_copy_kkt_to_device, gpu_copy_rhs_to_device)
 * @return 0 on success
 */
int gpu_batch_factor_solve_graph(
    const GPUFactorPattern* gpu_pattern,
    GPUBatchWorkspace*      workspace,
    int                     batch_size
);

/**
 * Free CUDA graph resources
 */
void gpu_free_graph(GPUBatchWorkspace* workspace);

//=============================================================================
// Per-Problem Rho Support
//=============================================================================

/**
 * Update the -1/rho diagonal entries in KKT matrices per-problem
 * This allows different rho values for each problem in the batch.
 *
 * The KKT matrix has structure:
 *   [P + sigma*I,  A' ]
 *   [A,           -1/rho * I]
 *
 * This function updates the -1/rho diagonal in the bottom-right block.
 *
 * @param  gpu_pattern    GPU pattern (contains KKT structure info)
 * @param  workspace      GPU workspace with d_Ax already populated
 * @param  d_rho          Device pointer to per-problem rho values [batch_size]
 * @param  n              Primal dimension (first n columns have P+sigma*I)
 * @param  m              Number of constraints (last m columns have -1/rho diag)
 * @param  d_rho_inv_diag_indices  Device pointer to indices of -1/rho entries in KKT [m]
 * @param  batch_size     Number of problems
 * @return                0 on success
 */
int gpu_batch_update_rho(
    const GPUFactorPattern* gpu_pattern,
    GPUBatchWorkspace*      workspace,
    const QDLDL_float*      d_rho,
    QDLDL_int               n,
    QDLDL_int               m,
    const QDLDL_int*        d_rho_inv_diag_indices,
    int                     batch_size
);

#ifdef __cplusplus
}
#endif

#endif // QDLDL_BATCH_GPU_H
