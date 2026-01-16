/**
 * Batched ADMM GPU Kernels
 *
 * CUDA kernels for parallelizing ADMM update functions across multiple QP problems.
 * Works with the batched QDLDL linear solver in qdldl_batch_gpu.
 */

#ifndef ADMM_BATCH_GPU_H
#define ADMM_BATCH_GPU_H

#ifdef __cplusplus
extern "C" {
#endif

#include "types.h"
#include "qdldl_batch_gpu.h"
#include "spmv_batch_gpu.h"

/**
 * Extended workspace for batched ADMM iterations
 */
typedef struct {
    GPUBatchWorkspace* base_ws;  // Base workspace (for linear solver)

    int n;                       // Primal dimension
    int m;                       // Constraint dimension
    int batch_size;

    // ADMM iterates [batch_size * dim] - different per problem
    c_float* d_x;            // [batch_size * n] primal variable
    c_float* d_x_prev;       // [batch_size * n]
    c_float* d_z;            // [batch_size * m] slack variable
    c_float* d_z_prev;       // [batch_size * m]
    c_float* d_y;            // [batch_size * m] dual variable
    c_float* d_delta_x;      // [batch_size * n]
    c_float* d_delta_y;      // [batch_size * m]
    c_float* d_xtilde;       // [batch_size * n] xtilde from KKT solution (extracted)
    c_float* d_ztilde;       // [batch_size * m] ztilde = A * xtilde

    // xz_tilde is stored in base_ws->d_x (RHS/solution buffer)

    // Problem data - different per problem
    c_float* d_q;            // [batch_size * n] linear cost
    c_float* d_l;            // [batch_size * m] lower bounds
    c_float* d_u;            // [batch_size * m] upper bounds

    // Constraint matrix A (for computing ztilde = A * xtilde)
    GPUSparseMatrix* A;      // A matrix on GPU

    // Per-problem rho (for adaptive rho per problem)
    c_float* d_rho_batch;        // [batch_size] rho per problem
    c_float* d_rho_inv_batch;    // [batch_size] 1/rho per problem
    c_int*   d_rho_inv_diag_indices;  // [m] indices of -1/rho entries in KKT

    // Shared/default rho (used for initialization)
    c_float* d_rho_vec;      // [m] rho per constraint (or NULL if scalar)
    c_float* d_rho_inv_vec;  // [m] 1/rho per constraint (or NULL if scalar)
    c_float  rho;            // default scalar rho
    c_float  rho_inv;        // 1/rho
    c_float  sigma;          // regularization parameter
    c_float  alpha;          // relaxation parameter

    int rho_is_vec;          // 1 if using rho_vec, 0 if using scalar rho
    int use_per_problem_rho; // 1 if using per-problem rho
} GPUBatchADMMWorkspace;

/**
 * Allocate ADMM workspace for batched iterations
 * @param  base_ws     Base workspace from alloc_gpu_workspace()
 * @param  n           Primal dimension
 * @param  m           Constraint dimension
 * @param  batch_size  Number of problems
 * @return             ADMM workspace (caller must free with free_gpu_admm_workspace)
 */
GPUBatchADMMWorkspace* alloc_gpu_admm_workspace(
    GPUBatchWorkspace* base_ws,
    int n,
    int m,
    int batch_size
);

/**
 * Free ADMM workspace
 */
void free_gpu_admm_workspace(GPUBatchADMMWorkspace* ws);

/**
 * Set the A matrix for computing ztilde = A * xtilde
 * @param  ws          ADMM workspace
 * @param  h_Ap        Host: column pointers [n+1]
 * @param  h_Ai        Host: row indices [nnz]
 * @param  h_Ax        Host: values [nnz]
 * @param  nnz         Number of non-zeros in A
 * @return             0 on success
 */
int gpu_admm_set_A_matrix(
    GPUBatchADMMWorkspace* ws,
    const c_int* h_Ap,
    const c_int* h_Ai,
    const c_float* h_Ax,
    c_int nnz
);

/**
 * Copy problem data to device for batched ADMM
 * @param  ws          ADMM workspace
 * @param  h_q         Host: linear costs [batch_size * n]
 * @param  h_l         Host: lower bounds [batch_size * m]
 * @param  h_u         Host: upper bounds [batch_size * m]
 * @param  h_rho_vec   Host: rho per constraint [m] (NULL for scalar rho)
 * @param  rho         Scalar rho value
 * @param  sigma       Regularization parameter
 * @param  alpha       Relaxation parameter
 */
int gpu_admm_copy_problem_data(
    GPUBatchADMMWorkspace* ws,
    const c_float* h_q,
    const c_float* h_l,
    const c_float* h_u,
    const c_float* h_rho_vec,
    c_float rho,
    c_float sigma,
    c_float alpha
);

/**
 * Copy initial iterates to device
 * @param  ws        ADMM workspace
 * @param  h_x       Host: initial x [batch_size * n]
 * @param  h_z       Host: initial z [batch_size * m]
 * @param  h_y       Host: initial y [batch_size * m]
 */
int gpu_admm_copy_initial_iterates(
    GPUBatchADMMWorkspace* ws,
    const c_float* h_x,
    const c_float* h_z,
    const c_float* h_y
);

/**
 * Copy solution from device to host
 * @param  ws        ADMM workspace
 * @param  h_x       Host: solution x [batch_size * n]
 * @param  h_z       Host: solution z [batch_size * m]
 * @param  h_y       Host: solution y [batch_size * m]
 */
int gpu_admm_copy_solution(
    GPUBatchADMMWorkspace* ws,
    c_float* h_x,
    c_float* h_z,
    c_float* h_y
);

/**
 * Swap x <-> x_prev and z <-> z_prev (pointer swap, no copy)
 */
void gpu_batch_swap_iterates(GPUBatchADMMWorkspace* ws);

/**
 * Batched compute_rhs for KKT system
 * Computes RHS in base_ws->d_x:
 *   xtilde part: sigma * x_prev - q
 *   ztilde part: z_prev - rho_inv * y
 */
int gpu_batch_compute_rhs(
    const GPUFactorPattern* pattern,
    GPUBatchADMMWorkspace* ws
);

/**
 * Batched update_x:
 *   x = alpha * xtilde + (1 - alpha) * x_prev
 *   delta_x = x - x_prev
 */
int gpu_batch_update_x(GPUBatchADMMWorkspace* ws);

/**
 * Batched update_z:
 *   z = alpha * ztilde + (1 - alpha) * z_prev + rho_inv * y
 *   z = project(z, l, u)
 */
int gpu_batch_update_z(GPUBatchADMMWorkspace* ws);

/**
 * Batched update_y:
 *   delta_y = alpha * ztilde + (1 - alpha) * z_prev - z
 *   delta_y = rho * delta_y
 *   y = y + delta_y
 */
int gpu_batch_update_y(GPUBatchADMMWorkspace* ws);

/**
 * Complete batched ADMM iteration (all updates in one call)
 * @param  pattern    GPU factorization pattern
 * @param  ws         ADMM workspace
 * @param  admm_iter  Current ADMM iteration number
 * @return            0 on success
 */
int gpu_batch_admm_iteration(
    const GPUFactorPattern* pattern,
    GPUBatchADMMWorkspace* ws,
    int admm_iter
);

//=============================================================================
// Per-Problem Rho Support
//=============================================================================

/**
 * Initialize per-problem rho arrays and KKT diagonal indices
 * Call this once during setup to enable per-problem rho adaptation.
 *
 * @param  ws         ADMM workspace
 * @param  h_KKTp     Host: KKT column pointers [n+m+1]
 * @param  h_KKTi     Host: KKT row indices [nnz_KKT]
 * @param  n          Primal dimension
 * @param  m          Number of constraints
 * @param  initial_rho  Initial rho value for all problems
 * @return            0 on success
 */
int gpu_admm_init_per_problem_rho(
    GPUBatchADMMWorkspace* ws,
    const c_int* h_KKTp,
    const c_int* h_KKTi,
    c_int n,
    c_int m,
    c_float initial_rho
);

/**
 * Update rho for a specific problem
 * This updates both the ADMM workspace and the KKT matrix diagonal.
 * Call gpu_batch_factorize after updating rho values.
 *
 * @param  ws           ADMM workspace
 * @param  gpu_pattern  GPU factorization pattern
 * @param  problem_idx  Index of problem to update (0 to batch_size-1)
 * @param  new_rho      New rho value
 * @return              0 on success
 */
int gpu_admm_update_problem_rho(
    GPUBatchADMMWorkspace* ws,
    const GPUFactorPattern* gpu_pattern,
    c_int problem_idx,
    c_float new_rho
);

/**
 * Update rho for all problems at once
 * This updates both the ADMM workspace and all KKT matrix diagonals.
 * Call gpu_batch_factorize after updating rho values.
 *
 * @param  ws           ADMM workspace
 * @param  gpu_pattern  GPU factorization pattern
 * @param  h_new_rho    Host: new rho values [batch_size]
 * @return              0 on success
 */
int gpu_admm_update_all_rho(
    GPUBatchADMMWorkspace* ws,
    const GPUFactorPattern* gpu_pattern,
    const c_float* h_new_rho
);

/**
 * Get current rho values from device
 * @param  ws         ADMM workspace
 * @param  h_rho      Host buffer for rho values [batch_size]
 * @return            0 on success
 */
int gpu_admm_get_rho(
    GPUBatchADMMWorkspace* ws,
    c_float* h_rho
);

#ifdef __cplusplus
}
#endif

#endif // ADMM_BATCH_GPU_H
