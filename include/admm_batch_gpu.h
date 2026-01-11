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

    // xz_tilde is stored in base_ws->d_x (RHS/solution buffer)

    // Problem data - different per problem
    c_float* d_q;            // [batch_size * n] linear cost
    c_float* d_l;            // [batch_size * m] lower bounds
    c_float* d_u;            // [batch_size * m] upper bounds

    // Shared across all problems
    c_float* d_rho_vec;      // [m] rho per constraint (or NULL if scalar)
    c_float* d_rho_inv_vec;  // [m] 1/rho per constraint (or NULL if scalar)
    c_float  rho;            // scalar rho (used if d_rho_vec is NULL)
    c_float  rho_inv;        // 1/rho
    c_float  sigma;          // regularization parameter
    c_float  alpha;          // relaxation parameter

    int rho_is_vec;          // 1 if using rho_vec, 0 if using scalar rho
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

#ifdef __cplusplus
}
#endif

#endif // ADMM_BATCH_GPU_H
