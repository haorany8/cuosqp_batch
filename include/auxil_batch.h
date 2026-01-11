/**
 * Batched ADMM Auxiliary Functions
 *
 * This file contains batch versions of the ADMM update functions from auxil.c.
 * These functions operate on multiple QP problems in parallel using GPU.
 *
 * The original single-problem functions in auxil.c are preserved for comparison.
 */

#ifndef AUXIL_BATCH_H
#define AUXIL_BATCH_H

#ifdef __cplusplus
extern "C" {
#endif

#include "types.h"
#include "admm_batch_gpu.h"  // includes qdldl_batch_gpu.h

/**
 * Batched OSQP Solver structure
 * Contains data for solving multiple QPs with the same structure in parallel
 */
typedef struct {
    // Problem dimensions
    c_int n;                    // Primal dimension
    c_int m;                    // Constraint dimension
    c_int batch_size;           // Number of problems in batch

    // GPU pattern and workspace
    GPUFactorPattern*      gpu_pattern;  // Factorization pattern (shared)
    GPUBatchWorkspace*     base_ws;      // Base GPU workspace
    GPUBatchADMMWorkspace* admm_ws;      // ADMM workspace

    // Host-side data buffers (for data transfer)
    c_float* h_q;               // [batch_size * n] linear costs
    c_float* h_l;               // [batch_size * m] lower bounds
    c_float* h_u;               // [batch_size * m] upper bounds
    c_float* h_x;               // [batch_size * n] primal solution
    c_float* h_z;               // [batch_size * m] slack variable
    c_float* h_y;               // [batch_size * m] dual variable

    // KKT data
    c_float* h_kkt_Ax;          // [nnz_KKT] KKT values (same for all)
    c_int    nnz_KKT;           // Number of nonzeros in KKT

    // ADMM parameters
    c_float rho;                // Penalty parameter
    c_float sigma;              // Regularization
    c_float alpha;              // Relaxation

    // Status
    c_int is_initialized;       // 1 if solver is ready
    c_int is_factorized;        // 1 if factorization is done
} OSQPBatchSolver;

/**
 * Allocate a batched OSQP solver
 *
 * @param n           Primal dimension
 * @param m           Constraint dimension
 * @param batch_size  Number of problems
 * @return            Allocated solver (NULL on failure)
 */
OSQPBatchSolver* osqp_batch_alloc(c_int n, c_int m, c_int batch_size);

/**
 * Free a batched OSQP solver
 */
void osqp_batch_free(OSQPBatchSolver* solver);

/**
 * Set the KKT pattern from a FactorPattern
 * This must be called after osqp_batch_alloc and before osqp_batch_setup_data
 *
 * @param solver   Batched solver
 * @param pattern  CPU FactorPattern from QDLDL
 * @return         0 on success
 */
c_int osqp_batch_set_pattern(OSQPBatchSolver* solver, const void* pattern);

/**
 * Setup problem data for all batch problems
 *
 * @param solver  Batched solver
 * @param q       Linear costs [batch_size * n] (different per problem)
 * @param l       Lower bounds [batch_size * m] (different per problem)
 * @param u       Upper bounds [batch_size * m] (different per problem)
 * @param Ax      KKT values [nnz_KKT] (same for all - P is same)
 * @param rho     Penalty parameter
 * @param sigma   Regularization
 * @param alpha   Relaxation
 * @return        0 on success
 */
c_int osqp_batch_setup_data(
    OSQPBatchSolver* solver,
    const c_float* q,
    const c_float* l,
    const c_float* u,
    const c_float* Ax,
    c_float rho,
    c_float sigma,
    c_float alpha
);

/**
 * Set initial iterates for warm start (internal version with z)
 *
 * @param solver  Batched solver
 * @param x       Initial x [batch_size * n] (NULL for cold start)
 * @param z       Initial z [batch_size * m] (NULL for cold start)
 * @param y       Initial y [batch_size * m] (NULL for cold start)
 * @return        0 on success
 */
c_int osqp_batch_warm_start_internal(
    OSQPBatchSolver* solver,
    const c_float* x,
    const c_float* z,
    const c_float* y
);

/**
 * Cold start all problems (zero initialization)
 *
 * @param solver  Batched solver
 * @return        0 on success
 */
c_int osqp_batch_cold_start(OSQPBatchSolver* solver);

/**
 * Perform one batched ADMM iteration
 * Executes: swap -> compute_rhs -> solve -> update_x -> update_z -> update_y
 *
 * @param solver     Batched solver
 * @param admm_iter  Current ADMM iteration number
 * @return           0 on success
 */
c_int osqp_batch_admm_iteration(OSQPBatchSolver* solver, c_int admm_iter);

/**
 * Perform factorization of KKT matrix
 * Only needs to be called once since P is the same for all problems
 *
 * @param solver  Batched solver
 * @return        0 on success
 */
c_int osqp_batch_factorize(OSQPBatchSolver* solver);

/**
 * Get solutions for all problems
 *
 * @param solver  Batched solver
 * @param x       Output: primal solutions [batch_size * n]
 * @param z       Output: slack variables [batch_size * m] (can be NULL)
 * @param y       Output: dual solutions [batch_size * m] (can be NULL)
 * @return        0 on success
 */
c_int osqp_batch_get_solution(
    OSQPBatchSolver* solver,
    c_float* x,
    c_float* z,
    c_float* y
);

/**
 * Batched compute_rhs for KKT system (GPU)
 *
 * Computes RHS for all batch problems:
 *   xtilde part: sigma * x_prev - q
 *   ztilde part: z_prev - rho_inv * y
 *
 * @param solver  Batched solver
 * @return        0 on success
 */
c_int batch_compute_rhs(OSQPBatchSolver* solver);

/**
 * Batched update_x (GPU)
 *
 * For all problems:
 *   x = alpha * xtilde + (1 - alpha) * x_prev
 *   delta_x = x - x_prev
 *
 * @param solver  Batched solver
 * @return        0 on success
 */
c_int batch_update_x(OSQPBatchSolver* solver);

/**
 * Batched update_z with projection (GPU)
 *
 * For all problems:
 *   z = alpha * ztilde + (1 - alpha) * z_prev + rho_inv * y
 *   z = project(z, l, u)
 *
 * @param solver  Batched solver
 * @return        0 on success
 */
c_int batch_update_z(OSQPBatchSolver* solver);

/**
 * Batched update_y (GPU)
 *
 * For all problems:
 *   delta_y = alpha * ztilde + (1 - alpha) * z_prev - z
 *   delta_y = rho * delta_y
 *   y = y + delta_y
 *
 * @param solver  Batched solver
 * @return        0 on success
 */
c_int batch_update_y(OSQPBatchSolver* solver);

/**
 * Swap iterates: x <-> x_prev, z <-> z_prev
 *
 * @param solver  Batched solver
 */
void batch_swap_iterates(OSQPBatchSolver* solver);

#ifdef __cplusplus
}
#endif

#endif // AUXIL_BATCH_H
