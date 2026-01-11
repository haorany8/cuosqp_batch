/**
 * OSQP Batch API
 *
 * High-level API for solving multiple QP problems with the same structure
 * in parallel on GPU.
 *
 * Usage:
 *   1. Create batch solver with osqp_batch_create()
 *   2. Setup with osqp_batch_setup() using problem data
 *   3. Solve with osqp_batch_solve()
 *   4. Get solutions with osqp_batch_get_solution()
 *   5. Cleanup with osqp_batch_destroy()
 *
 * All problems in the batch must have:
 *   - Same P matrix (objective Hessian)
 *   - Same A matrix (constraint matrix)
 *   - Same structure (n, m dimensions)
 *
 * Each problem can have different:
 *   - q vector (linear cost)
 *   - l, u vectors (constraint bounds)
 *   - Initial iterates (warm start)
 */

#ifndef OSQP_API_BATCH_H
#define OSQP_API_BATCH_H

#ifdef __cplusplus
extern "C" {
#endif

#include "types.h"
#include "osqp_api_types.h"

// Forward declaration - opaque type for batch solver
typedef struct OSQPBatchSolverAPI_ OSQPBatchSolverAPI;

/**
 * Batch solver settings
 */
typedef struct {
    c_int   max_iter;           // Maximum ADMM iterations
    c_float eps_abs;            // Absolute tolerance
    c_float eps_rel;            // Relative tolerance
    c_float rho;                // ADMM penalty parameter
    c_float sigma;              // Regularization parameter
    c_float alpha;              // Relaxation parameter (1.0-2.0)
    c_int   check_termination;  // Check convergence every N iterations
    c_int   warm_start;         // Enable warm starting
    c_int   verbose;            // Print progress
} OSQPBatchSettings;

/**
 * Batch solver info (per-batch statistics)
 */
typedef struct {
    c_int   iter;               // Number of iterations
    c_int   status;             // Solver status
    c_float solve_time;         // Total solve time (seconds)
    c_float setup_time;         // Setup time (seconds)
} OSQPBatchInfo;

/**
 * Set default settings
 *
 * @param settings  Settings structure to initialize
 */
void osqp_batch_set_default_settings(OSQPBatchSettings* settings);

/**
 * Create a batch solver
 *
 * @param batch_size  Number of problems to solve in parallel
 * @return            Opaque solver handle (NULL on failure)
 */
OSQPBatchSolverAPI* osqp_batch_create(c_int batch_size);

/**
 * Setup the batch solver with problem data
 *
 * All problems share the same P, A matrices but can have different q, l, u.
 *
 * @param solver      Batch solver handle
 * @param P           Objective Hessian (n x n, upper triangular CSC)
 * @param q           Linear costs [batch_size * n]
 * @param A           Constraint matrix (m x n, CSC)
 * @param l           Lower bounds [batch_size * m]
 * @param u           Upper bounds [batch_size * m]
 * @param n           Primal dimension
 * @param m           Constraint dimension
 * @param settings    Solver settings (NULL for defaults)
 * @return            0 on success
 */
c_int osqp_batch_setup(
    OSQPBatchSolverAPI* solver,
    const csc* P,
    const c_float* q,
    const csc* A,
    const c_float* l,
    const c_float* u,
    c_int n,
    c_int m,
    const OSQPBatchSettings* settings
);

/**
 * Solve all QP problems in the batch
 *
 * @param solver  Batch solver handle
 * @return        0 on success
 */
c_int osqp_batch_solve(OSQPBatchSolverAPI* solver);

/**
 * Get solutions for all problems
 *
 * @param solver  Batch solver handle
 * @param x       Output: primal solutions [batch_size * n]
 * @param y       Output: dual solutions [batch_size * m] (can be NULL)
 * @return        0 on success
 */
c_int osqp_batch_get_solutions(
    OSQPBatchSolverAPI* solver,
    c_float* x,
    c_float* y
);

/**
 * Update linear costs for all problems
 *
 * @param solver  Batch solver handle
 * @param q       New linear costs [batch_size * n]
 * @return        0 on success
 */
c_int osqp_batch_update_q(OSQPBatchSolverAPI* solver, const c_float* q);

/**
 * Update constraint bounds for all problems
 *
 * @param solver  Batch solver handle
 * @param l       New lower bounds [batch_size * m]
 * @param u       New upper bounds [batch_size * m]
 * @return        0 on success
 */
c_int osqp_batch_update_bounds(
    OSQPBatchSolverAPI* solver,
    const c_float* l,
    const c_float* u
);

/**
 * Warm start all problems
 *
 * @param solver  Batch solver handle
 * @param x       Initial x [batch_size * n]
 * @param y       Initial y [batch_size * m]
 * @return        0 on success
 */
c_int osqp_batch_warm_start(
    OSQPBatchSolverAPI* solver,
    const c_float* x,
    const c_float* y
);

/**
 * Get solver info
 *
 * @param solver  Batch solver handle
 * @param info    Output: solver info
 * @return        0 on success
 */
c_int osqp_batch_get_info(OSQPBatchSolverAPI* solver, OSQPBatchInfo* info);

/**
 * Destroy the batch solver and free resources
 *
 * @param solver  Batch solver handle
 */
void osqp_batch_destroy(OSQPBatchSolverAPI* solver);

/**
 * Get the number of problems in the batch
 *
 * @param solver  Batch solver handle
 * @return        Batch size
 */
c_int osqp_batch_get_batch_size(const OSQPBatchSolverAPI* solver);

/**
 * Get problem dimensions
 *
 * @param solver  Batch solver handle
 * @param n       Output: primal dimension (can be NULL)
 * @param m       Output: constraint dimension (can be NULL)
 */
void osqp_batch_get_dimensions(const OSQPBatchSolverAPI* solver, c_int* n, c_int* m);

#ifdef __cplusplus
}
#endif

#endif // OSQP_API_BATCH_H
