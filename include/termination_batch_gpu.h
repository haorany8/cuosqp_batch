/**
 * Batched Termination Checking on GPU
 *
 * Checks convergence criteria for batched OSQP problems:
 *   - pri_res <= eps_pri (primal feasibility)
 *   - dua_res <= eps_dua (dual feasibility)
 *
 * Tracks per-problem status and counts converged problems.
 */

#ifndef TERMINATION_BATCH_GPU_H
#define TERMINATION_BATCH_GPU_H

#ifdef __cplusplus
extern "C" {
#endif

#include "types.h"
#include "osqp_api_constants.h"

/**
 * Workspace for batched termination checking
 */
typedef struct {
    int batch_size;

    // Device arrays
    c_int* d_status;          // [batch_size] per-problem status
    c_int* d_converged;       // [batch_size] convergence flags (0 or 1)
    c_float* d_pri_res;       // [batch_size] primal residuals
    c_float* d_dua_res;       // [batch_size] dual residuals
    c_float* d_eps_pri;       // [batch_size] primal tolerances
    c_float* d_eps_dua;       // [batch_size] dual tolerances

    // Host mirror for status tracking
    c_int* h_status;          // [batch_size] copy of d_status
    c_int* h_converged;       // [batch_size] copy of d_converged
    c_float* h_pri_res;       // [batch_size] copy of d_pri_res
    c_float* h_dua_res;       // [batch_size] copy of d_dua_res

    // Aggregate statistics
    c_int num_converged;      // Count of converged problems
    c_int num_pri_infeas;     // Count of primal infeasible problems
    c_int num_dua_infeas;     // Count of dual infeasible problems
} GPUBatchTermination;

/**
 * Allocate termination workspace
 *
 * @param batch_size  Number of problems
 * @return            Workspace (caller must free with free_gpu_termination_workspace)
 */
GPUBatchTermination* alloc_gpu_termination_workspace(int batch_size);

/**
 * Free termination workspace
 */
void free_gpu_termination_workspace(GPUBatchTermination* ws);

/**
 * Initialize termination status for all problems
 * Sets all statuses to OSQP_UNSOLVED
 *
 * @param ws          Termination workspace
 * @return            0 on success
 */
int gpu_termination_init(GPUBatchTermination* ws);

/**
 * Check convergence for all problems
 *
 * A problem is converged if:
 *   pri_res[b] <= eps_pri[b] AND dua_res[b] <= eps_dua[b]
 *
 * @param ws          Termination workspace
 * @param d_pri_res   Device: primal residuals [batch_size]
 * @param d_dua_res   Device: dual residuals [batch_size]
 * @param d_eps_pri   Device: primal tolerances [batch_size]
 * @param d_eps_dua   Device: dual tolerances [batch_size]
 * @param all_converged  Output: 1 if all problems converged, 0 otherwise
 * @return            0 on success
 */
int gpu_batch_check_convergence(
    GPUBatchTermination* ws,
    const c_float* d_pri_res,
    const c_float* d_dua_res,
    const c_float* d_eps_pri,
    const c_float* d_eps_dua,
    int* all_converged
);

/**
 * Check convergence with scalar tolerances (simpler interface)
 *
 * @param ws          Termination workspace
 * @param d_pri_res   Device: primal residuals [batch_size]
 * @param d_dua_res   Device: dual residuals [batch_size]
 * @param eps_pri     Scalar primal tolerance (same for all)
 * @param eps_dua     Scalar dual tolerance (same for all)
 * @param all_converged  Output: 1 if all problems converged, 0 otherwise
 * @return            0 on success
 */
int gpu_batch_check_convergence_scalar(
    GPUBatchTermination* ws,
    const c_float* d_pri_res,
    const c_float* d_dua_res,
    c_float eps_pri,
    c_float eps_dua,
    int* all_converged
);

/**
 * Mark max iterations reached for unconverged problems
 * Sets status to OSQP_MAX_ITER_REACHED for problems not yet converged
 *
 * @param ws          Termination workspace
 * @return            0 on success
 */
int gpu_termination_mark_max_iter(GPUBatchTermination* ws);

/**
 * Update status to OSQP_SOLVED for converged problems
 *
 * @param ws          Termination workspace
 * @return            0 on success
 */
int gpu_termination_mark_solved(GPUBatchTermination* ws);

/**
 * Copy termination data from device to host
 *
 * @param ws          Termination workspace
 * @return            0 on success
 */
int gpu_termination_sync_to_host(GPUBatchTermination* ws);

/**
 * Get number of converged problems (after sync)
 *
 * @param ws          Termination workspace
 * @return            Number of converged problems
 */
c_int gpu_termination_get_num_converged(GPUBatchTermination* ws);

/**
 * Get status for a specific problem (after sync)
 *
 * @param ws          Termination workspace
 * @param idx         Problem index
 * @return            Status code (OSQP_SOLVED, OSQP_MAX_ITER_REACHED, etc.)
 */
c_int gpu_termination_get_status(GPUBatchTermination* ws, int idx);

/**
 * Get residuals for a specific problem (after sync)
 *
 * @param ws          Termination workspace
 * @param idx         Problem index
 * @param pri_res     Output: primal residual (can be NULL)
 * @param dua_res     Output: dual residual (can be NULL)
 */
void gpu_termination_get_residuals(
    GPUBatchTermination* ws,
    int idx,
    c_float* pri_res,
    c_float* dua_res
);

#ifdef __cplusplus
}
#endif

#endif // TERMINATION_BATCH_GPU_H
