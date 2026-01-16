/**
 * Batched Residual Computation on GPU
 *
 * Computes primal and dual residuals for convergence checking:
 *   - Primal residual: ||Ax - z||_inf
 *   - Dual residual: ||Px + q + A'y||_inf
 */

#ifndef RESIDUAL_BATCH_GPU_H
#define RESIDUAL_BATCH_GPU_H

#ifdef __cplusplus
extern "C" {
#endif

#include "types.h"
#include "spmv_batch_gpu.h"

/**
 * Workspace for batched residual computation
 */
typedef struct {
    int n;               // Primal dimension
    int m;               // Constraint dimension
    int batch_size;

    // GPU sparse matrices (shared across batch)
    GPUSparseMatrix* P;  // [n x n] upper triangular
    GPUSparseMatrix* A;  // [m x n]

    // Temporary buffers on device
    c_float* d_Ax;       // [batch_size * m] A*x
    c_float* d_Px;       // [batch_size * n] P*x
    c_float* d_Aty;      // [batch_size * n] A'*y
    c_float* d_temp;     // [batch_size * max(n,m)] for intermediate results

    // Residual outputs
    c_float* d_pri_res;  // [batch_size] primal residuals
    c_float* d_dua_res;  // [batch_size] dual residuals
} GPUBatchResidualWorkspace;

/**
 * Allocate residual workspace
 *
 * @param n           Primal dimension
 * @param m           Constraint dimension
 * @param batch_size  Number of problems
 * @return            Workspace (caller must free with free_gpu_residual_workspace)
 */
GPUBatchResidualWorkspace* alloc_gpu_residual_workspace(
    int n,
    int m,
    int batch_size
);

/**
 * Free residual workspace
 */
void free_gpu_residual_workspace(GPUBatchResidualWorkspace* ws);

/**
 * Set P and A matrices for residual computation
 *
 * @param ws          Residual workspace
 * @param h_Pp        Host: P column pointers [n+1]
 * @param h_Pi        Host: P row indices [nnz_P]
 * @param h_Px        Host: P values [nnz_P]
 * @param nnz_P       Number of nonzeros in P
 * @param h_Ap        Host: A column pointers [n+1]
 * @param h_Ai        Host: A row indices [nnz_A]
 * @param h_Ax        Host: A values [nnz_A]
 * @param nnz_A       Number of nonzeros in A
 * @return            0 on success
 */
int gpu_residual_set_matrices(
    GPUBatchResidualWorkspace* ws,
    const c_int* h_Pp, const c_int* h_Pi, const c_float* h_Px, c_int nnz_P,
    const c_int* h_Ap, const c_int* h_Ai, const c_float* h_Ax, c_int nnz_A
);

/**
 * Compute primal residual: ||Ax - z||_inf for all problems
 *
 * @param ws          Residual workspace
 * @param d_x         Device: primal variables [batch_size * n]
 * @param d_z         Device: slack variables [batch_size * m]
 * @param d_pri_res   Device: output residuals [batch_size]
 * @return            0 on success
 */
int gpu_batch_primal_residual(
    GPUBatchResidualWorkspace* ws,
    const c_float* d_x,
    const c_float* d_z,
    c_float* d_pri_res
);

/**
 * Compute dual residual: ||Px + q + A'y||_inf for all problems
 *
 * @param ws          Residual workspace
 * @param d_x         Device: primal variables [batch_size * n]
 * @param d_q         Device: linear costs [batch_size * n]
 * @param d_y         Device: dual variables [batch_size * m]
 * @param d_dua_res   Device: output residuals [batch_size]
 * @return            0 on success
 */
int gpu_batch_dual_residual(
    GPUBatchResidualWorkspace* ws,
    const c_float* d_x,
    const c_float* d_q,
    const c_float* d_y,
    c_float* d_dua_res
);

/**
 * Compute both primal and dual residuals
 *
 * @param ws          Residual workspace
 * @param d_x         Device: primal variables [batch_size * n]
 * @param d_z         Device: slack variables [batch_size * m]
 * @param d_y         Device: dual variables [batch_size * m]
 * @param d_q         Device: linear costs [batch_size * n]
 * @param d_pri_res   Device: output primal residuals [batch_size]
 * @param d_dua_res   Device: output dual residuals [batch_size]
 * @return            0 on success
 */
int gpu_batch_compute_residuals(
    GPUBatchResidualWorkspace* ws,
    const c_float* d_x,
    const c_float* d_z,
    const c_float* d_y,
    const c_float* d_q,
    c_float* d_pri_res,
    c_float* d_dua_res
);

/**
 * Compute primal and dual tolerances
 *
 * eps_pri = eps_abs * sqrt(m) + eps_rel * max(||Ax||, ||z||)
 * eps_dua = eps_abs * sqrt(n) + eps_rel * max(||Px||, ||A'y||, ||q||)
 *
 * @param ws          Residual workspace
 * @param d_x         Device: primal variables [batch_size * n]
 * @param d_z         Device: slack variables [batch_size * m]
 * @param d_y         Device: dual variables [batch_size * m]
 * @param d_q         Device: linear costs [batch_size * n]
 * @param eps_abs     Absolute tolerance
 * @param eps_rel     Relative tolerance
 * @param d_eps_pri   Device: output primal tolerances [batch_size]
 * @param d_eps_dua   Device: output dual tolerances [batch_size]
 * @return            0 on success
 */
int gpu_batch_compute_tolerances(
    GPUBatchResidualWorkspace* ws,
    const c_float* d_x,
    const c_float* d_z,
    const c_float* d_y,
    const c_float* d_q,
    c_float eps_abs,
    c_float eps_rel,
    c_float* d_eps_pri,
    c_float* d_eps_dua
);

/**
 * Copy residuals from device to host
 *
 * @param ws          Residual workspace
 * @param h_pri_res   Host: primal residuals [batch_size]
 * @param h_dua_res   Host: dual residuals [batch_size]
 * @return            0 on success
 */
int gpu_residual_copy_to_host(
    GPUBatchResidualWorkspace* ws,
    c_float* h_pri_res,
    c_float* h_dua_res
);

#ifdef __cplusplus
}
#endif

#endif // RESIDUAL_BATCH_GPU_H
