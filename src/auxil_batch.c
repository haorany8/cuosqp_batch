/**
 * Batched ADMM Auxiliary Functions - Implementation
 *
 * This file implements batch versions of the ADMM update functions.
 * The GPU kernels are in qdldl_batch_gpu.cu, this file provides the
 * high-level interface.
 */

#include "auxil_batch.h"
#include "qdldl_symbolic.h"
#include <stdlib.h>
#include <string.h>

OSQPBatchSolver* osqp_batch_alloc(c_int n, c_int m, c_int batch_size) {
    OSQPBatchSolver* solver = (OSQPBatchSolver*)calloc(1, sizeof(OSQPBatchSolver));
    if (!solver) return NULL;

    solver->n = n;
    solver->m = m;
    solver->batch_size = batch_size;

    // Allocate host buffers
    solver->h_q = (c_float*)calloc(batch_size * n, sizeof(c_float));
    solver->h_l = (c_float*)calloc(batch_size * m, sizeof(c_float));
    solver->h_u = (c_float*)calloc(batch_size * m, sizeof(c_float));
    solver->h_x = (c_float*)calloc(batch_size * n, sizeof(c_float));
    solver->h_z = (c_float*)calloc(batch_size * m, sizeof(c_float));
    solver->h_y = (c_float*)calloc(batch_size * m, sizeof(c_float));

    if (!solver->h_q || !solver->h_l || !solver->h_u ||
        !solver->h_x || !solver->h_z || !solver->h_y) {
        osqp_batch_free(solver);
        return NULL;
    }

    // GPU resources are allocated in osqp_batch_set_pattern
    solver->gpu_pattern = NULL;
    solver->base_ws = NULL;
    solver->admm_ws = NULL;
    solver->h_kkt_Ax = NULL;
    solver->nnz_KKT = 0;

    // Default ADMM parameters
    solver->rho = 0.1f;
    solver->sigma = 1e-6f;
    solver->alpha = 1.6f;

    solver->is_initialized = 0;
    solver->is_factorized = 0;

    return solver;
}

void osqp_batch_free(OSQPBatchSolver* solver) {
    if (!solver) return;

    // Free GPU resources
    if (solver->admm_ws) free_gpu_admm_workspace(solver->admm_ws);
    if (solver->base_ws) free_gpu_workspace(solver->base_ws);
    if (solver->gpu_pattern) free_gpu_pattern(solver->gpu_pattern);

    // Free host buffers
    if (solver->h_q) free(solver->h_q);
    if (solver->h_l) free(solver->h_l);
    if (solver->h_u) free(solver->h_u);
    if (solver->h_x) free(solver->h_x);
    if (solver->h_z) free(solver->h_z);
    if (solver->h_y) free(solver->h_y);
    if (solver->h_kkt_Ax) free(solver->h_kkt_Ax);

    free(solver);
}

c_int osqp_batch_set_pattern(OSQPBatchSolver* solver, const void* pattern) {
    if (!solver || !pattern) return -1;

    // Copy pattern to GPU
    solver->gpu_pattern = copy_pattern_to_gpu(pattern);
    if (!solver->gpu_pattern) return -1;

    // Get KKT size from pattern
    const FactorPattern* p = (const FactorPattern*)pattern;
    solver->nnz_KKT = p->nnz_KKT;

    // Allocate KKT buffer
    solver->h_kkt_Ax = (c_float*)malloc(solver->nnz_KKT * sizeof(c_float));
    if (!solver->h_kkt_Ax) return -1;

    // Allocate base GPU workspace
    solver->base_ws = alloc_gpu_workspace(solver->gpu_pattern, solver->batch_size);
    if (!solver->base_ws) return -1;

    // Allocate ADMM workspace
    solver->admm_ws = alloc_gpu_admm_workspace(
        solver->base_ws,
        solver->n,
        solver->m,
        solver->batch_size
    );
    if (!solver->admm_ws) return -1;

    solver->is_initialized = 1;
    return 0;
}

c_int osqp_batch_setup_data(
    OSQPBatchSolver* solver,
    const c_float* q,
    const c_float* l,
    const c_float* u,
    const c_float* Ax,
    c_float rho,
    c_float sigma,
    c_float alpha
) {
    if (!solver || !solver->is_initialized) return -1;

    c_int n = solver->n;
    c_int m = solver->m;
    c_int batch_size = solver->batch_size;

    // Copy data to host buffers
    memcpy(solver->h_q, q, batch_size * n * sizeof(c_float));
    memcpy(solver->h_l, l, batch_size * m * sizeof(c_float));
    memcpy(solver->h_u, u, batch_size * m * sizeof(c_float));
    memcpy(solver->h_kkt_Ax, Ax, solver->nnz_KKT * sizeof(c_float));

    solver->rho = rho;
    solver->sigma = sigma;
    solver->alpha = alpha;

    // Copy problem data to GPU
    int ret = gpu_admm_copy_problem_data(
        solver->admm_ws,
        solver->h_q,
        solver->h_l,
        solver->h_u,
        NULL,  // rho_vec (scalar rho for now)
        rho,
        sigma,
        alpha
    );
    if (ret != 0) return ret;

    // Copy KKT values to GPU (broadcast to all batch elements)
    ret = gpu_batch_factor_broadcast_host(
        solver->gpu_pattern,
        solver->base_ws,
        solver->h_kkt_Ax,
        batch_size
    );

    solver->is_factorized = 0;  // Need to refactorize
    return ret;
}

c_int osqp_batch_warm_start_internal(
    OSQPBatchSolver* solver,
    const c_float* x,
    const c_float* z,
    const c_float* y
) {
    if (!solver || !solver->is_initialized) return -1;

    c_int n = solver->n;
    c_int m = solver->m;
    c_int batch_size = solver->batch_size;

    // Copy to host buffers
    if (x) memcpy(solver->h_x, x, batch_size * n * sizeof(c_float));
    if (z) memcpy(solver->h_z, z, batch_size * m * sizeof(c_float));
    if (y) memcpy(solver->h_y, y, batch_size * m * sizeof(c_float));

    // Copy to GPU
    return gpu_admm_copy_initial_iterates(
        solver->admm_ws,
        solver->h_x,
        solver->h_z,
        solver->h_y
    );
}

c_int osqp_batch_cold_start(OSQPBatchSolver* solver) {
    if (!solver || !solver->is_initialized) return -1;

    c_int n = solver->n;
    c_int m = solver->m;
    c_int batch_size = solver->batch_size;

    // Zero out host buffers
    memset(solver->h_x, 0, batch_size * n * sizeof(c_float));
    memset(solver->h_z, 0, batch_size * m * sizeof(c_float));
    memset(solver->h_y, 0, batch_size * m * sizeof(c_float));

    // Copy to GPU
    return gpu_admm_copy_initial_iterates(
        solver->admm_ws,
        solver->h_x,
        solver->h_z,
        solver->h_y
    );
}

c_int osqp_batch_factorize(OSQPBatchSolver* solver) {
    if (!solver || !solver->is_initialized) return -1;

    int ret = gpu_batch_factor_device(
        solver->gpu_pattern,
        solver->base_ws,
        solver->batch_size
    );

    if (ret == 0) {
        solver->is_factorized = 1;
    }

    return ret;
}

c_int osqp_batch_update_factorization(
    OSQPBatchSolver* solver,
    const c_float* new_KKTx
) {
    if (!solver || !solver->is_initialized || !new_KKTx) return -1;

    // Update host KKT values
    memcpy(solver->h_kkt_Ax, new_KKTx, solver->nnz_KKT * sizeof(c_float));

    // Broadcast new values to GPU
    int ret = gpu_batch_factor_broadcast_host(
        solver->gpu_pattern,
        solver->base_ws,
        solver->h_kkt_Ax,
        solver->batch_size
    );
    if (ret != 0) return ret;

    // Mark as needing refactorization
    solver->is_factorized = 0;

    // Refactorize
    return osqp_batch_factorize(solver);
}

c_int osqp_batch_admm_iteration(OSQPBatchSolver* solver, c_int admm_iter) {
    if (!solver || !solver->is_initialized) return -1;

    // Ensure factorization is done
    if (!solver->is_factorized) {
        int ret = osqp_batch_factorize(solver);
        if (ret != 0) return ret;
    }

    return gpu_batch_admm_iteration(
        solver->gpu_pattern,
        solver->admm_ws,
        admm_iter
    );
}

c_int osqp_batch_get_solution(
    OSQPBatchSolver* solver,
    c_float* x,
    c_float* z,
    c_float* y
) {
    if (!solver || !solver->is_initialized) return -1;

    // Copy from GPU to host buffers
    int ret = gpu_admm_copy_solution(
        solver->admm_ws,
        solver->h_x,
        solver->h_z,
        solver->h_y
    );
    if (ret != 0) return ret;

    // Copy to user buffers
    c_int n = solver->n;
    c_int m = solver->m;
    c_int batch_size = solver->batch_size;

    if (x) memcpy(x, solver->h_x, batch_size * n * sizeof(c_float));
    if (z) memcpy(z, solver->h_z, batch_size * m * sizeof(c_float));
    if (y) memcpy(y, solver->h_y, batch_size * m * sizeof(c_float));

    return 0;
}

// Wrapper functions that call the GPU implementations

c_int batch_compute_rhs(OSQPBatchSolver* solver) {
    if (!solver || !solver->is_initialized) return -1;
    return gpu_batch_compute_rhs(solver->gpu_pattern, solver->admm_ws);
}

c_int batch_update_x(OSQPBatchSolver* solver) {
    if (!solver || !solver->is_initialized) return -1;
    return gpu_batch_update_x(solver->admm_ws);
}

c_int batch_update_z(OSQPBatchSolver* solver) {
    if (!solver || !solver->is_initialized) return -1;
    return gpu_batch_update_z(solver->admm_ws);
}

c_int batch_update_y(OSQPBatchSolver* solver) {
    if (!solver || !solver->is_initialized) return -1;
    return gpu_batch_update_y(solver->admm_ws);
}

void batch_swap_iterates(OSQPBatchSolver* solver) {
    if (!solver || !solver->is_initialized) return;
    gpu_batch_swap_iterates(solver->admm_ws);
}
