/**
 *  QDLDL CUDA Batch Interface
 *
 *  This file provides the C interface for the batched CUDA QDLDL solver.
 *  It combines:
 *  - qdldl_interface: For initial CPU solver setup and pattern extraction
 *  - qdldl_symbolic: For recording the factorization pattern
 *  - qdldl_batch_gpu: For GPU-accelerated batched factorization and solves
 */

#include "qdldl_cuda_batch_interface.h"
#include "qdldl_interface.h"
#include "qdldl_symbolic.h"
#include "qdldl_batch_gpu.h"
#include "kkt.h"
#include "glob_opts.h"

#include <string.h>


/*******************************************************************************
 *                         Helper Functions                                    *
 *******************************************************************************/

/**
 * Update KKT matrix values on host with P values
 */
static void update_KKT_P_host(c_float       *KKT_x,
                              const c_float *Px,
                              const c_int   *Pp,
                              c_int          n,
                              const c_int   *PtoKKT,
                              c_float        sigma,
                              const c_int   *Pdiag_idx,
                              c_int          Pdiag_n) {
    c_int i, j;

    // Update P elements in KKT
    for (j = 0; j < n; j++) {
        for (i = Pp[j]; i < Pp[j + 1]; i++) {
            KKT_x[PtoKKT[i]] = Px[i];
        }
    }

    // Add sigma to diagonal
    for (i = 0; i < Pdiag_n; i++) {
        KKT_x[Pdiag_idx[i]] += sigma;
    }
}

/**
 * Update KKT matrix values on host with A values
 */
static void update_KKT_A_host(c_float       *KKT_x,
                              const c_float *Ax,
                              const c_int   *Ap,
                              c_int          n,
                              const c_int   *AtoKKT) {
    c_int i, j;

    for (j = 0; j < n; j++) {
        for (i = Ap[j]; i < Ap[j + 1]; i++) {
            KKT_x[AtoKKT[i]] = Ax[i];
        }
    }
}

/**
 * Update KKT matrix values on host with rho values
 */
static void update_KKT_rho_host(c_float       *KKT_x,
                                const c_float *rho_inv_vec,
                                c_float        rho_inv,
                                const c_int   *rhotoKKT,
                                c_int          m) {
    c_int i;

    for (i = 0; i < m; i++) {
        c_float val = rho_inv_vec ? -rho_inv_vec[i] : -rho_inv;
        KKT_x[rhotoKKT[i]] = val;
    }
}


/*******************************************************************************
 *                         Interface Implementation                            *
 *******************************************************************************/

c_int init_linsys_solver_qdldl_cuda_batch(
    qdldl_cuda_batch_solver **sp,
    const OSQPMatrix         *P,
    const OSQPMatrix         *A,
    const OSQPVectorf        *rho_vec,
    OSQPSettings             *settings,
    c_int                     batch_size
) {
    c_int i;
    qdldl_cuda_batch_solver *s = OSQP_NULL;
    qdldl_solver *cpu_solver = OSQP_NULL;
    FactorPattern *pattern = OSQP_NULL;
    GPUFactorPattern *gpu_pattern = OSQP_NULL;
    GPUBatchWorkspace *gpu_workspace = OSQP_NULL;

    // Create a CPU solver to get the initial factorization pattern
    c_int status = init_linsys_solver_qdldl(&cpu_solver, P, A, rho_vec, settings, 0);
    if (status != 0) {
        return status;
    }

    // Extract the factorization pattern
    pattern = record_pattern_from_qdldl_solver(cpu_solver);
    if (!pattern) {
        free_linsys_solver_qdldl(cpu_solver);
        return OSQP_LINSYS_SOLVER_INIT_ERROR;
    }

    // Copy pattern to GPU
    gpu_pattern = copy_pattern_to_gpu(pattern);
    if (!gpu_pattern) {
        free_pattern(pattern);
        free_linsys_solver_qdldl(cpu_solver);
        return OSQP_LINSYS_SOLVER_INIT_ERROR;
    }

    // Allocate GPU workspace
    gpu_workspace = alloc_gpu_workspace(gpu_pattern, batch_size);
    if (!gpu_workspace) {
        free_gpu_pattern(gpu_pattern);
        free_pattern(pattern);
        free_linsys_solver_qdldl(cpu_solver);
        return OSQP_LINSYS_SOLVER_INIT_ERROR;
    }

    // Allocate the batch solver structure
    s = (qdldl_cuda_batch_solver *)c_calloc(1, sizeof(qdldl_cuda_batch_solver));
    if (!s) {
        free_gpu_workspace(gpu_workspace);
        free_gpu_pattern(gpu_pattern);
        free_pattern(pattern);
        free_linsys_solver_qdldl(cpu_solver);
        return OSQP_LINSYS_SOLVER_INIT_ERROR;
    }

    *sp = s;

    // Set batch parameters
    s->batch_size = batch_size;
    s->n = cpu_solver->n;
    s->m = cpu_solver->m;
    s->n_plus_m = s->n + s->m;
    s->sigma = settings->sigma;
    s->rho_inv = 1.0 / settings->rho;

    // Pattern sizes
    s->nnz_KKT = pattern->nnz_KKT;
    s->nnz_L = pattern->nnz_L;
    s->nnz_P = OSQPMatrix_get_nz(P);
    s->nnz_A = OSQPMatrix_get_nz(A);
    s->Pdiag_n = cpu_solver->Pdiag_n;

    // Function pointers
    s->type = QDLDL_SOLVER;
    s->solve = &solve_linsys_qdldl_cuda_batch;
    s->solve_host = &solve_linsys_qdldl_cuda_batch_host;
    s->free = &free_linsys_solver_qdldl_cuda_batch;
    s->update_matrices_batch = &update_linsys_solver_matrices_qdldl_cuda_batch;
    s->update_rho_vec_batch = &update_linsys_solver_rho_vec_qdldl_cuda_batch;

    // Store internal GPU structures
    s->gpu_pattern = gpu_pattern;
    s->gpu_workspace = gpu_workspace;

    // Copy pointers from GPU pattern to structure fields
    s->d_KKT_p = gpu_pattern->d_Ap;
    s->d_KKT_i = gpu_pattern->d_Ai;
    s->d_L_p = gpu_pattern->d_Lp;
    s->d_L_i = gpu_pattern->d_Li;
    s->d_etree = gpu_pattern->d_etree;
    s->d_Lnz = gpu_pattern->d_Lnz;
    s->d_perm = gpu_pattern->d_P;

    // Copy pointers from GPU workspace to structure fields
    s->d_L_x_batch = gpu_workspace->d_Lx;
    s->d_D_batch = gpu_workspace->d_D;
    s->d_Dinv_batch = gpu_workspace->d_Dinv;
    s->d_work_batch = gpu_workspace->d_work;
    s->d_fwork_batch = gpu_workspace->d_fwork;
    s->d_iwork_batch = gpu_workspace->d_iwork;
    s->d_bwork_batch = gpu_workspace->d_bwork;

    // These are not in the standard workspace, set to NULL
    s->d_KKT_x_batch = OSQP_NULL;
    s->d_sol_batch = OSQP_NULL;
    s->d_rho_inv_vec_batch = OSQP_NULL;
    s->d_PtoKKT = OSQP_NULL;
    s->d_AtoKKT = OSQP_NULL;
    s->d_rhotoKKT = OSQP_NULL;
    s->d_Pdiag_idx = OSQP_NULL;

    // Save host copies of mapping indices
    s->h_perm = (c_int *)c_malloc(s->n_plus_m * sizeof(c_int));
    s->h_PtoKKT = (c_int *)c_malloc(s->nnz_P * sizeof(c_int));
    s->h_AtoKKT = (c_int *)c_malloc(s->nnz_A * sizeof(c_int));
    s->h_rhotoKKT = (c_int *)c_malloc(s->m * sizeof(c_int));
    s->h_Pdiag_idx = (c_int *)c_malloc(s->Pdiag_n * sizeof(c_int));
    s->h_KKT_x = (c_float *)c_malloc(s->nnz_KKT * sizeof(c_float));

    if (!s->h_perm || !s->h_PtoKKT || !s->h_AtoKKT ||
        !s->h_rhotoKKT || !s->h_Pdiag_idx || !s->h_KKT_x) {
        free_linsys_solver_qdldl_cuda_batch(s);
        *sp = OSQP_NULL;
        free_pattern(pattern);
        free_linsys_solver_qdldl(cpu_solver);
        return OSQP_LINSYS_SOLVER_INIT_ERROR;
    }

    // Copy from CPU solver
    memcpy(s->h_perm, pattern->P, s->n_plus_m * sizeof(c_int));
    memcpy(s->h_PtoKKT, cpu_solver->PtoKKT, s->nnz_P * sizeof(c_int));
    memcpy(s->h_AtoKKT, cpu_solver->AtoKKT, s->nnz_A * sizeof(c_int));
    memcpy(s->h_rhotoKKT, cpu_solver->rhotoKKT, s->m * sizeof(c_int));
    memcpy(s->h_Pdiag_idx, cpu_solver->Pdiag_idx, s->Pdiag_n * sizeof(c_int));
    memcpy(s->h_KKT_x, cpu_solver->KKT->x, s->nnz_KKT * sizeof(c_float));

    // Initialize rho_inv_vec if provided
    if (rho_vec) {
        c_float *rhov = OSQPVectorf_data(rho_vec);
        c_float *h_rho_inv_vec = (c_float *)c_malloc(s->m * sizeof(c_float));
        if (!h_rho_inv_vec) {
            free_linsys_solver_qdldl_cuda_batch(s);
            *sp = OSQP_NULL;
            free_pattern(pattern);
            free_linsys_solver_qdldl(cpu_solver);
            return OSQP_LINSYS_SOLVER_INIT_ERROR;
        }

        for (i = 0; i < s->m; i++) {
            h_rho_inv_vec[i] = 1.0 / rhov[i];
        }

        // Update the KKT template with rho
        update_KKT_rho_host(s->h_KKT_x, h_rho_inv_vec, s->rho_inv,
                           s->h_rhotoKKT, s->m);

        c_free(h_rho_inv_vec);
    }

    // Perform initial factorization for all batch elements using the template KKT
    // For initialization, all batch elements have the same KKT values
    // Use broadcast version which copies single KKT to all batch elements
    status = gpu_batch_factor_broadcast_host(gpu_pattern, gpu_workspace,
                                              (const QDLDL_float *)s->h_KKT_x, batch_size);

    // Clean up CPU resources
    free_pattern(pattern);
    free_linsys_solver_qdldl(cpu_solver);

    if (status != 0) {
        free_linsys_solver_qdldl_cuda_batch(s);
        *sp = OSQP_NULL;
        return OSQP_LINSYS_SOLVER_INIT_ERROR;
    }

    return 0;
}


c_int solve_linsys_qdldl_cuda_batch(
    qdldl_cuda_batch_solver *s,
    c_float                 *d_b_batch,
    c_int                    admm_iter
) {
    (void)admm_iter;

    GPUFactorPattern *gpu_pattern = (GPUFactorPattern *)s->gpu_pattern;
    GPUBatchWorkspace *gpu_workspace = (GPUBatchWorkspace *)s->gpu_workspace;

    if (!gpu_pattern || !gpu_workspace) {
        return -1;
    }

    // The b_batch vector is [batch_size * n_plus_m]
    // gpu_batch_solve expects [batch_size * n] (just the KKT dimension)
    // Since n_plus_m == KKT dimension, this should work
    return gpu_batch_solve(gpu_pattern, gpu_workspace, d_b_batch, s->batch_size);
}


c_int solve_linsys_qdldl_cuda_batch_host(
    qdldl_cuda_batch_solver *s,
    c_float                 *h_b_batch,
    c_int                    admm_iter
) {
    (void)admm_iter;

    GPUFactorPattern *gpu_pattern = (GPUFactorPattern *)s->gpu_pattern;
    GPUBatchWorkspace *gpu_workspace = (GPUBatchWorkspace *)s->gpu_workspace;

    if (!gpu_pattern || !gpu_workspace) {
        return -1;
    }

    // Use host memory interface - copies to GPU, solves, copies back
    return gpu_batch_solve_host(gpu_pattern, gpu_workspace,
                                (QDLDL_float *)h_b_batch, s->batch_size);
}


c_int update_linsys_solver_matrices_qdldl_cuda_batch(
    qdldl_cuda_batch_solver *s,
    const c_float           *d_Px_batch,
    const c_float           *d_Ax_batch
) {
    (void)d_Px_batch;
    (void)d_Ax_batch;

    // This function updates the KKT matrices and refactorizes
    // For a proper implementation, we need to:
    // 1. Update KKT values on GPU using d_Px_batch and d_Ax_batch
    // 2. Refactorize using gpu_batch_factor

    GPUFactorPattern *gpu_pattern = (GPUFactorPattern *)s->gpu_pattern;
    GPUBatchWorkspace *gpu_workspace = (GPUBatchWorkspace *)s->gpu_workspace;

    if (!gpu_pattern || !gpu_workspace) {
        return -1;
    }

    // Note: This requires d_KKT_x_batch to be allocated and updated
    // The current implementation is incomplete as we don't have the
    // GPU kernels for updating KKT from P and A batches

#ifdef PRINTING
    c_eprint("update_linsys_solver_matrices_qdldl_cuda_batch: Not fully implemented");
#endif

    return -1;
}


c_int update_linsys_solver_rho_vec_qdldl_cuda_batch(
    qdldl_cuda_batch_solver *s,
    const c_float           *d_rho_vec_batch,
    c_float                  rho_sc
) {
    (void)d_rho_vec_batch;

    // Update rho_inv
    s->rho_inv = 1.0 / rho_sc;

    // This function updates the rho values in KKT and refactorizes
    // For a proper implementation, we need to:
    // 1. Update KKT diagonal on GPU with new rho values
    // 2. Refactorize using gpu_batch_factor

    GPUFactorPattern *gpu_pattern = (GPUFactorPattern *)s->gpu_pattern;
    GPUBatchWorkspace *gpu_workspace = (GPUBatchWorkspace *)s->gpu_workspace;

    if (!gpu_pattern || !gpu_workspace) {
        return -1;
    }

#ifdef PRINTING
    c_eprint("update_linsys_solver_rho_vec_qdldl_cuda_batch: Not fully implemented");
#endif

    return -1;
}


void free_linsys_solver_qdldl_cuda_batch(qdldl_cuda_batch_solver *s) {
    if (!s) return;

    // Free GPU structures
    if (s->gpu_workspace) {
        free_gpu_workspace((GPUBatchWorkspace *)s->gpu_workspace);
        s->gpu_workspace = OSQP_NULL;
    }

    if (s->gpu_pattern) {
        free_gpu_pattern((GPUFactorPattern *)s->gpu_pattern);
        s->gpu_pattern = OSQP_NULL;
    }

    // Free host copies
    if (s->h_perm) c_free(s->h_perm);
    if (s->h_PtoKKT) c_free(s->h_PtoKKT);
    if (s->h_AtoKKT) c_free(s->h_AtoKKT);
    if (s->h_rhotoKKT) c_free(s->h_rhotoKKT);
    if (s->h_Pdiag_idx) c_free(s->h_Pdiag_idx);
    if (s->h_KKT_x) c_free(s->h_KKT_x);

    // Clear pointers that were copied from GPU structures
    s->d_KKT_p = OSQP_NULL;
    s->d_KKT_i = OSQP_NULL;
    s->d_L_p = OSQP_NULL;
    s->d_L_i = OSQP_NULL;
    s->d_etree = OSQP_NULL;
    s->d_Lnz = OSQP_NULL;
    s->d_perm = OSQP_NULL;
    s->d_L_x_batch = OSQP_NULL;
    s->d_D_batch = OSQP_NULL;
    s->d_Dinv_batch = OSQP_NULL;
    s->d_work_batch = OSQP_NULL;
    s->d_fwork_batch = OSQP_NULL;
    s->d_iwork_batch = OSQP_NULL;
    s->d_bwork_batch = OSQP_NULL;

    c_free(s);
}


size_t get_gpu_memory_usage_qdldl_cuda_batch(const qdldl_cuda_batch_solver *s) {
    if (!s) return 0;

    QDLDL_int n = s->n_plus_m;
    QDLDL_int nnz_KKT = s->nnz_KKT;
    QDLDL_int nnz_L = s->nnz_L;
    c_int batch_size = s->batch_size;

    // Pattern (shared)
    size_t pattern_size =
        (n + 1) * sizeof(QDLDL_int) +      // Ap
        nnz_KKT * sizeof(QDLDL_int) +      // Ai
        (n + 1) * sizeof(QDLDL_int) +      // Lp
        nnz_L * sizeof(QDLDL_int) +        // Li
        n * sizeof(QDLDL_int) +            // etree
        n * sizeof(QDLDL_int) +            // Lnz
        n * sizeof(QDLDL_int);             // P

    // Workspace (per batch)
    size_t workspace_size =
        batch_size * nnz_L * sizeof(QDLDL_float) +    // Lx
        batch_size * n * sizeof(QDLDL_float) +        // D
        batch_size * n * sizeof(QDLDL_float) +        // Dinv
        batch_size * n * sizeof(QDLDL_float) +        // work
        batch_size * n * sizeof(QDLDL_float) +        // fwork
        batch_size * 3 * n * sizeof(QDLDL_int) +      // iwork
        batch_size * n * sizeof(QDLDL_bool);          // bwork

    return pattern_size + workspace_size;
}
